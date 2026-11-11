"""Multi-view Music2Emo fitting with the retained reproduction loss protocol.

The original mixture-only trainer stays unchanged. This independent loop keeps
its optimizer, epoch RNG, minimum-delta stopping, top-five ranking, and atomic
resume rules, while accepting a multi-view model and dataset factory.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging

import numpy as np
import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from stemfusion.configuration import configuration_dict
from stemfusion.training.reporting import file_checksum, fingerprint, finish_artifacts, reusable_artifacts, write_json
from stemfusion.training.trainer import seed_worker, set_seed

LOGGER = logging.getLogger(__name__)


def epoch_seed(seed, epoch):
    return int.from_bytes(hashlib.sha256(f"music2emo-reproduction:{seed}:{epoch}".encode()).digest()[:4], "little")


def weighted_bce(prevalence, device):
    probabilities = torch.from_numpy(np.asarray(prevalence).copy()).to(device)
    positive = torch.tensor(1., device=device) / probabilities
    weight = torch.tensor(2., device=device) / (torch.tensor(1., device=device) + positive)
    return torch.nn.BCEWithLogitsLoss(pos_weight=positive, weight=weight)


def _inputs(batch, device):
    return {name: batch[name].to(device) for name in ("x_mert", "x_chord_root", "x_chord_attr", "x_key")}


def threshold_metrics(arrays, threshold):
    targets = np.asarray(arrays["targets"], dtype=bool)
    predictions = np.asarray(arrays["probabilities"]) >= threshold
    true_positive = (predictions & targets).sum(axis=0)
    false_positive = (predictions & ~targets).sum(axis=0)
    false_negative = (~predictions & targets).sum(axis=0)
    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    return {"accuracy": float((predictions == targets).mean()),
            "subset_accuracy": float((predictions == targets).all(axis=1).mean()),
            "precision_macro": float(ratio(true_positive, true_positive + false_positive).mean()),
            "recall_macro": float(ratio(true_positive, true_positive + false_negative).mean()),
            "f1_macro": float(ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative).mean())}


def select_threshold(validation_arrays):
    """One global threshold; maximize validation macro F1, ties to lowest grid value."""
    values = np.asarray(validation_arrays["probabilities"])
    labels = np.asarray(validation_arrays["targets"])
    if values.ndim != 2 or values.shape != labels.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected aligned finite validation probabilities and targets.")
    if not ((values >= 0) & (values <= 1)).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Threshold tuning requires probabilities and binary targets.")
    grid = np.linspace(0.0, 1.0, 101)
    scores = [threshold_metrics(validation_arrays, threshold)["f1_macro"] for threshold in grid]
    return float(grid[int(np.argmax(scores))])


def _evaluate(model, records, criterion, config, device, *, loader, predictions=False, task=None):
    model.eval()
    total, ids, logits, targets = 0., [], [], []
    with torch.inference_mode():
        for batch in loader(records, config, training=False, seed=0):
            output = model(_inputs(batch, device))
            labels = batch["labels"].to(device)
            loss = criterion(output, labels)
            if not torch.isfinite(loss) or not torch.isfinite(output).all():
                raise ValueError("Non-finite validation/test outputs.")
            total += float(loss) * len(labels)
            if predictions:
                ids.extend(batch["track_id"])
                logits.append(output.cpu())
                targets.append(labels.cpu())
    result = {"loss": total / len(records)}
    if not predictions:
        return result
    from torchmetrics.functional.classification import multilabel_auroc, multilabel_average_precision

    values, labels = torch.cat(logits), torch.cat(targets)
    probabilities = values.sigmoid()
    if task is None:
        roc = multilabel_auroc(probabilities, labels.int(), num_labels=56, average=None, thresholds=None)
        pr = multilabel_average_precision(probabilities, labels.int(), num_labels=56, average=None, thresholds=None)
        if not torch.isfinite(roc).all() or not torch.isfinite(pr).all():
            raise ValueError("Undefined macro AUC; every tag must have positive and negative examples.")
        result.update(roc_auc_macro=float(roc.mean()), pr_auc_macro=float(pr.mean()),
                      per_tag_roc_auc=roc.tolist(), per_tag_average_precision=pr.tolist())
    else:
        result.update(task.ranking_metrics(probabilities, labels))
    return result, {"track_ids": np.asarray(ids), "targets": labels.numpy().astype(np.uint8),
                    "logits": values.numpy(), "probabilities": probabilities.numpy()}


def _save_torch(path, payload):
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_predictions(path, arrays):
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_early_stopping(state, loss, epoch, *, patience, min_delta):
    """Match Lightning's min-mode improvement and validation-check counting.

    This significant-improvement score is deliberately separate from the
    unrestricted minimum used by ModelCheckpoint.
    """
    if loss + min_delta < state["best_score"]:
        state["best_score"], state["wait_count"] = loss, 0
    else:
        state["wait_count"] += 1
        if state["wait_count"] >= patience:
            state["stopped_epoch"] = epoch


def _rank_checkpoint(directory, entries, *, epoch, loss, signature, model, optimizer, early_stopping):
    """Keep the five lowest validation losses, retaining earlier exact ties."""
    if len(entries) == 5 and loss >= entries[-1]["validation_loss"]:
        return entries, []
    relative = f"checkpoints/epoch_{epoch:04d}.pt"
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_torch(path, {"signature": signature, "epoch": epoch, "validation_loss": loss,
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "early_stopping": dict(early_stopping)})
    entry = {"epoch": epoch, "validation_loss": loss, "file": relative, "sha256": file_checksum(path)}
    ranked = sorted([*entries, entry], key=lambda row: (row["validation_loss"], row["epoch"]))
    return ranked[:5], ranked[5:]


def fit_model(config, data, seed, *, model_factory, dataset_factory, implementation, protocol_updates, epoch_callback=None, task=None):
    """Use released stopping/checkpoint rules under the paper's epoch cap."""
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu.")
    protocol = {"author_commit": "c5f7984f78d9928b841f25bd05a47377c12c58ca", "task": "jamendo_mid_fusion",
                "source": "mixture", "multitask": False, "knowledge_distillation": False,
                "epochs": config.epochs, "batch_size": config.batch_size, "learning_rate": config.learning_rate,
                "optimizer": "Adam", "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.,
                "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": config.early_stopping_patience,
                                   "min_delta": config.early_stopping_min_delta, "check_every_n_epochs": 1},
                "save_top_k": 5, "save_last": True,
                "scheduler": "none", "selection": "min_validation_weighted_bce",
                "ties": "earliest_epoch", "device": str(device), "num_workers": config.num_workers,
                "matmul_precision": "medium", "attention_backend": "math", "deterministic_algorithms": True,
                "sampling": "epoch_seeded_conditional_uniform_consecutive_segment_mean",
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "torchmetrics")}}
    protocol.update(protocol_updates)
    if task is not None:
        protocol.update(task.protocol())
    evaluation_options = {} if task is None else {"task": task}
    signature = fingerprint({"seed": seed, "protocol": protocol, "data": data.data_fingerprint, "implementation": implementation})
    directory = config.output_dir / f"seed_{seed}"
    if not config.overwrite and config.resume and (directory / "run_manifest.json").exists():
        if reusable_artifacts(directory, signature):
            return json.loads((directory / "result.json").read_text())
    last_path = directory / "last.pt"
    existing = directory.exists() and any(directory.iterdir())
    if existing and not config.overwrite and not (config.resume and last_path.is_file()):
        raise ValueError(f"Incomplete/incompatible output: {directory}. Use a new root or explicit --overwrite.")
    set_seed(seed)
    torch.set_float32_matmul_precision("medium")
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
    criterion = weighted_bce(data.prevalence, device)

    def loader(records, config, *, training, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.utils.data.DataLoader(dataset_factory(records, training=training),
            batch_size=config.batch_size, shuffle=training, num_workers=config.num_workers,
            worker_init_fn=seed_worker, generator=generator, persistent_workers=False, drop_last=False)

    history, first_epoch, best_epoch, best_loss, best_state = [], 1, 0, float("inf"), None
    early_stopping = {"best_score": float("inf"), "wait_count": 0, "stopped_epoch": None}
    top_checkpoints = []
    if existing and config.resume and not config.overwrite:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError(f"Checkpoint settings/data/implementation mismatch: {directory}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        first_epoch, best_epoch = checkpoint["completed_epoch"] + 1, checkpoint["best_epoch"]
        best_loss, best_state = checkpoint["best_validation_loss"], checkpoint["best_model_state"]
        early_stopping, top_checkpoints = checkpoint["early_stopping"], checkpoint["top_checkpoints"]
        for entry in top_checkpoints:
            expected_name = f"checkpoints/epoch_{entry['epoch']:04d}.pt"
            if entry["file"] != expected_name or file_checksum(directory / expected_name) != entry["sha256"]:
                raise ValueError(f"Ranked checkpoint is missing or changed: {directory / expected_name}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").unlink(missing_ok=True)
    write_json(directory / "resolved_config.json", configuration_dict(config))
    # Explicit math attention avoids nondeterministic fused-kernel backward paths.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH), logging_redirect_tqdm(loggers=[LOGGER]), tqdm(
            total=config.epochs, initial=first_epoch - 1, desc=f"{protocol_updates['condition']} seed {seed}", unit="epoch", dynamic_ncols=True) as progress:
        epochs = range(first_epoch, config.epochs + 1) if early_stopping["stopped_epoch"] is None else ()
        for epoch in epochs:
            local_seed = epoch_seed(seed, epoch)
            set_seed(local_seed)
            model.train()
            training_loss = 0.
            for batch in loader(data.splits["train"], config, training=True, seed=local_seed):
                optimizer.zero_grad(set_to_none=True)
                labels = batch["labels"].to(device)
                logits = model(_inputs(batch, device))
                loss = criterion(logits, labels)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise ValueError("Non-finite training loss/logits.")
                loss.backward()
                optimizer.step()
                training_loss += float(loss.detach()) * len(labels)
            training_loss /= len(data.splits["train"])
            validation = _evaluate(model, data.splits["validation"], criterion, config, device, loader=loader, **evaluation_options)
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            _update_early_stopping(early_stopping, validation["loss"], epoch,
                patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta)
            top_checkpoints, retired = _rank_checkpoint(directory, top_checkpoints, epoch=epoch,
                loss=validation["loss"], signature=signature, model=model, optimizer=optimizer,
                early_stopping=early_stopping)
            history.append({"epoch": epoch, "epoch_seed": local_seed, "training_loss": training_loss,
                            "validation_loss": validation["loss"], "best_epoch": best_epoch,
                            "early_stopping_best_score": early_stopping["best_score"],
                            "early_stopping_wait_count": early_stopping["wait_count"]})
            _save_torch(last_path, {"signature": signature, "completed_epoch": epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "history": history,
                "best_epoch": best_epoch, "best_validation_loss": best_loss, "best_model_state": best_state,
                "early_stopping": dict(early_stopping), "top_checkpoints": top_checkpoints})
            write_json(directory / "history.json", history)
            # Only retire a known ranked file after the new last.pt is published.
            for entry in retired:
                (directory / entry["file"]).unlink(missing_ok=True)
            progress.update()
            progress.set_postfix(train=f"{training_loss:.4f}", val=f"{validation['loss']:.4f}",
                                 best=best_epoch, wait=f"{early_stopping['wait_count']}/{config.early_stopping_patience}")
            if epoch_callback is not None:
                epoch_callback(epoch)
            if early_stopping["stopped_epoch"] is not None:
                LOGGER.info("Seed %d stopped at epoch %d: validation loss did not improve by more than %.6g for %d checks.",
                            seed, epoch, config.early_stopping_min_delta, config.early_stopping_patience)
                break
        if best_state is None:
            raise ValueError("No valid best checkpoint was produced.")
        model.load_state_dict(best_state)
        _save_torch(directory / "best.pt", {"signature": signature, "epoch": best_epoch,
                     "validation_loss": best_loss, "model_state": best_state})
        validation, validation_arrays = _evaluate(model, data.splits["validation"], criterion, config, device, loader=loader, predictions=True, **evaluation_options)
        test, test_arrays = _evaluate(model, data.splits["test"], criterion, config, device, loader=loader, predictions=True, **evaluation_options)
    threshold_selector = select_threshold if task is None else task.select_threshold
    metric_function = threshold_metrics if task is None else task.threshold_metrics
    threshold = threshold_selector(validation_arrays)
    for metrics, arrays in ((validation, validation_arrays), (test, test_arrays)):
        metrics.update(metric_function(arrays, threshold))
        arrays["predictions"] = (arrays["probabilities"] >= threshold).astype(np.uint8)
        arrays["threshold"] = np.asarray(threshold)
    _save_predictions(directory / "predictions_validation.npz", validation_arrays)
    _save_predictions(directory / "predictions_test.npz", test_arrays)
    write_json(directory / "history.json", history)
    completed = history[-1]["epoch"]
    result = {"seed": seed, "threshold": threshold, "best_epoch": best_epoch, "validation": validation, "test": test,
              "epochs_completed": completed, "max_epochs": config.epochs,
              "stopped_early": early_stopping["stopped_epoch"] is not None and completed < config.epochs,
              "stop_reason": "early_stopping" if early_stopping["stopped_epoch"] is not None else "max_epochs",
              "early_stopping": early_stopping, "top_checkpoints": top_checkpoints,
              "checkpoint": str(directory / "best.pt"),
              "trainable_parameters": sum(p.numel() for p in model.parameters()), "protocol": protocol,
              "data_fingerprint": data.data_fingerprint, "implementation_sha256": implementation,
              "tag_names": data.tag_names, "data_audit": data.audit, "signature": signature}
    write_json(directory / "result.json", result)
    finish_artifacts(directory, signature, ["best.pt", "last.pt", "history.json", "resolved_config.json",
                     "predictions_validation.npz", "predictions_test.npz", "result.json",
                     *[entry["file"] for entry in top_checkpoints]])
    LOGGER.info("Seed %d: best epoch=%d, test PR-AUC=%.4f, ROC-AUC=%.4f", seed, best_epoch, test["pr_auc_macro"], test["roc_auc_macro"])
    return result
