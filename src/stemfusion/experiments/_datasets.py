"""Dataset adapters for the shared genre experiment orchestration."""

from dataclasses import dataclass
from typing import Callable

SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals")


@dataclass(frozen=True)
class GenreDataset:
    name: str
    number_of_classes: int
    inspect_manifest: Callable
    cache_paths: Callable
    load_features: Callable
    historical_selection_note: str | None = None


def get_dataset(dataset: str) -> GenreDataset:
    if dataset == "fma_medium":
        from stemfusion.data.fma_features import inspect_manifest, cache_paths, load_fma_features

        return GenreDataset("FMA-Medium", 16, inspect_manifest, cache_paths, load_fma_features,
            "The fixed nested stem sets were influenced by historical FMA test results; these reruns are not an independent confirmatory holdout.")
    if dataset == "gtzan":
        from stemfusion.data.gtzan_features import inspect_manifest, cache_paths, load_gtzan_features

        return GenreDataset("GTZAN", 10, inspect_manifest, cache_paths, load_gtzan_features)
    raise ValueError(f"Unsupported genre dataset: {dataset}")
