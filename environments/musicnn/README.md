# Official musicnn extraction environment

This independent `pyproject.toml` pins the verified Python 3.12 TensorFlow
runtime for frozen official MSD_musicnn extraction. The repository root's
`pyproject.toml` remains the main PyTorch research environment. This Linux
x86_64 CUDA preset is not a portable macOS/Windows installation recipe.

## Install with uv

Run from the repository root, not this environment directory:

```bash
# Install/synchronize the runtime into the existing root-level musicnn venv.
UV_PROJECT_ENVIRONMENT="$PWD/musicnn" uv sync --project environments/musicnn --inexact

# Install unchanged official code/weights without resolving its obsolete deps.
uv pip install --python musicnn/bin/python --no-deps 'musicnn @ git+https://github.com/jordipons/musicnn.git@516acb2a0ff5ef73f64547898e018e793152c506'
```

`uv sync` creates `musicnn/` if absent. If it already exists, it is reused;
do not recreate/delete a working environment. `--inexact` retains the separately
installed official musicnn package during future synchronizations. Do not use
plain `uv sync` inside this directory: that would default to another `.venv`.
The commands above explicitly target `musicnn/`, not the active main environment.

The official package declares NumPy `<1.17`, incompatible with modern TensorFlow.
The verified runtime uses NumPy 1.26.4. Installing musicnn with `--no-deps` is an
intentional dependency-metadata override, not a model-code change. Consequently
`uv pip check --python musicnn/bin/python` reports this known conflict. Do not
resolve it by downgrading NumPy or installing musicnn through normal resolution.
The extractor verifies original source/checkpoint checksums and refuses silently
unrestored weights. No model training happens in this environment.

## Run extraction

```bash
source musicnn/bin/activate
cd src/stemfusion/experiments/gtzan
python extraction_cnn.py --dry-run
python extraction_cnn.py
```

The entry point automatically configures legacy Keras and this environment's
CUDA library paths before importing TensorFlow. No launcher script is required.
The preset was tested on an RTX 5090 with TensorFlow 2.21.0, tf-keras 2.21.0,
CUDA runtime 12.9.79 and cuDNN 9.28.1.4. PTX JIT/deprecation warnings can occur;
actual GPU inference was verified, not native SM120 binary support. A compatible
host NVIDIA driver is still required; pip packages do not install that driver.

For classifier training, deactivate this environment and activate your main
PyTorch environment, then run `python train_mid_cnn.py`. Do not install PyTorch
or the root project's training extras into this extraction environment.

Existing users whose extraction already works do not need to reinstall.
This configuration migration does not alter model weights, feature settings,
caches, or completed results.
