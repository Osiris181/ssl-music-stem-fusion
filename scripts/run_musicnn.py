"""Launch a script using the current venv's CUDA libraries and legacy Keras.

Example: musicnn/bin/python scripts/run_musicnn.py path/to/extraction_cnn.py
"""
import os
from pathlib import Path
import site
import sys

packages = Path(site.getsitepackages()[0])
libraries = sorted(str(path) for path in (packages / "nvidia").glob("*/lib") if path.is_dir())
previous = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + ([previous] if previous else []))
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
cuda_bin = packages / "nvidia/cuda_nvcc/bin"
if cuda_bin.is_dir():
    os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
