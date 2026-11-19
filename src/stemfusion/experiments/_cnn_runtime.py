"""Process setup for the isolated official musicnn environment."""
import os
from pathlib import Path
import site
import sys


def prepare_runtime():
    """Set official musicnn compatibility before imports; restart once if needed.

    The dynamic linker reads LD_LIBRARY_PATH at process startup, so merely
    setting it immediately before importing TensorFlow is insufficient.
    """
    packages = Path(site.getsitepackages()[0])
    libraries = sorted(str(path) for path in (packages / "nvidia").glob("*/lib") if path.is_dir())
    environment = os.environ.copy()
    previous = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        libraries + [path for path in previous if path and path not in libraries]
    )
    environment["TF_USE_LEGACY_KERAS"] = "1"
    environment.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    cuda_bin = packages / "nvidia/cuda_nvcc/bin"
    if cuda_bin.is_dir():
        previous_path = environment.get("PATH", "").split(os.pathsep)
        environment["PATH"] = os.pathsep.join(
            [str(cuda_bin)] + [path for path in previous_path if path != str(cuda_bin)]
        )
    if environment != dict(os.environ):
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
