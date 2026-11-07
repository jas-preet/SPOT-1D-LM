"""Shared fixtures.

The end-to-end tests run the real command line and network architectures with random
weights and random embeddings. ESM-1b and ProtT5 are replaced by tests/fakes.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAKES = REPO_ROOT / "tests" / "fakes"
SCRIPTS = ("run_inference.py", "main.py", "generate_esm.py", "generate_prottrans.py",
           "run_SPOT-1D-LM.sh")


def _save_random_checkpoints(directory):
    from models.bilstm import Network as Network1
    from models.bilstm_reg import Network as Network4
    from models.ms_res_lstm import Network as Network3
    from models.ms_res_lstm_reg import Network as Network6
    from models.ms_resnet import Network as Network2
    from models.ms_resnet_reg import Network as Network5

    directory.mkdir()
    torch.manual_seed(0)
    networks = (Network1, Network2, Network3, Network4, Network5, Network6)
    for number, network in enumerate(networks, start=1):
        torch.save(network().state_dict(), directory / f"model{number}.pt")


@pytest.fixture(scope="session")
def install(tmp_path_factory):
    """A copy of the program with random checkpoints in the usual location."""
    root = tmp_path_factory.mktemp("install with space")
    for name in SCRIPTS:
        shutil.copy(REPO_ROOT / name, root)
    for name in ("dataset", "models"):
        shutil.copytree(REPO_ROOT / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (root / "inputs").mkdir()
    _save_random_checkpoints(root / "checkpoints")
    return root


@pytest.fixture
def run_script(install):
    def run(script, *args, cwd=install):
        env = dict(os.environ, PYTHONPATH=str(FAKES),
                   PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
        interpreter = ["bash"] if script.endswith(".sh") else [sys.executable]
        return subprocess.run(
            [*interpreter, str(install / script), *map(str, args)],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
        )

    return run


@pytest.fixture
def make_list(tmp_path):
    """Write FASTA files and a file list that points at them."""

    def make(records, trailer=""):
        paths = []
        for filename, text in records.items():
            path = tmp_path / filename
            path.write_text(text)
            paths.append(str(path))
        file_list = tmp_path / "list.txt"
        file_list.write_text("\n".join(paths) + "\n" + trailer)
        return file_list

    return make


@pytest.fixture
def write_embeddings(install):
    def write(name, length):
        rng = np.random.default_rng(0)
        for suffix, width in (("esm", 1280), ("pt", 1024)):
            values = rng.normal(size=(length, width)).astype(np.float32)
            np.save(install / "inputs" / f"{name}_{suffix}.npy", values)

    return write
