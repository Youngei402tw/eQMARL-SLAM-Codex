"""Hash-verifiable CPU checkpoints with explicit complete continuation state.

Payloads use tensors and plain containers and load with ``weights_only=True``;
no arbitrary pickled Python objects are needed for RNG restoration.
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


def _plain_state(value: Any, path: str = "state") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_plain_state(v, f"{path}[]") for v in value)
    if isinstance(value, dict) and all(isinstance(k, (int, str)) for k in value):
        return {k: _plain_state(v, f"{path}.{k}") for k, v in value.items()}
    raise TypeError(f"{path} must contain only plain scalars, tensors, lists, tuples and dictionaries")


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return result


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["algorithm"], ns["keys"].cpu().numpy().astype(np.uint32), ns["position"], ns["has_gauss"], ns["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    recurrent_state: Any,
    progress: dict,
    normalization: Any = None,
    scheduler: Any = None,
    environment_state: Any = None,
) -> str:
    """Atomically save a continuation checkpoint and return its SHA256.

    Independent ``numpy.random.Generator`` or ``random.Random`` objects belong
    in ``environment_state`` as explicit plain state dictionaries; global RNG
    capture cannot discover them. Model construction must precede load, which
    restores the RNG only after loading the model and optimizer.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "normalization": _plain_state(normalization, "normalization"),
        "rng": _rng_state(),
        "recurrent_state": _plain_state(recurrent_state, "recurrent_state"),
        "environment_state": _plain_state(environment_state, "environment_state"),
        "progress": _plain_state(progress, "progress"),
    }
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return checkpoint_sha256(destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_sha256: str | None = None,
    scheduler: Any = None,
    map_location: str = "cpu",
) -> dict:
    """Restore a same-stack checkpoint and return progress and runtime state.

    For archived experiments pass the artifact inventory's expected SHA256;
    an omitted hash is permitted only for a newly written local checkpoint.
    """
    source = Path(path)
    if expected_sha256 is not None and checkpoint_sha256(source) != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if payload.get("schema_version") != "checkpoint-v1":
        raise ValueError("unsupported checkpoint schema")
    if (payload["scheduler"] is None) != (scheduler is None):
        raise ValueError("scheduler presence differs from the checkpoint")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"])
    return {name: payload[name] for name in ["progress", "recurrent_state", "normalization", "environment_state"]}
