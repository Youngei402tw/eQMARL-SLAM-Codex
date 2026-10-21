# Stage 03 GTSAM compatibility record

The official PyPI `gtsam==4.3.0` CPython 3.11 Linux x86-64 wheel works with the
existing NumPy 2.4.6 / Torch / PennyLane runtime. No NumPy downgrade, dependency
constraint suppression, renamed wheel, or separate estimator process is needed.

The package metadata, exact wheel filename, URL, size, and SHA-256 are recorded in
`pypi-wheel-metadata.json` and `verified-wheel.json`. A fresh wheel retrieval
matched PyPI SHA-256
`63cae1c925f0094832b7f8d629951d4631cac57c6d7faba140c8ecc9f3c57752`.

A clean Python 3.11.17 environment at `/workspace/slam-env/current` first installed
GTSAM 4.3.0 and NumPy 2.4.6. `compatibility_probe.py` optimized an anchored Pose2
graph containing a pose prior, body-frame odometry between factor, and a real
bearing/range landmark measurement factor. It extracted the resulting Pose2 and
Point2 and finite positive marginal covariances. The same native probe then
passed in the primary `/workspace/eqmarl-venv` environment. Both environments
passed `uv pip check`; all primary Torch/PennyLane/GTSAM imports coexist.

The updated root `requirements.lock` is an actual freeze of the validated primary
environment; `requirements-slam.lock` records the clean GTSAM-only probe runtime.
Stage 01 dependency identities remain preserved in their historical commits.

The primary installation command was:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv pip install \
  --python /workspace/eqmarl-venv/bin/python gtsam==4.3.0
```

Recheck native installation compatibility with:

```bash
/workspace/eqmarl-venv/bin/python configs/stage03/runtime/compatibility_probe.py
```

The tiny probe verifies native installation and array interchange. It is not a
scientific trajectory, revisit, calibration, or SLAM comparison experiment.
Stage 03 scientific diagnostics must use the published source/config identity
and their registered experiment bundle. The selected frontend uses the direct
`eqmarl_slam.slam` backend.
