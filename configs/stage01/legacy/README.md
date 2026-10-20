# Isolated original reference runtime

The upstream Python 3.9 requirement and TensorFlow 2.7.0 / TensorFlow Quantum
0.7.2 constraints are retained. The full original `requirements.txt` resolved and
installed on Python 3.9.25 Linux x86-64. Import checks and `uv pip check` passed.
`requirements-legacy.lock` contains the actual resolved distribution versions.

Initial TFQ import failed because setuptools 82 removed `pkg_resources`, which
old google-api-core imports. The ordinary compatible requirement
`setuptools<82` resolved to 81.0.0; this correction preserves all upstream
constraints. Both failed and successful import logs are retained.

Recreate a clean Python 3.9 environment, then install the lock from
`https://pypi.org/simple`. This runtime is separate from the new Torch runtime.
The original source is attributed under `third_party/eqmarl`; source files remain
unchanged. `reference_probe.py` imports only the original tools, operations,
circuit construction, and layer through a private namespace, bypassing unrelated
package initializer imports.

The committed fixtures use independent NumPy oracle seed 2701. They cover 4 and
8 total qubits, all four Bell states, and a matched input-entanglement-disabled
condition. Original one-wire partitions attempt an invalid self-CZ and are not
patched in this adapter. Two-qubit diagnostic checks belong to the independent
NumPy/PennyLane path. `fixture_manifest.json` records tolerances and oracle hash.

After the source/config commit is published and verified, run:

```bash
/path/to/legacy/bin/python eqmarl_slam/reference_probe.py \
  --fixtures configs/stage01/legacy/fixtures.json --output /path/to/reference-results.json
```

The registered adapter uses analytic forward evaluation only. Original TFQ
float32 expectations use a prespecified `2e-5` absolute tolerance; Cirq complex128
uses `1e-8`, with state norm error at most `1e-10`. No TFQ derivative or original
training reproduction is claimed. Exactly one Cirq and one TFQ forward circuit
execution per fixture are counted; no backward or physical shots are requested.
Installation/import feasibility is recorded here; parity results are archived
only in the subsequent registered experiment bundle.
