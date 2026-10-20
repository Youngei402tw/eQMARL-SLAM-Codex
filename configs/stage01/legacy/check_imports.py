"""Installation/import feasibility probe only; no parity or research result."""
import importlib.metadata as md
import json
import platform
import sys
import traceback
from pathlib import Path
packages = ['tensorflow', 'tensorflow-quantum', 'cirq-core', 'cirq-google', 'numpy', 'protobuf', 'sympy']
result = {'python': sys.version, 'platform': platform.platform(), 'original_commit': '6b1b2e3817f661da81aad5e7188a362380dbdd1d', 'packages': {p: md.version(p) for p in packages}, 'modules': {}, 'scientific_experiments_executed': False}
for name in ['numpy', 'cirq', 'tensorflow', 'tensorflow_quantum']:
    try:
        module = __import__(name)
        result['modules'][name] = {'imported': True, 'version': getattr(module, '__version__', None)}
    except Exception as exc:
        result['modules'][name] = {'imported': False, 'error_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
raise SystemExit(0 if all(m['imported'] for m in result['modules'].values()) else 1)
