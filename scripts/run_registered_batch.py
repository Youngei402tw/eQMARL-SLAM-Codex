"""Sequential execution from an immutable published worktree; no automatic retries.

Run from the clean source worktree, with a verified source receipt. Outputs live
outside source so later runs keep the exact same source identity. The normal CLI
enforces preregistration, clean source, and cumulative stage budgets per run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--log-root', required=True)
    args = parser.parse_args()
    logs = Path(args.log_root)
    logs.mkdir(parents=True, exist_ok=True)
    for filename in args.configs:
        config = json.loads(Path(filename).read_text())
        run = config['run_id']
        destination = Path(config['artifact_output_root'])/config['stage']/run
        if destination.exists():
            raise RuntimeError(f'Refuse to replace or silently skip existing outcome: {run}')
        print(datetime.now(timezone.utc).isoformat(), 'BEGIN', run, flush=True)
        with (logs/f'{run}.log').open('x') as log:
            result = subprocess.run([sys.executable, '-m', 'eqmarl_slam.cli', 'run',
                                     '--config', filename, '--source-receipt', args.receipt],
                                    stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
        if not (destination/'manifest.json').exists():
            raise RuntimeError(f'Execution failed before finalized archive: {run}; inspect log')
        manifest = json.loads((destination/'manifest.json').read_text())
        print(datetime.now(timezone.utc).isoformat(), 'END', run, manifest['status'],
              manifest['resources_consumed'], 'exit', result.returncode, flush=True)
        if result.returncode and manifest['status'] == 'completed':
            raise RuntimeError(f'Completed numerical outcome failed archive validation: {run}; stop before next run')
        # A finalized failed outcome remains evidence, not an implicit retry.


if __name__ == '__main__':
    main()
