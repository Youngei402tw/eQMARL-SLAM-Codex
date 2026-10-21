"""Exploratory seed-level validation summaries; never reads final-test data."""
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(root, phase):
    root = Path(root)
    index = json.loads((root/'experiments/05/index.json').read_text())
    rows = []
    for registered in index['runs']:
        if not registered['run_id'].startswith('05-'+phase+'-'):
            continue
        folder = root/'experiments/05'/registered['run_id']
        config = json.loads((folder/'resolved_config.yaml').read_text())
        manifest = json.loads((folder/'manifest.json').read_text())
        metrics = folder/'metrics.jsonl'
        validation = [json.loads(line) for line in metrics.read_text().splitlines()
                      if json.loads(line)['kind'] == 'validation_episode']
        row = {'run_id': registered['run_id'], 'method': config['critic_method'],
               'seed': config['actor_seed'], 'learning_rate': config['learning_rate'],
               'status': manifest['status'], 'validation_episodes': len(validation),
               'source_commit': manifest['source_commit'],
               'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest(),
               'resources': manifest['resources_consumed']}
        for field in ['return', 'physical_coverage', 'estimated_coverage',
                      'fixed_landmark_accuracy', 'success', 'trajectory_ate_m',
                      'pose_error_m', 'collisions', 'communication_bytes']:
            row[field] = float(np.mean([r[field] for r in validation])) if validation else None
        rows.append(row)
    return rows


def select_learning_rates(root):
    root = Path(root)
    plan = json.loads((root/'configs/stage05/comparison-plan.json').read_text())
    rows = summarize(root, 'tuning')
    scores = {}
    selected = {}
    for method in plan['methods']:
        scores[method] = {}
        for rate in plan['learning_rates']:
            trials = [r for r in rows if r['method'] == method and r['learning_rate'] == rate]
            if sorted(r['seed'] for r in trials) != sorted(plan['tuning_seeds']):
                raise ValueError('Every registered tuning trial needs a recorded outcome before selection')
            scores[method][str(rate)] = (float(np.mean([r['return'] for r in trials]))
                                         if all(r['status'] == 'completed' for r in trials) else None)
        finite = [(float(score), -float(rate), float(rate)) for rate, score in scores[method].items()
                  if score is not None]
        if not finite:
            raise ValueError(f'No valid tuning configuration for {method}; diagnose before dependent pilots')
        selected[method] = max(finite)[2]
    return {'selection_rule': plan['selection'], 'selected_learning_rates': selected,
            'validation_return_scores': scores, 'raw_seed_rows': rows,
            'final_test_accessed': False}
