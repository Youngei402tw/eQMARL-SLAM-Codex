"""Regenerate prespecified seed-level tables exclusively from archived raw rows."""
import hashlib,json
from pathlib import Path
import numpy as np

METRICS=['physical_coverage','estimated_coverage','fixed_landmark_accuracy','pose_error_m','trajectory_ate_m','trajectory_rpe_m','travel_distance_m','collisions','communication_bytes','messages','success','localization_failure','return','primitive_steps']


def analyze(root,output):
    root=Path(root);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    plan=json.loads((root/'configs/stage08/analysis-plan.json').read_text())
    index=json.loads((root/'experiments/08/index.json').read_text())
    seed_rows=[];provenance=[];failures=[]
    for run in index['runs']:
        folder=root/'experiments/08'/run['run_id']
        manifest=json.loads((folder/'manifest.json').read_text())
        config=json.loads((folder/'resolved_config.yaml').read_text())
        raw=folder/'metrics.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
        tests=[r for r in rows if r['kind']=='test_episode']
        provenance.append({'run_id':run['run_id'],'source_commit':manifest['source_commit'],'metrics_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'status':manifest['status'],'test_rows':len(tests)})
        expected=[r['scenario_id'] for r in config['test_scenarios']]
        if manifest['status']!='completed' or [r['scenario_id'] for r in tests]!=expected:
            failures.append({'run_id':run['run_id'],'status':manifest['status'],'observed_test_rows':len(tests),'failure':manifest['failure']})
            continue
        row={'run_id':run['run_id'],'method':config['critic_method'],'seed':config['actor_seed'],'test_episodes':len(tests)}
        row.update({metric:float(np.mean([r[metric] for r in tests])) for metric in METRICS})
        row['action_counts_by_agent']=np.sum([r['action_counts_by_agent'] for r in tests],axis=0).tolist()
        row['mean_reward_component_totals']={k:float(np.mean([r['reward_component_totals'][k] for r in tests])) for k in tests[0]['reward_component_totals']}
        row['resources']=manifest['resources_consumed'];seed_rows.append(row)
    groups={method:[r for r in seed_rows if r['method']==method] for method in plan['methods']}
    summaries={}
    for method,rows in groups.items():
        summaries[method]={'trained_seeds':len(rows),'metrics':{}}
        for metric in METRICS:
            values=np.array([r[metric] for r in rows])
            summaries[method]['metrics'][metric]={'mean':float(values.mean()) if len(values) else None,'seed_sd':float(values.std(ddof=1)) if len(values)>1 else None,'seed_values':values.tolist()}
    rng=np.random.default_rng(plan['bootstrap_seed']);comparisons=[]
    for first,second in plan['contrasts']:
        a={r['seed']:r for r in groups[first]};b={r['seed']:r for r in groups[second]}
        common=sorted(a.keys()&b.keys())
        for metric in plan['contrast_metrics']:
            difference=np.array([a[seed][metric]-b[seed][metric] for seed in common])
            if len(difference):
                draws=difference[rng.integers(0,len(difference),(plan['bootstrap_repetitions'],len(difference)))].mean(-1)
                interval=np.quantile(draws,[.025,.975]).tolist()
            else:interval=[None,None]
            comparisons.append({'first':first,'second':second,'metric':metric,'paired_seeds':common,'differences':difference.tolist(),'mean_difference':float(difference.mean()) if len(difference) else None,'percentile_95_interval':interval,'complete_locked_sample':len(common)==len(plan['seeds'])})
    result={'analysis_version':'confirmatory-v1-seed-bootstrap','analysis_plan_sha256':hashlib.sha256((root/'configs/stage08/analysis-plan.json').read_bytes()).hexdigest(),'seed_rows':seed_rows,'method_summaries':summaries,'contrasts':comparisons,'failures':failures,'raw_provenance':provenance,'interpretation':'Training seeds are the replication unit;100 episodes per seed are averaged before inference. Percentile paired-seed bootstrap intervals are descriptive with eight seeds. Nondetection is not equivalence. No hyperparameter or checkpoint choice uses test results.'}
    (output/'confirmatory-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Locked confirmatory results','','Each cell is a mean across independent trained seeds. Test policies are frozen and greedy; the critic is not executed during deployment.','','| Method | Seeds | Physical coverage | Map accuracy | Success | ATE (m) | Collisions | Bytes |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,s in summaries.items():
        v=s['metrics'];fmt=lambda k: f"{v[k]['mean']:.5f}" if v[k]['mean'] is not None else 'missing'
        lines.append(f"| {method} | {s['trained_seeds']} | {fmt('physical_coverage')} | {fmt('fixed_landmark_accuracy')} | {fmt('success')} | {fmt('trajectory_ate_m')} | {fmt('collisions')} | {fmt('communication_bytes')} |")
    lines+=['','Paired differences (first minus second);95% percentile intervals resample paired trained seeds:','']
    for c in comparisons:lines.append(f"- {c['first']} versus {c['second']}, {c['metric']}: {c['mean_difference']}, interval {c['percentile_95_interval']}; {len(c['paired_seeds'])} paired seeds.")
    lines+=['',f"Failed/incomplete runs: {len(failures)}. All remain in the ledger.",'','This is a fixed short-budget2D landmark-SLAM experiment in a known empty workspace. No convergence, hardware speedup, general quantum advantage, or equivalence claim follows. Exact state-vector training is distinct from frozen-critic finite-shot sensitivity.']
    (output/'confirmatory-results.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=".")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=analyze(args.root,args.output)
    print(json.dumps({"seed_rows":len(result["seed_rows"]),"failures":len(result["failures"]),"output":args.output}))
