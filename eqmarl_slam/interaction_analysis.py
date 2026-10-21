"""Raw-only component summaries and complete parameter-group connectivity audit."""
import hashlib,json
from pathlib import Path
import numpy as np


def analyze(root,output):
    root=Path(root);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    folder=root/'experiments/06/06-local-edge-fit-v1_attempt-01';raw=folder/'metrics.jsonl'
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    results=[r for r in rows if r['kind']=='interaction_result']
    assert len(results)==56
    local=['encoder.weight','encoder.bias','local.0.weight','local.0.bias','local.2.weight','local.2.bias']
    audits=[];means=[]
    for result in results:
        key=(result['condition'],result['method'],result['seed'])
        updates=[r for r in rows if r['kind']=='interaction_update' and (r['condition'],r['method'],r['seed'])==key]
        expected=local+(['edge'] if result['method']=='bilinear' else ['theta','scales','head.weight','head.bias'] if result['method'] in ['eqmarl','quantum_noent'] else [])
        check=len(updates)==32 and all(set(r['gradient_norms'])==set(expected) and all(np.isfinite(v) for v in r['gradient_norms'].values()) for r in updates)
        connected={name:any(r['gradient_norms'].get(name,0)>1e-12 for r in updates) for name in expected}
        audits.append({'condition':key[0],'method':key[1],'seed':key[2],'all_declared_groups_finite':bool(check),'group_connected':connected,'passed':bool(check and all(connected.values()))})
    for condition in dict.fromkeys(r['condition'] for r in results):
        for method in ['none','bilinear','quantum_noent','eqmarl']:
            cases=[r for r in results if r['condition']==condition and r['method']==method]
            means.append({'condition':condition,'method':method,'mean_validation_mse':float(np.mean([r['validation_mse'] for r in cases])),'seed_validation_mse':{str(r['seed']):r['validation_mse'] for r in cases}})
    result={'analysis_version':'interaction-v1-raw-audit','metrics_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'gradient_audits':audits,'all_gradient_groups_passed':all(a['passed'] for a in audits),'mean_results':means,'selection':'Retain the Stage05 task architectures. This synthetic/frozen-corpus fitting study does not establish a repeatable policy-validation improvement that justifies a more expensive new interaction model. Strong classical remains centralized classical; matched noent/eQMARL retained for final comparison.','scope':'Two fitting seeds,16 generated training and8 validation inputs per synthetic utility, plus unchanged real frozen-return corpus. Different parameter counts and initialization can affect32-update outcomes. This is conditional component evidence, not environmental causal attribution or a task-policy improvement.'}
    (output/'interaction-analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args();r=analyze(a.root,a.output);print(json.dumps({'gradient_groups_passed':r['all_gradient_groups_passed'],'mean_results':r['mean_results']},indent=2))
