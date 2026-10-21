"""Distinguish eight-replicate mean-gradient error from replicate RMS error."""
import hashlib,json
from pathlib import Path
import numpy as np


def analyze(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=root/'experiments/07/07-frozen-shot-noise-v1_attempt-01/metrics.jsonl'
    rows=[json.loads(line) for line in raw.read_text().splitlines()];result=[]
    for ref in [r for r in rows if r['kind']=='analytic_single_gradient']:
        g=np.array(ref['gradient']);mask=[]
        for group in ref['parameter_groups']:mask.extend([not group['name'].startswith('head.')] * group['size'])
        mask=np.array(mask,dtype=bool)
        reps=np.array([r['gradient'] for r in rows if r['kind']=='shot_replicate' and r['method']==ref['method'] and r['shots_per_execution']==ref['shots_condition']])
        norms=np.linalg.norm((reps-g)[:,mask],axis=1)/np.linalg.norm(g[mask])
        mean_error=float(np.linalg.norm((reps.mean(0)-g)[mask])/np.linalg.norm(g[mask]))
        result.append({'method':ref['method'],'shots':ref['shots_condition'],'relative_error_of_eight_replicate_mean':mean_error,'single_replicate_relative_errors':norms.tolist(),'single_replicate_relative_rms_error':float(np.sqrt(np.mean(norms**2)))})
    d={'analysis_version':'shot-v1-error-clarification','metrics_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'gradient_results':result,'interpretation':'The immutable original field gradient_relative_rmse computes relative error of the eight-replicate mean gradient, not RMS across single replicates. The registered0.5 flag applies to that mean. This supplemental raw-only analysis records both quantities without changing gates, data, configurations or model selection.'}
    (output/'shot-analysis.json').write_text(json.dumps(d,indent=2)+'\n');return d


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(analyze(a.root,a.output),indent=2))
