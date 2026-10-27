"""Locked paired-seed confirmatory analysis."""
import hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=("central_classical","bilinear","flexible_interaction","quantum_noent","eqmarl")
CONDITIONS=("additive","low","high")
SEEDS=tuple(range(13001,13009))


def interval(values,rng,resamples=10000):
    a=np.asarray(values,dtype=float); draws=a[rng.integers(0,len(a),size=(resamples,len(a)))].mean(axis=1)
    return [float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5))]


def main():
    cells={}; ledger=[]; failures=[]
    for folder in sorted((ROOT/"experiments/13").glob("13-confirmatory-*")):
        c=json.loads((folder/"resolved_config.yaml").read_text());m=json.loads((folder/"manifest.json").read_text())
        d=json.loads((folder/"diagnostics.json").read_text());s=json.loads((folder/"summary.json").read_text())
        key=(c["condition"]["name"],c["method"],c["actor_seed"])
        test=d.get("summary",{}).get("test",[])
        row={"run_id":folder.name,"condition":key[0],"method":key[1],"seed":key[2],"status":m["status"],
             "test_episodes":len(test),"gates":d.get("gates",{}),"metrics_sha256":s["metrics_sha256"],
             "resources":m["resources_consumed"]}
        ledger.append(row)
        if m["status"]!="completed" or len(test)!=100 or not d.get("gates",{}).get("frozen_test_policy") or not d.get("gates",{}).get("complete_locked_test"):
            failures.append(row);continue
        metric_names=("physical_coverage","success","fixed_landmark_accuracy","pose_error_m","trajectory_ate_m","trajectory_rpe_m","communication_bytes","travel_distance","collisions","return","macro_decisions")
        cells[key]={name:statistics.fmean(float(x[name]) for x in test) for name in metric_names}
    expected={(c,m,s) for c in CONDITIONS for m in METHODS for s in SEEDS}
    missing=sorted(expected-set(cells))
    primary=None;secondary=[]
    if not missing and not failures:
        effects={condition:[cells[(condition,"eqmarl",seed)]["physical_coverage"]-cells[(condition,"quantum_noent",seed)]["physical_coverage"] for seed in SEEDS] for condition in CONDITIONS}
        did=[effects["high"][i]-effects["additive"][i] for i in range(len(SEEDS))]
        rng=np.random.default_rng(14001)
        primary_interval=interval(did,rng)
        primary={"seed_effects_by_condition":effects,"seed_interactions":did,"estimate":statistics.fmean(did),
                 "interval_95":primary_interval,"minimum_effect_of_scientific_interest":.02,
                 "interval_spans_zero":primary_interval[0]<=0<=primary_interval[1]}
        for comparator in ("central_classical","bilinear","flexible_interaction"):
            values=[cells[("high","eqmarl",seed)]["physical_coverage"]-cells[("high",comparator,seed)]["physical_coverage"] for seed in SEEDS]
            secondary.append({"contrast":f"eqmarl-minus-{comparator}_high","seed_effects":values,
                              "estimate":statistics.fmean(values),"interval_95_unadjusted":interval(values,rng)})
    table=[{"condition":c,"method":m,"seed":s,**cells[(c,m,s)]} for c,m,s in sorted(cells)]
    resources={k:sum(row["resources"][k] for row in ledger) for k in ("primitive_steps","decisions","updates","wall_seconds","cpu_core_hours","gpu_device_hours","circuit_evaluations","shots")}
    output={"analysis_version":"mechanism-confirmatory-v1","complete":not missing and not failures,
      "planned_outcomes":120,"completed_cells":len(cells),"missing_cells":missing,"failures":failures,
      "primary":primary,"secondary_descriptive":secondary,"seed_cell_table":table,"run_ledger":ledger,"resources":resources,
      "interpretation_rule":"An interval spanning zero is a completed null result. No equivalence or general quantum-advantage conclusion."}
    dest=ROOT/"analysis/13/confirmatory-analysis.json";dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__": main()
