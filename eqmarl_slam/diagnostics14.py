"""Clean mechanism-release audit without training or held-out test replay."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
import torch

from .archive import validate_bundle
from .critics import make_critic
from .mappo import ActorCritic
from .policy_evaluation import evaluate_actor

ROOT=Path(__file__).resolve().parents[1]
METHODS=("central_classical","bilinear","flexible_interaction","quantum_noent","eqmarl")
CONDITIONS=("additive","low","high")
SEEDS=tuple(range(13001,13009))


def run_diagnostics(config,output,emit,progress=None):
    bundles=sorted((ROOT/"experiments/13").glob("13-confirmatory-*"))
    validations=[];checkpoints=[];cells={};resources={k:0 for k in ("primitive_steps","decisions","updates","wall_seconds","cpu_core_hours","gpu_device_hours","circuit_evaluations","shots")}
    for folder in bundles:
        validations.append(validate_bundle(folder,expected_source_commit=config["confirmatory_source_commit"],repository_root=ROOT))
        c=json.loads((folder/"resolved_config.yaml").read_text());d=json.loads((folder/"diagnostics.json").read_text());m=json.loads((folder/"manifest.json").read_text())
        payload=torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
        finite=all(torch.isfinite(value).all() for value in payload["model"].values())
        checkpoints.append({"run_id":folder.name,"sha256":hashlib.sha256((folder/"final.pt").read_bytes()).hexdigest(),"finite":bool(finite)})
        test=d["summary"]["test"]
        key=(c["condition"]["name"],c["method"],c["actor_seed"])
        cells[key]=statistics.fmean(row["physical_coverage"] for row in test)
        for name in resources: resources[name]+=m["resources_consumed"][name]
    replay=[]
    for condition in CONDITIONS:
      for method in METHODS:
        folder=ROOT/"experiments/13"/f"13-confirmatory-{condition}-{method}_seed-13001_attempt-01"
        c=json.loads((folder/"resolved_config.yaml").read_text());d=json.loads((folder/"diagnostics.json").read_text())
        payload=torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
        model=ActorCritic(critic=make_critic(method,c["critic_seed"]),actor_seed=c["actor_seed"],critic_seed=c["critic_seed"],hidden=c["actor_hidden"])
        model.load_state_dict(payload["model"]);model.eval();counter={"primitive_steps":0,"decisions":0}
        actual=evaluate_actor(model.actor,c["task"],c["validation_scenarios"][0],counter)
        expected=d["summary"]["validation"][0]
        errors={name:abs(float(actual[name])-float(expected[name])) for name in ("physical_coverage","return","fixed_landmark_accuracy","pose_error_m","trajectory_ate_m","trajectory_rpe_m")}
        row={"condition":condition,"method":method,"seed":13001,"scenario_id":actual["scenario_id"],"max_abs_error":max(errors.values()),"metric_errors":errors,"resources":counter}
        replay.append(row);emit({"kind":"release_validation_replay",**row})
    effects={condition:[cells[(condition,"eqmarl",seed)]-cells[(condition,"quantum_noent",seed)] for seed in SEEDS] for condition in CONDITIONS}
    interactions=np.asarray([effects["high"][i]-effects["additive"][i] for i in range(8)])
    rng=np.random.default_rng(14001);draws=interactions[rng.integers(0,8,size=(10000,8))].mean(axis=1)
    reproduced={"estimate":float(interactions.mean()),"interval_95":[float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5))],"seed_interactions":interactions.tolist()}
    published=json.loads((ROOT/"analysis/13/confirmatory-analysis.json").read_text())["primary"]
    analysis_match=abs(reproduced["estimate"]-published["estimate"])<=1e-15 and np.allclose(reproduced["interval_95"],published["interval_95"],atol=1e-15,rtol=0)
    gates={"all_120_bundles_valid":len(validations)==120 and all(x["valid"] for x in validations),
      "all_checkpoints_finite":len(checkpoints)==120 and all(x["finite"] for x in checkpoints),
      "all_15_replays_exact":len(replay)==15 and max(x["max_abs_error"] for x in replay)<=1e-12,
      "primary_analysis_exact":bool(analysis_match),"no_failed_confirmatory_runs":all(x["status"]=="completed" for x in validations),
      "zero_physical_shots":resources["shots"]==0}
    paper={"primary":{"estimate":reproduced["estimate"],"interval_95":reproduced["interval_95"],"minimum_effect":.02,"completed_null":reproduced["interval_95"][0]<=0<=reproduced["interval_95"][1]},
      "resources":resources,"methods":list(METHODS),"conditions":list(CONDITIONS),"trained_seeds_per_cell":8,"test_episodes_per_seed":100}
    claims=[
      {"claim":"Controlled complementarity increased scripted reward/value interaction from 0 to 4.","support":"Stage11 fixed-state diagnostic","scope":"registered two-robot analytic simulator"},
      {"claim":"The eQMARL-minus-no-entanglement coverage effect did not reliably increase from additive to high complementarity.","support":reproduced,"scope":"eight trained seeds; interval spans zero; not equivalence"},
      {"claim":"Flexible classical interaction critics remained competitive under the same sample budget.","support":"Stage13 descriptive high-condition contrasts","scope":"descriptive, unadjusted"},
      {"claim":"The tested analytic quantum critics required circuit evaluations without a demonstrated coverage benefit.","support":{"circuit_evaluations":resources["circuit_evaluations"],"shots":resources["shots"]},"scope":"simulator/backprop only; no hardware speed claim"}
    ]
    (Path(output)/"reproduced_analysis.json").write_text(json.dumps(reproduced,indent=2)+"\n")
    (Path(output)/"paper_tables.json").write_text(json.dumps(paper,indent=2)+"\n")
    (Path(output)/"claim_limitations.json").write_text(json.dumps(claims,indent=2)+"\n")
    audit_resources={"primitive_steps":sum(x["resources"]["primitive_steps"] for x in replay),"decisions":sum(x["resources"]["decisions"] for x in replay),"updates":0,"circuit_evaluations":0,"shots":0}
    return {"passed":all(gates.values()),"gates":gates,"resources":audit_resources,
      "summary":{"bundles_validated":len(validations),"checkpoints_loaded":len(checkpoints),"validation_replays":replay,"reproduced_primary":reproduced,"confirmatory_resources":resources,"claims":claims}}
