"""Generate the locked Stage 13 matrix without reading held-out scenarios."""
import copy,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=("central_classical","bilinear","flexible_interaction","quantum_noent","eqmarl")
CONDITIONS=(("additive",0.0),("low",0.5),("high",1.0))
SEEDS=range(13001,13009)
RATES={"central_classical":.001,"bilinear":.003,"flexible_interaction":.003,"quantum_noent":.001,"eqmarl":.003}
COUNTS={"central_classical":249,"bilinear":125,"flexible_interaction":250,"quantum_noent":86,"eqmarl":86}
TEST={"path":"configs/data/test-mechanism-v1.json","sha256":"2085bcfadf915a196588ae4db12a4e0ca5e45462f21456a1c878107127879a0e","expected_count":100}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv)!=2 or len(sys.argv[1])!=40: raise SystemExit("usage: generate_stage13_configs.py STAGE12_COMPLETION_SHA")
    dependency=sys.argv[1]
    template=json.loads(next((ROOT/"configs/stage12").glob("12-pilot-additive-central_classical_seed-12101*.yaml")).read_text())
    plan=ROOT/"configs/stage13/analysis-plan.json"
    output=ROOT/"configs/stage13"; output.mkdir(exist_ok=True)
    runs=[]
    for condition,alpha in CONDITIONS:
      for seed in SEEDS:
       for method in METHODS:
        c=copy.deepcopy(template); rate=RATES[method]
        run_id=f"13-confirmatory-{condition}-{method}_seed-{seed}_attempt-01"
        c.update(stage="13",protocol_id="mechanism-confirmatory-v1",dependency_completion_commit=dependency,
          run_id=run_id,experiment="confirmatory-training",method=method,profile="confirmatory",
          hypothesis="The eQMARL-minus-no-entanglement coverage effect increases from additive to high complementarity under the locked paired design.",
          wall_seconds=7200,max_primitive_steps=15000,actor_seed=seed,critic_seed=seed+1000,action_seed=seed+2000,
          learning_rate=rate,training_steps=10000,training_wall_seconds=6600,critic_method=method,
          interpretation="Locked confirmatory analytic-simulation outcome; trained seed is the replication unit. Nulls and failures retained; no equivalence, hardware, speedup, or broad quantum-advantage claim.",
          condition={"name":condition,"complementarity_alpha":alpha},
          action_modes={"training":"sampled","validation":"greedy","test":"greedy"},
          checkpoint_selection="final_fixed_budget_weights_no_validation_or_test_selection",
          analysis_version="mechanism-confirmatory-v1",analysis_plan=str(plan.relative_to(ROOT)),analysis_plan_sha256=sha(plan),
          test_scenarios_manifest=copy.deepcopy(TEST))
        c["task"]["complementarity_alpha"]=alpha
        c["model"]["critic"]=method;c["model"]["parameter_counts"]["critic"]=COUNTS[method]
        c["model"]["optimizer"]["learning_rate"]=rate
        c["seed_streams"].update(model_init=seed,critic_init=seed+1000,action=seed+2000)
        c["seed_streams"]["sensor"]["splits"]=["train","validation","test"]
        c["split_manifests"]["test"]={"path":TEST["path"],"sha256":TEST["sha256"]}
        path=output/f"{run_id}.yaml"; path.write_text(json.dumps(c,indent=2)+"\n")
        runs.append({"run_id":run_id,"status":"planned","condition":condition,"config":str(path.relative_to(ROOT)),"config_sha256":sha(path)})
    exp=ROOT/"experiments/13";exp.mkdir(exist_ok=True)
    (exp/"index.json").write_text(json.dumps({"stage":"13","dependency_completion_commit":dependency,
      "source_locked_before_test_access":False,"test_accessed":False,"runs":runs},indent=2)+"\n")


if __name__=="__main__": main()
