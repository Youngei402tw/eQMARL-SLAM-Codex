"""Generate the fixed three-condition Stage 12 pilot matrix after tuning."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "configs/stage12"
METHODS = ("central_classical", "bilinear", "flexible_interaction", "quantum_noent", "eqmarl")
CONDITIONS = (("additive", 0.0), ("low", 0.5), ("high", 1.0))
SEEDS = (12101, 12102, 12103)
COUNTS = {"central_classical":249,"bilinear":125,"flexible_interaction":250,"quantum_noent":86,"eqmarl":86}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(name):
    path = ROOT / f"configs/data/{name}-mechanism-v1.json"
    return path, json.loads(path.read_text())["scenarios"]


def main():
    selection = json.loads((ROOT / "analysis/12/tuning-selection.json").read_text())
    rates = selection["selected_learning_rates"]
    train_path, training = manifest("train")
    dev_path, _ = manifest("development")
    val_path, validation = manifest("validation")
    split = {"train":{"path":str(train_path.relative_to(ROOT)),"sha256":sha(train_path)},
             "development":{"path":str(dev_path.relative_to(ROOT)),"sha256":sha(dev_path)},
             "validation":{"path":str(val_path.relative_to(ROOT)),"sha256":sha(val_path)},"test":None}
    task_base = {"protocol":"belief-controller-mechanism-v1","horizon":36,"macro_max_steps":3,
        "speed":0.45,"goal_tolerance":0.18,"sensor_range":1.25,"landmark_count":9,"gamma":0.99,
        "coverage_threshold":0.85,"map_accuracy_threshold":0.7,"landmark_error_threshold":0.35,
        "pose_error_threshold":0.25,"coverage_weight":10.0,"information_weight":0.001,
        "travel_weight":0.01,"collision_weight":0.1,"byte_weight":1e-6,
        "prior_sigmas":[0.05,0.05,0.02],"motion_std":[0.015,0.015,0.01],
        "odometry_sigmas":[0.05,0.05,0.02],"observation_sigmas":[0.02,0.1],
        "sensor_min_range":0.25,"revisit_standoff":0.5,"coordination_weight":0.05,"group_cue_weight":0.25}
    quantum={"device":"default.qubit","shots":None,"dtype":"float64/complex128","diff_method":"backprop",
             "wires":4,"layers":1,"observable":"ZZZZ","physical_hardware":False,
             "intervention":"Bell input preparation only for eqmarl"}
    index_path=ROOT/"experiments/12/index.json"
    index=json.loads(index_path.read_text())
    if any(row.get("phase") == "pilot" for row in index["runs"]):
        raise RuntimeError("Pilot matrix already registered")
    for condition, alpha in CONDITIONS:
        for seed in SEEDS:
            for method in METHODS:
                rate=rates[method]
                run_id=f"12-pilot-{condition}-{method}_seed-{seed}_attempt-01"
                task=copy.deepcopy(task_base); task["complementarity_alpha"]=alpha
                config={"stage":"12","protocol_id":"mechanism-development-v1",
                  "dependency_completion_commit":"7442b2354fc5f630c3c0753697f60078682f7b10",
                  "model":{"actor":"shared Linear27to16,Tanh,GRUCell16,Linear5; separate agent memory",
                    "encoder":"shared trainable Linear27to2 critic encoder","critic":method,
                    "parameter_counts":{"actor":2165,"critic":COUNTS[method]},
                    "optimizer":{"type":"Adam","learning_rate":rate,"ppo_epochs":2,"clip":0.2,"entropy":0.01}},
                  "quantum":copy.deepcopy(quantum),"task":task,"run_id":run_id,
                  "experiment":"critic-training","method":method,
                  "hypothesis":"Pilot variance and feasibility across the fixed complementarity phase diagram inform only the locked confirmatory execution.",
                  "profile":"pilot","wall_seconds":7200,"max_primitive_steps":12200,"max_cpu_threads":1,
                  "artifact_output_root":"/workspace/experiment-artifacts","actor_seed":seed,
                  "critic_seed":seed+1000,"action_seed":seed+2000,"actor_hidden":16,
                  "learning_rate":rate,"training_steps":10000,"training_wall_seconds":6900,
                  "episodes_per_update":4,"ppo_epochs":2,"ppo_clip":0.2,"entropy_coefficient":0.01,
                  "training_scenarios":training,"validation_scenarios":validation,
                  "interpretation":"Development pilot only. Fixed final weights and greedy validation; no mechanism-test access, method removal, or confirmatory claim.",
                  "seed_streams":{"model_init":seed,"critic_init":seed+1000,
                    "map":{"split":"train","field":"map_seed"},
                    "evaluation_map":{"split":"validation","field":"map_seed"},
                    "sensor":{"splits":["train","validation"],"base_field":"noise_seed",
                      "offsets":{"motion":0,"odometry":1,"sensor":2,"relative":3,"prior":4}},
                    "action":seed+2000,"quantum_shots":None,"communication":None},
                  "split_manifests":split,"critic_method":method,"policy_entropy_is_diagnostic_only":True,
                  "condition":{"name":condition,"complementarity_alpha":alpha},
                  "action_modes":{"training":"sampled","validation":"greedy"},
                  "checkpoint_selection":"final_fixed_budget_weights","analysis_version":"mechanism-development-v1"}
                path=STAGE/f"{run_id}.yaml"; path.write_text(json.dumps(config,indent=2)+"\n")
                index["runs"].append({"run_id":run_id,"status":"planned","phase":"pilot",
                    "condition":condition,"config":str(path.relative_to(ROOT)),"config_sha256":sha(path)})
    index_path.write_text(json.dumps(index,indent=2)+"\n")


if __name__ == "__main__":
    main()
