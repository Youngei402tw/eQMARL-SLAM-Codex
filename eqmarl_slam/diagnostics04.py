"""Preregistered controller/task falsifiers and installation-size learning test."""
from pathlib import Path
import time

import numpy as np
import torch

from .belief_task import BeliefTask
from .classical import SharedGRUActor
from .mappo import train


def run_diagnostics(config, output, emit, progress=None):
    torch.set_num_threads(1)
    resources = {"primitive_steps":0,"decisions":0,"updates":0,"circuit_evaluations":0,"shots":0}
    if progress is not None: progress["resources"] = resources
    gates, episodes = {}, []
    task, scenarios = config["task"], config["development_scenarios"]
    env = BeliefTask(task, scenarios[0]); obs, mask = env.reset()
    before = obs.copy()
    env.sim._poses[1,:2] += 1
    env.sim._landmarks[0] += 1
    after, _ = env.observe()
    gates["actor_truth_isolation"] = bool(np.array_equal(before,after))
    env.close()
    emit({"kind":"contract_gate","gate":"actor_truth_isolation","passed":gates["actor_truth_isolation"]})
    for method in ["random","reactive","frontier","uncertainty"]:
        for scenario in scenarios:
            env = BeliefTask(task,scenario); env.reset()
            rng = np.random.default_rng(config["action_seed"])
            done, total = False, 0.
            started = time.perf_counter()
            while not done:
                actions = env.heuristic(method,rng)
                obs, mask, reward, done, info = env.step(actions)
                resources["primitive_steps"] += info["duration"]
                resources["decisions"] += 2
                total += info["undiscounted_reward"]
                expected = sum(task["gamma"]**i*sum(parts.values()) for i,parts in enumerate(info["reward_components"]))
                gates["primitive_discount"] = gates.get("primitive_discount", True) and abs(reward-expected)<1e-10
                gates["observations_finite"] = gates.get("observations_finite", True) and bool(np.isfinite(obs).all())
                emit({"kind":"heuristic_macro","method":method,"scenario_id":scenario["scenario_id"],"actions":actions.tolist(),
                      "duration":info["duration"],"discounted_reward":reward,"components":info["reward_components"],
                      "metrics":info["metrics"]})
            metrics = {"method":method,"scenario_id":scenario["scenario_id"],"return":total,
                       "wall_seconds":time.perf_counter()-started,**env.metrics()}
            episodes.append(metrics); emit({"kind":"heuristic_episode",**metrics}); env.close()
    competent = [row for row in episodes if row["method"] in ["frontier","uncertainty"]]
    gates["heuristic_tractability"] = any(row["success"] for row in competent)
    # This supervised discrimination check isolates the recurrent optimization
    # path from task difficulty. It is not called SLAM learning performance.
    torch.manual_seed(config["actor_seed"])
    actor = SharedGRUActor(27,16,5).double()
    optimizer = torch.optim.Adam(actor.parameters(),lr=.02)
    observations = torch.zeros(4,2,27,dtype=torch.float64)
    observations[:2,:,0] = 1; observations[2:,:,0] = -1
    target = torch.tensor([[0,0],[0,0],[1,1],[1,1]])
    flags = torch.zeros(4,2,5,dtype=torch.bool); flags[:,:,:2] = True
    for update in range(100):
        distribution,_ = actor(observations,actor.initial_state((4,),agents=2),flags)
        loss = -distribution.log_prob(target).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step(); resources["updates"] += 1
        if update%20 == 0: emit({"kind":"synthetic_learning_update","update":update+1,"loss":float(loss.detach())})
    distribution,_ = actor(observations,actor.initial_state((4,),agents=2),flags)
    accuracy = float((distribution.probs.argmax(-1)==target).double().mean())
    gates["recurrent_learning_test"] = accuracy>=.95
    emit({"kind":"synthetic_learning_result","accuracy":accuracy,"scope":"synthetic discrimination, not task performance"})
    smoke = dict(config["smoke_training"])
    smoke["task"] = task
    smoke["training_scenarios"] = scenarios
    smoke["validation_scenarios"] = []
    counters = {}
    try:
        result = train(smoke, output, emit, progress=counters)
    finally:
        for key,value in counters.get("resources", {}).items(): resources[key] += value
    gates.update(result["gates"])
    return {"passed":all(gates.values()),"gates":gates,"resources":resources,
            "summary":{"heuristics":episodes,"synthetic_discrimination_accuracy":accuracy,
                       "smoke_training":result["summary"],"scope":"Task tractability and bounded software correctness; no comparative learning claim."}}
