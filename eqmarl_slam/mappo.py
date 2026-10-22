"""Common recurrent PPO: complete episode unrolls and duration-aware GAE."""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .belief_task import BeliefTask
from .classical import SharedGRUActor
from .checkpoint import save_checkpoint


class ClassicalCritic(nn.Module):
    def __init__(self, obs_dim=27, latent_dim=2, hidden=32):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, latent_dim, dtype=torch.float64)
        self.head = nn.Sequential(nn.Linear(2*latent_dim, hidden, dtype=torch.float64), nn.Tanh(),
                                  nn.Linear(hidden, 1, dtype=torch.float64))

    def forward(self, observations):
        return self.head(self.encoder(observations).flatten(start_dim=-2)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, critic=None, actor_seed=1, critic_seed=2, hidden=16):
        super().__init__()
        torch.manual_seed(actor_seed)
        self.actor = SharedGRUActor(27, hidden, 5).double()
        torch.manual_seed(critic_seed)
        self.critic = critic if critic is not None else ClassicalCritic()


def duration_gae(rewards, values, next_value, durations, terminals, gamma=.99, lam=.95):
    """Rewards already include primitive-time discount within each macro."""
    advantages = np.zeros(len(rewards), dtype=np.float64)
    carry = 0.
    for t in reversed(range(len(rewards))):
        continuation = 0. if terminals[t] else 1.
        bootstrap = next_value if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma**durations[t]*continuation*bootstrap - values[t]
        carry = delta + (gamma*lam)**durations[t]*continuation*carry
        advantages[t] = carry
    return advantages, advantages + np.asarray(values)


def collect_episode(model, task_config, scenario, action_generator, *, primitive_limit, training=True, on_transition=None, on_decision=None):
    env = BeliefTask(task_config, scenario)
    observation, mask = env.reset()
    if on_transition is not None: env.on_primitive = on_transition
    hidden = model.actor.initial_state(agents=2)
    rows, total_return = [], 0.
    component_totals = {k:0. for k in ['coverage','information','travel','collision','communication','coordination']}
    primitive_steps = 0
    done = False
    while not done and primitive_steps < primitive_limit:
        obs = torch.tensor(observation, dtype=torch.float64)
        flags = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            distribution, hidden = model.actor(obs, hidden, flags)
            action = (torch.multinomial(distribution.probs, 1, generator=action_generator).squeeze(-1)
                      if training else distribution.probs.argmax(-1))
            old_log = distribution.log_prob(action).clone()
            value = float(model.critic(obs))*10.
        if on_decision is not None: on_decision()
        observation_next, mask_next, reward, done, info = env.step(action.numpy(), primitive_limit=primitive_limit-primitive_steps)
        duration = info["duration"]
        rows.append({"obs":observation.copy(),"mask":mask.copy(),"actions":action.numpy(),"old_log":old_log.numpy(),
                     "value":value,"reward":reward,"duration":duration,"terminal":done})
        primitive_steps += duration; total_return += info["undiscounted_reward"]
        for parts in info['reward_components']:
            for key, amount in parts.items(): component_totals[key] += amount
        observation, mask = observation_next, mask_next
    with torch.no_grad():
        bootstrap = 0. if done else float(model.critic(torch.tensor(observation, dtype=torch.float64)))*10.
    advantages, targets = duration_gae([r["reward"] for r in rows], [r["value"] for r in rows], bootstrap,
                                      [r["duration"] for r in rows], [r["terminal"] for r in rows], gamma=task_config["gamma"])
    for row, advantage, target in zip(rows, advantages, targets):
        row.update(advantage=advantage, target=target)
    metrics = env.metrics()
    env.close()
    action_counts = [[sum(int(row['actions'][agent])==choice for row in rows) for choice in range(5)] for agent in range(2)]
    return rows, {**metrics,"return":total_return,"macro_decisions":len(rows),"external_collection_cutoff":not done,
                  "scenario_id":scenario["scenario_id"],"action_counts_by_agent":action_counts,
                  "macro_duration_counts":{duration:sum(row['duration']==duration for row in rows) for duration in [1,2,3]},
                  "reward_component_totals":component_totals}


def update(model, optimizer, episodes, *, epochs=2, clip=.2, entropy_coefficient=.01, on_update=None):
    flat = [row for episode in episodes for row in episode]
    advantages = np.array([row["advantage"] for row in flat])
    mean, std = advantages.mean(), max(advantages.std(), 1e-8)
    history = []
    for _ in range(epochs):
        actor_terms, values, targets, entropy, ratios = [], [], [], [], []
        for episode in episodes:
            hidden = model.actor.initial_state(agents=2)
            for row in episode:
                obs = torch.tensor(row["obs"], dtype=torch.float64)
                flags = torch.tensor(row["mask"], dtype=torch.bool)
                distribution, hidden = model.actor(obs, hidden, flags)
                actions = torch.tensor(row["actions"], dtype=torch.int64)
                ratio = torch.exp(distribution.log_prob(actions)-torch.tensor(row["old_log"]))
                advantage = (row["advantage"]-mean)/std
                actor_terms.append(-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean())
                entropy.append(distribution.entropy().mean())
                ratios.append(ratio)
                values.append(model.critic(obs))
                targets.append(row["target"]/10.)
        value_loss = (torch.stack(values)-torch.tensor(targets,dtype=torch.float64)).square().mean()
        policy_loss = torch.stack(actor_terms).mean()
        entropy_value = torch.stack(entropy).mean()
        loss = policy_loss + .5*value_loss - entropy_coefficient*entropy_value
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite recurrent PPO loss")
        optimizer.zero_grad(); loss.backward()
        actor_grad = float(torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5))
        critic_grad = float(torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5))
        optimizer.step()
        history.append({"loss":float(loss.detach()),"policy_loss":float(policy_loss.detach()),
                        "normalized_value_mse":float(value_loss.detach()),"entropy":float(entropy_value.detach()),
                        "actor_gradient_norm":actor_grad,"critic_gradient_norm":critic_grad,
                        "mean_probability_ratio":float(torch.cat(ratios).mean().detach()),
                        "advantage_mean":float(mean),"advantage_std":float(std),
                        "target_min":min(targets),"target_max":max(targets),
                        "target_mean":float(np.mean(targets)),"target_variance":float(np.var(targets)),
                        "normalized_value_mean":float(torch.stack(values).detach().mean()),
                        "normalized_value_variance":float(torch.stack(values).detach().var(unbiased=False)),
                        "critic_gradient_groups_after_clip":{name:float(parameter.grad.norm()) if parameter.grad is not None else None
                                                             for name,parameter in model.critic.named_parameters()}})
        if on_update is not None: on_update(history[-1])
    return history


def train(config, output, emit, *, critic=None, progress=None):
    """One bounded trained seed; checkpoints are explicit new-episode boundaries."""
    torch.set_num_threads(1)
    model = ActorCritic(critic=critic, actor_seed=config["actor_seed"], critic_seed=config["critic_seed"], hidden=config["actor_hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["action_seed"])
    scenarios = config["training_scenarios"]
    consumed = {"primitive_steps":0,"decisions":0,"updates":0,"circuit_evaluations":0,"shots":0}
    if progress is not None: progress["resources"] = consumed
    started, batch, episodes, update_rows = time.perf_counter(), [], [], []
    def charge(duration):
        consumed["primitive_steps"] += duration
        if time.perf_counter()-started >= config["training_wall_seconds"]:
            raise TimeoutError("Registered training wall cap reached")
    def decision():
        consumed["decisions"] += 2
    def updated(row):
        consumed["updates"] += 1
        update_rows.append(row)
        emit({"kind":"ppo_update","update":consumed["updates"],**row})
    index = 0
    while consumed["primitive_steps"] < config["training_steps"]:
        if time.perf_counter()-started >= config["training_wall_seconds"]:
            raise TimeoutError("Registered training wall cap reached")
        scenario = scenarios[index % len(scenarios)]
        rows, metrics = collect_episode(model, config["task"], scenario, generator,
                                        primitive_limit=config["training_steps"]-consumed["primitive_steps"],
                                        on_transition=charge,on_decision=decision)
        index += 1; batch.append(rows); episodes.append(metrics)
        emit({"kind":"training_episode","episode_index":index,**metrics})
        if len(batch)>=config["episodes_per_update"] or consumed["primitive_steps"]>=config["training_steps"]:
            current = update(model, optimizer, batch, epochs=config["ppo_epochs"], clip=config["ppo_clip"],
                             entropy_coefficient=config["entropy_coefficient"],on_update=updated)
            batch.clear()
            save_checkpoint(Path(output)/"final.pt", model, optimizer, recurrent_state=model.actor.initial_state(agents=2),
                            progress={**consumed,"episodes":index}, normalization={"return_mean":0.,"return_scale":10.},
                            environment_state={"boundary":"new_episode_only","next_scenario_index":index,
                                               "action_generator_state":generator.get_state(),
                                               "exact_mid_episode_resume_supported":False})
    validations = []
    for scenario in config["validation_scenarios"]:
        collected, metrics = collect_episode(model, config["task"], scenario, generator,
                                     primitive_limit=config["task"]["horizon"], training=False, on_transition=charge,on_decision=decision)
        validations.append(metrics)
        emit({"kind":"validation_episode","policy_mode":"greedy",**metrics})
    gates = {"finite_updates":bool(update_rows) and all(np.isfinite(row["loss"]) for row in update_rows),
             "actor_connected":any(row["actor_gradient_norm"]>1e-10 for row in update_rows),
             "critic_connected":any(row["critic_gradient_norm"]>1e-10 for row in update_rows),
             "noncollapsed_policy":bool(update_rows) and update_rows[-1]["entropy"]>.05}
    if config.get('policy_entropy_is_diagnostic_only',False):
        del gates['noncollapsed_policy']
        gates['finite_policy_entropy'] = bool(update_rows) and all(np.isfinite(row['entropy']) for row in update_rows)
    return {"passed":all(gates.values()),"gates":gates,"resources":consumed,
            "summary":{"training_steps":config["training_steps"],"training_episodes":len(episodes),
                       "validation":validations,"update_count":len(update_rows),
                       "mean_training_return":float(np.mean([r["return"] for r in episodes])),
                       "fixed_return_normalization":{"mean":0,"scale":10},
                       "checkpoint_selection":"final fixed-budget weights; no outcome selection",
                       "resume_boundary":"new episodes only, no exact mid-episode continuation claim"}}
