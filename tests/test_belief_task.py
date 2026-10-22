import numpy as np
from eqmarl_slam.belief_task import BeliefTask, DEFAULT_TASK, coordination_score


def test_belief_actor_does_not_receive_hidden_truth():
    env = BeliefTask(DEFAULT_TASK,{"scenario_id":"unit","map_seed":3,"noise_seed":4})
    before, masks = env.reset()
    assert before.shape == (2,27)
    env.sim._poses[1,:2] += 1
    after,_ = env.observe()
    np.testing.assert_array_equal(before,after)
    assert masks[:,0].all()
    env.close()


def test_macro_steps_and_discount_use_actual_duration():
    env = BeliefTask(DEFAULT_TASK,{"scenario_id":"unit-duration","map_seed":5,"noise_seed":6})
    env.reset()
    _,_,reward,done,info = env.step(np.array([0,0]),primitive_limit=1)
    assert info["duration"] == 1
    assert reward == info["undiscounted_reward"]
    assert env.metrics()["primitive_steps"] == 1
    env.close()


def test_near_field_measurements_are_censored_and_recorded():
    env = BeliefTask(DEFAULT_TASK,{"scenario_id":"unit-nearfield","map_seed":5,"noise_seed":6})
    env.reset()
    env.sim._landmarks[0] = env.sim._poses[0,:2]+[.01,0]
    packets = env.sim.observe()
    assert not any(m["owner"] == 0 and m["landmark_id"] == 0 for m in packets)
    assert any(m["status"] == "discarded_near_field" for m in env.sim.sensor_events)
    env.close()


def test_mechanism_reward_is_scale_matched_and_interaction_monotone():
    interactions = []
    for alpha in [0., .5, 1.]:
        table = {(a,b):coordination_score([a,b],alpha) for a in [0,1] for b in [0,1]}
        assert np.isclose(np.mean(list(table.values())), 1.)
        interactions.append(table[1,1]-table[1,0]-table[0,1]+table[0,0])
    assert interactions[0] == 0
    assert interactions[0] < interactions[1] < interactions[2]


def test_mechanism_observation_contract_and_reward_accounting():
    task = {**DEFAULT_TASK, "protocol":"belief-controller-mechanism-v1",
            "complementarity_alpha":1., "coordination_weight":.05, "group_cue_weight":.25}
    env = BeliefTask(task,{"scenario_id":"unit-mechanism","map_seed":17,"noise_seed":18})
    obs, _ = env.reset()
    assert obs.shape == (2,27)
    _,_,reward,_,info = env.step(np.array([0,0]),primitive_limit=1)
    assert "coordination" in info["reward_components"][0]
    assert np.isclose(reward,sum(info["reward_components"][0].values()))
    env.close()
