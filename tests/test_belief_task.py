import numpy as np
from eqmarl_slam.belief_task import BeliefTask, DEFAULT_TASK


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
