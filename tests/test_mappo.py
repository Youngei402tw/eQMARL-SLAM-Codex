import numpy as np
from eqmarl_slam.mappo import duration_gae


def test_duration_discount_and_true_terminal_are_distinct_from_cutoff():
    a,t = duration_gae([1.],[2.],10.,[3],[False],gamma=.5,lam=.8)
    np.testing.assert_allclose(t,[2.25])
    a,t = duration_gae([1.],[2.],10.,[3],[True],gamma=.5,lam=.8)
    np.testing.assert_allclose(t,[1.])


def test_gae_does_not_cross_terminal_boundary():
    a,t = duration_gae([1.,100.],[0.,0.],0.,[2,1],[True,True],gamma=.9,lam=.95)
    np.testing.assert_allclose(t,[1.,100.])
