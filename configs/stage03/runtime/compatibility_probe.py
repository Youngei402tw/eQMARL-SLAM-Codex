"""Tiny installation probe: anchored Pose2 graph with bearing/range factor.

This validates native wheel operations and NumPy interchange; it is not a
registered trajectory, calibration, revisit, or performance experiment.
"""
import faulthandler
faulthandler.enable()
import importlib.metadata
import json
import math
import platform
import sys
import numpy as np
import gtsam

x0,x1,l0=(gtsam.symbol('x',0),gtsam.symbol('x',1),gtsam.symbol('l',0))
graph=gtsam.NonlinearFactorGraph()
graph.add(gtsam.PriorFactorPose2(x0,gtsam.Pose2(0.,0.,0.),gtsam.noiseModel.Diagonal.Sigmas(np.array([.01,.01,.005]))))
graph.add(gtsam.BetweenFactorPose2(x0,x1,gtsam.Pose2(1.,0.,0.),gtsam.noiseModel.Diagonal.Sigmas(np.array([.1,.1,.05]))))
graph.add(gtsam.BearingRangeFactor2D(x1,l0,gtsam.Rot2(math.pi/4),math.sqrt(2),gtsam.noiseModel.Diagonal.Sigmas(np.array([.03,.08]))))
initial=gtsam.Values()
initial.insert(x0,gtsam.Pose2(.1,-.1,.05))
initial.insert(x1,gtsam.Pose2(1.1,.1,-.05))
initial.insert(l0,np.array([2.1,.8]))
optimized=gtsam.LevenbergMarquardtOptimizer(graph,initial).optimize()
marginals=gtsam.Marginals(graph,optimized)
pose=optimized.atPose2(x1)
landmark=np.asarray(optimized.atPoint2(l0))
pose_cov=np.asarray(marginals.marginalCovariance(x1))
landmark_cov=np.asarray(marginals.marginalCovariance(l0))
passed=bool(np.isfinite(pose_cov).all() and np.isfinite(landmark_cov).all()
            and np.linalg.eigvalsh(pose_cov).min()>0 and np.linalg.eigvalsh(landmark_cov).min()>0
            and np.allclose([pose.x(),pose.y(),pose.theta()],[1,0,0],atol=1e-6)
            and np.allclose(landmark,[2,1],atol=1e-6))
result={'passed':passed,'purpose':'Installation/native-wheel compatibility only; no scientific trajectory result',
        'python':sys.version,'platform':platform.platform(),'gtsam':importlib.metadata.version('gtsam'),
        'numpy':np.__version__,'graph_factors':int(graph.size()),'measurement_factor':'BearingRangeFactor2D',
        'gauge':'Pose0 Gaussian prior world origin; standard deviations x/y=.01m,heading=.005rad',
        'units':{'position':'metres','heading_bearing':'radians','odometry':'body-frame Pose2 between'},
        'pose1':[pose.x(),pose.y(),pose.theta()],'landmark0':landmark.tolist(),
        'pose1_covariance':pose_cov.tolist(),'landmark0_covariance':landmark_cov.tolist(),
        'final_graph_error':float(graph.error(optimized))}
print(json.dumps(result,indent=2))
raise SystemExit(0 if passed else 1)
