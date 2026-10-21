"""Full-history 2D pose/landmark smoothing behind an optional GTSAM boundary.

All lengths are metres and angles radians. Pose2 motion and relative-robot
measurements are expressed in the source robot's body frame. Landmark
coordinates are unknown; a first measurement initializes them from its ray,
and does not introduce an independent position prior. Every pose variable is
retained so later observations may revise historical estimates.

This module imports neither GTSAM nor NumPy until the corresponding runtime
operation. ``backend='subprocess'`` supports an isolated optional environment
through a persistent JSON-lines worker; no posterior prior is fused into the
graph when raw factors are delivered.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys


class MissingPoseError(ValueError):
    """A factor refers to a pose whose prior/odometry has not arrived."""


def _vector(value, size, name, *, positive=False):
    value = [float(item) for item in value]
    if len(value) != size or not all(math.isfinite(item) for item in value):
        raise ValueError(f"{name} must contain {size} finite numbers")
    if positive and not all(item > 0 for item in value):
        raise ValueError(f"{name} must be strictly positive")
    return value


def _index(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(record):
    return hashlib.sha256(_canonical(record)).hexdigest()


class _DirectBackend:
    def __init__(self):
        import gtsam
        import numpy as np

        self.gtsam = gtsam
        self.np = np
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.pose_keys = {}
        self.landmark_keys = {}
        self.optimized = False
        self.marginals = None
        self.optimization_count = 0

    def _pose_key(self, robot, step):
        identity = (robot, step)
        if identity not in self.pose_keys:
            raise MissingPoseError(f"Missing pose robot={robot}, step={step}")
        return self.pose_keys[identity]

    def _new_pose_key(self, robot, step):
        if step >= 1_000_000 or robot >= 1_000_000:
            raise ValueError("Pose key allocation requires robot and step below one million")
        return self.gtsam.symbol("x", robot * 1_000_000 + step)

    def _noise(self, sigmas):
        return self.gtsam.noiseModel.Diagonal.Sigmas(self.np.asarray(sigmas, dtype=float))

    def add_factor(self, record):
        gtsam = self.gtsam
        kind = record["kind"]
        if kind == "prior":
            identity = (record["robot"], record["step"])
            key = self._new_pose_key(*identity)
            value = gtsam.Pose2(*record["pose"])
            factor = gtsam.PriorFactorPose2(key, value, self._noise(record["sigmas"]))
            if identity not in self.pose_keys:
                self.pose_keys[identity] = key
                self.values.insert(key, value)
        elif kind == "odometry":
            robot, step = record["robot"], record["step"]
            before = self._pose_key(robot, step - 1)
            after = self._new_pose_key(robot, step)
            delta = gtsam.Pose2(*record["delta"])
            factor = gtsam.BetweenFactorPose2(before, after, delta, self._noise(record["sigmas"]))
            if (robot, step) not in self.pose_keys:
                self.pose_keys[(robot, step)] = after
                self.values.insert(after, self.values.atPose2(before).compose(delta))
        elif kind == "landmark":
            pose_key = self._pose_key(record["robot"], record["step"])
            landmark = record["landmark"]
            landmark_key = gtsam.symbol("l", landmark)
            factor = gtsam.BearingRangeFactor2D(
                pose_key, landmark_key, gtsam.Rot2(record["bearing"]), record["range"],
                self._noise(record["sigmas"]),
            )
            if landmark not in self.landmark_keys:
                pose = self.values.atPose2(pose_key)
                angle = pose.theta() + record["bearing"]
                point = self.np.array([pose.x() + record["range"] * math.cos(angle),
                                       pose.y() + record["range"] * math.sin(angle)])
                self.landmark_keys[landmark] = landmark_key
                self.values.insert(landmark_key, point)
        elif kind == "relative_pose":
            before = self._pose_key(record["robot_a"], record["step_a"])
            after = self._pose_key(record["robot_b"], record["step_b"])
            factor = gtsam.BetweenFactorPose2(before, after, gtsam.Pose2(*record["delta"]),
                                             self._noise(record["sigmas"]))
        else:
            raise ValueError(f"Unsupported factor kind {kind!r}")
        self.graph.add(factor)
        self.optimized = False
        self.marginals = None

    def optimize(self):
        if self.graph.size() == 0:
            raise ValueError("Cannot optimize an empty factor graph")
        if not self.optimized:
            params = self.gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(100)
            params.setRelativeErrorTol(1e-10)
            params.setAbsoluteErrorTol(1e-10)
            self.values = self.gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params).optimize()
            self.optimized = True
            self.optimization_count += 1
        error = float(self.graph.error(self.values))
        if not math.isfinite(error):
            raise FloatingPointError("Nonfinite factor-graph error")
        return {"factor_count": int(self.graph.size()), "pose_count": len(self.pose_keys),
                "landmark_count": len(self.landmark_keys), "graph_error": error,
                "optimization_count": self.optimization_count}

    def pose(self, robot, step):
        self.optimize()
        pose = self.values.atPose2(self._pose_key(robot, step))
        return [pose.x(), pose.y(), pose.theta()]

    def _marginal(self, key):
        self.optimize()
        if self.marginals is None:
            self.marginals = self.gtsam.Marginals(self.graph, self.values)
        covariance = self.np.asarray(self.marginals.marginalCovariance(key), dtype=float)
        if not self.np.isfinite(covariance).all():
            raise FloatingPointError("Nonfinite GTSAM marginal covariance")
        return covariance

    def pose_covariance(self, robot, step, world_frame=True):
        covariance = self._marginal(self._pose_key(robot, step))
        if world_frame:
            pose = self.values.atPose2(self._pose_key(robot, step))
            c, s = math.cos(pose.theta()), math.sin(pose.theta())
            jacobian = self.np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            covariance = jacobian @ covariance @ jacobian.T
        return covariance.tolist()

    def landmarks(self):
        self.optimize()
        return {str(identity): self.np.asarray(self.values.atPoint2(key)).tolist()
                for identity, key in self.landmark_keys.items()}

    def landmark_covariance(self, landmark):
        return self._marginal(self.landmark_keys[landmark]).tolist()

    def trajectory(self, robot):
        self.optimize()
        return {str(step): self.pose(robot, step) for owner, step in sorted(self.pose_keys) if owner == robot}

    def snapshot(self):
        # Initial guesses as well as optimized estimates must survive cloning:
        # replay alone can otherwise initialize a nonlinear graph differently.
        return {
            "poses": [[robot, step, [self.values.atPose2(key).x(), self.values.atPose2(key).y(),
                                      self.values.atPose2(key).theta()]]
                      for (robot, step), key in sorted(self.pose_keys.items())],
            "landmarks": [[identity, self.np.asarray(self.values.atPoint2(key)).tolist()]
                          for identity, key in sorted(self.landmark_keys.items())],
            "optimized": self.optimized,
            "optimization_count": self.optimization_count,
        }

    def restore_snapshot(self, snapshot):
        for robot, step, pose in snapshot["poses"]:
            self.values.update(self._pose_key(robot, step), self.gtsam.Pose2(*pose))
        for identity, point in snapshot["landmarks"]:
            self.values.update(self.landmark_keys[identity], self.np.asarray(point))
        self.optimized = bool(snapshot["optimized"])
        self.optimization_count = int(snapshot["optimization_count"])
        self.marginals = None

    def close(self):
        pass


class _SubprocessBackend:
    """One persistent optional runtime; all RPC messages are JSON data."""
    def __init__(self, python_executable):
        task_env = os.environ.copy()
        task_root = str(Path(__file__).resolve().parents[1])
        task_env["PYTHONPATH"] = task_root + os.pathsep + task_env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [str(python_executable), "-m", "eqmarl_slam.slam", "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=task_env,
        )
        self._request("init")

    def _request(self, method, *args):
        if self.process.poll() is not None:
            raise RuntimeError(f"SLAM worker exited: {self.process.stderr.read()[-4000:]}")
        self.process.stdin.write(json.dumps({"method": method, "args": args}, allow_nan=False) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=60):
                self.process.terminate()
                raise TimeoutError("SLAM worker exceeded 60-second RPC limit")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"SLAM worker produced no reply: {self.process.stderr.read()[-4000:]}")
        response = json.loads(line)
        if not response["ok"]:
            exception = MissingPoseError if response["type"] == "MissingPoseError" else ValueError
            raise exception(response["error"])
        return response.get("result")

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)
        return lambda *args: self._request(method, *args)

    def close(self):
        if self.process.poll() is None:
            try:
                self._request("close")
                self.process.wait(timeout=5)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()


class GTSAMSmoother:
    """Owned raw-factor graph with exact-ID deduplication and history.

    The graph may contain only locally owned factors, or the union of raw
    factors explicitly delivered to its owner. Calling ``merge_factors``
    represents delivery; it is never an implicit global map synchronization.
    A simulator/controller must account for transmission bytes and delivery
    timing. Existing marginal posteriors are never added as new priors.
    """
    def __init__(self, prior_sigmas=(0.05, 0.05, 0.02), odometry_sigmas=(0.05, 0.05, 0.02),
                 observation_sigmas=(0.02, 0.10), relative_sigmas=(0.05, 0.05, 0.02),
                 *, backend="auto", python_executable=None):
        self.prior_sigmas = _vector(prior_sigmas, 3, "prior_sigmas", positive=True)
        self.odometry_sigmas = _vector(odometry_sigmas, 3, "odometry_sigmas", positive=True)
        self.observation_sigmas = _vector(observation_sigmas, 2, "observation_sigmas", positive=True)
        self.relative_sigmas = _vector(relative_sigmas, 3, "relative_sigmas", positive=True)
        if backend not in ("auto", "direct", "subprocess"):
            raise ValueError("backend must be auto, direct or subprocess")
        if backend == "auto":
            backend = "direct" if importlib.util.find_spec("gtsam") is not None else "subprocess"
        if backend == "subprocess":
            if python_executable is None:
                candidates = [os.environ.get("EQMARL_SLAM_PYTHON"),
                              "/workspace/slam-env/bin/python", "/workspace/slam-env/current/bin/python"]
                python_executable = next((path for path in candidates if path and Path(path).is_file()), None)
            if python_executable is None:
                raise ImportError("GTSAM is not installed; declare an optional SLAM runtime executable")
            self._backend = _SubprocessBackend(python_executable)
        else:
            self._backend = _DirectBackend()
        self.backend = backend
        self.python_executable = str(python_executable) if python_executable is not None else None
        self._factors = {}
        self._hashes = {}

    def _add(self, record):
        identity = record["factor_id"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("factor_id must be a nonempty canonical measurement ID")
        digest = _digest(record)
        if identity in self._hashes:
            if self._hashes[identity] != digest:
                raise ValueError(f"Conflicting payload for existing factor ID {identity!r}")
            return False
        self._backend.add_factor(record)
        self._factors[identity] = copy.deepcopy(record)
        self._hashes[identity] = digest
        return True

    def add_prior(self, robot_id, pose, factor_id, *, step=0, sigmas=None):
        return self._add({"factor_id": factor_id, "kind": "prior",
                          "robot": _index(robot_id, "robot_id"), "step": _index(step, "step"),
                          "pose": _vector(pose, 3, "pose"),
                          "sigmas": _vector(self.prior_sigmas if sigmas is None else sigmas, 3, "sigmas", positive=True)})

    def add_odometry(self, robot_id, step, body_delta, factor_id, *, sigmas=None):
        step = _index(step, "step")
        if step == 0:
            raise ValueError("Odometry step must follow an existing prior pose")
        return self._add({"factor_id": factor_id, "kind": "odometry",
                          "robot": _index(robot_id, "robot_id"), "step": step,
                          "delta": _vector(body_delta, 3, "body_delta"),
                          "sigmas": _vector(self.odometry_sigmas if sigmas is None else sigmas, 3, "sigmas", positive=True)})

    def add_observation(self, robot_id, step, landmark_id, bearing, range_m, factor_id, *, sigmas=None):
        bearing, range_m = float(bearing), float(range_m)
        if not math.isfinite(bearing) or not math.isfinite(range_m) or range_m <= 0:
            raise ValueError("Bearing must be finite and range positive")
        # Preserve already canonical floating-point payloads exactly so
        # exporting and replaying a measurement cannot alter its hash.
        if not -math.pi <= bearing <= math.pi:
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return self._add({"factor_id": factor_id, "kind": "landmark",
                          "robot": _index(robot_id, "robot_id"), "step": _index(step, "step"),
                          "landmark": _index(landmark_id, "landmark_id"),
                          "bearing": bearing, "range": range_m,
                          "sigmas": _vector(self.observation_sigmas if sigmas is None else sigmas, 2, "sigmas", positive=True)})

    def add_relative(self, robot_a, step_a, robot_b, step_b, relative_pose, factor_id, *, sigmas=None):
        a = (_index(robot_a, "robot_a"), _index(step_a, "step_a"))
        b = (_index(robot_b, "robot_b"), _index(step_b, "step_b"))
        if a == b:
            raise ValueError("Relative observations require two different pose variables")
        return self._add({"factor_id": factor_id, "kind": "relative_pose",
                          "robot_a": a[0], "step_a": a[1], "robot_b": b[0], "step_b": b[1],
                          "delta": _vector(relative_pose, 3, "relative_pose"),
                          "sigmas": _vector(self.relative_sigmas if sigmas is None else sigmas, 3, "sigmas", positive=True)})

    def add_measurement(self, measurement):
        """Adapt simulator packets; covariance entries are variances.

        Simulator landmark values/covariances use [range,bearing]. GTSAM
        observation noise and ``add_observation`` use [bearing,range].
        """
        record = dict(measurement)
        identity, kind = record["measurement_id"], record["kind"]
        variance = _vector(record["covariance"], 2 if kind == "landmark" else 3,
                           "covariance diagonal", positive=True)
        sigmas = [math.sqrt(value) for value in variance]
        if kind == "prior":
            robot, step = record["to_pose"]
            return self.add_prior(robot, record["value"], identity, step=step, sigmas=sigmas)
        if kind == "odometry":
            robot, step = record["to_pose"]
            if record["from_pose"] != [robot, step - 1]:
                raise ValueError("Odometry packet must join consecutive poses of one robot")
            return self.add_odometry(robot, step, record["value"], identity, sigmas=sigmas)
        if kind == "landmark":
            robot, step = record["to_pose"]
            distance, bearing = record["value"]
            return self.add_observation(robot, step, record["landmark_id"], bearing, distance,
                                        identity, sigmas=[sigmas[1], sigmas[0]])
        if kind == "relative_pose":
            a, b = record["from_pose"], record["to_pose"]
            return self.add_relative(*a, *b, record["value"], identity, sigmas=sigmas)
        raise ValueError(f"Unsupported simulator measurement kind {kind!r}")

    def _replay(self, record):
        kind, identity, sigmas = record["kind"], record["factor_id"], record["sigmas"]
        if kind == "prior":
            return self.add_prior(record["robot"], record["pose"], identity, step=record["step"], sigmas=sigmas)
        if kind == "odometry":
            return self.add_odometry(record["robot"], record["step"], record["delta"], identity, sigmas=sigmas)
        if kind == "landmark":
            return self.add_observation(record["robot"], record["step"], record["landmark"], record["bearing"],
                                        record["range"], identity, sigmas=sigmas)
        if kind == "relative_pose":
            return self.add_relative(record["robot_a"], record["step_a"], record["robot_b"], record["step_b"],
                                     record["delta"], identity, sigmas=sigmas)
        raise ValueError(f"Unsupported delivered raw factor kind {kind!r}")

    def merge_factors(self, delivered):
        """Union explicitly delivered raw factors, preserving measurement IDs."""
        pending = copy.deepcopy(list(delivered))
        for record in pending:
            identity = record["factor_id"]
            if identity in self._hashes and self._hashes[identity] != _digest(record):
                raise ValueError(f"Conflicting payload for existing factor ID {identity!r}")
        inserted = 0
        while pending:
            deferred = []
            for record in pending:
                try:
                    inserted += int(self._replay(record))
                except MissingPoseError:
                    deferred.append(record)
            if len(deferred) == len(pending):
                raise MissingPoseError("Delivered packet lacks required prior or odometry history")
            pending = deferred
        return inserted

    @property
    def factor_count(self):
        return len(self._factors)

    def export_factors(self):
        return copy.deepcopy(list(self._factors.values()))

    def clone(self):
        clone = type(self)(self.prior_sigmas, self.odometry_sigmas, self.observation_sigmas,
                           self.relative_sigmas, backend=self.backend,
                           python_executable=self.python_executable)
        clone.merge_factors(self.export_factors())
        clone._backend.restore_snapshot(self._backend.snapshot())
        return clone

    def optimize(self):
        return self._backend.optimize()

    def pose(self, robot_id, step):
        import numpy as np
        return np.asarray(self._backend.pose(_index(robot_id, "robot_id"), _index(step, "step")), dtype=float)

    def pose_covariance(self, robot_id, step, *, world_frame=True):
        """3x3 covariance: world dx/dy/dtheta, or GTSAM local tangent dx/dy/dtheta."""
        import numpy as np
        return np.asarray(self._backend.pose_covariance(_index(robot_id, "robot_id"),
                                                        _index(step, "step"), bool(world_frame)), dtype=float)

    def landmarks(self):
        import numpy as np
        return {int(identity): np.asarray(point, dtype=float) for identity, point in self._backend.landmarks().items()}

    def landmark_covariance(self, landmark_id):
        import numpy as np
        return np.asarray(self._backend.landmark_covariance(_index(landmark_id, "landmark_id")), dtype=float)

    def trajectory(self, robot_id):
        import numpy as np
        return {int(step): np.asarray(pose, dtype=float)
                for step, pose in self._backend.trajectory(_index(robot_id, "robot_id")).items()}

    def close(self):
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _serve():
    backend = None
    allowed = {"add_factor", "optimize", "pose", "pose_covariance", "landmarks",
               "landmark_covariance", "trajectory", "snapshot", "restore_snapshot"}
    for line in sys.stdin:
        stopping = False
        try:
            request = json.loads(line)
            method, args = request["method"], request.get("args", [])
            if method == "init":
                if backend is not None:
                    raise ValueError("Worker is already initialized")
                backend = _DirectBackend()
                result = {"backend": "gtsam", "protocol": "json-lines-v1"}
            elif method == "close":
                result, stopping = None, True
            elif backend is not None and method in allowed:
                result = getattr(backend, method)(*args)
            else:
                raise ValueError("Unknown worker request or uninitialized backend")
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(response, allow_nan=False), flush=True)
        if stopping:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.parse_args()
    _serve()
