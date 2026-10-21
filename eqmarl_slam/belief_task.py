"""Versioned holonomic landmark task; policies consume estimated beliefs only."""
import json
import math

import numpy as np

from .simulation import LandmarkSimulator, compose_pose, between_poses, wrap_angle, generate_landmarks
from .slam import GTSAMSmoother
from .diagnostics03 import fixed_frame_metrics


class BoundedSimulator(LandmarkSimulator):
    """Simulator-only walls and swept two-robot contact; no privileged controls."""
    def observe(self):
        measurements = super().observe()
        accepted = []
        for m in measurements:
            r, j = m["owner"], m["landmark_id"]
            distance = np.linalg.norm(self._poses[r,:2]-self._landmarks[j])
            if distance < self.minimum_range or m["value"][0] < self.minimum_range:
                identity = m["measurement_id"]
                if identity not in self._seen_sensor_events:
                    self.sensor_events.append({"measurement_id":identity,"status":"discarded_near_field",
                                               "minimum_range_m":self.minimum_range})
                    self._seen_sensor_events.add(identity)
            else:
                accepted.append(m)
        return accepted

    def step(self, commands):
        previous = self._poses.copy()
        self.timestamp += 1
        proposed = np.array([compose_pose(p, c + self._noise("motion", r, self.timestamp, "execution", self.motion_std))
                             for r, (p, c) in enumerate(zip(previous, commands))])
        blocked = np.any((proposed[:, :2] < .12) | (proposed[:, :2] > 3.88), axis=1)
        delta = (proposed[0, :2] - previous[0, :2]) - (proposed[1, :2] - previous[1, :2])
        separation = previous[0, :2] - previous[1, :2]
        t = np.clip(-separation @ delta / max(float(delta @ delta), 1e-15), 0, 1)
        if np.linalg.norm(separation + t * delta) < .24:
            blocked[:] = True
        self._poses = np.where(blocked[:, None], previous, proposed)
        self.last_collisions = int(blocked.sum())
        self.last_distance = float(np.linalg.norm(self._poses[:, :2] - previous[:, :2], axis=1).sum())
        measurements = []
        for r in range(2):
            measured = between_poses(previous[r], self._poses[r]) + self._noise("odometry", r, self.timestamp, "measurement", self.odom_std)
            measured[2] = wrap_angle(measured[2])
            measurements.append({"kind": "odometry", "measurement_id": self._id("odometry", r, self.timestamp),
                                 "owner": r, "timestamp": self.timestamp, "from_pose": [r, self.timestamp-1],
                                 "to_pose": [r, self.timestamp], "value": measured.tolist(),
                                 "covariance": (self.odom_std**2).tolist(), "units": ["m_body", "m_body", "rad"]})
        self._history.append(self._poses.copy())
        return measurements + self.observe()


DEFAULT_TASK = {"protocol": "belief-controller-v2", "horizon": 36, "macro_max_steps": 3,
                "speed": .45, "goal_tolerance": .18, "sensor_range": 1.25,
                "landmark_count": 9, "gamma": .99, "coverage_threshold": .85,
                "map_accuracy_threshold": .7, "landmark_error_threshold": .35,
                "pose_error_threshold": .25, "coverage_weight": 10., "information_weight": .001,
                "travel_weight": .01, "collision_weight": .1, "byte_weight": 1e-6,
                "prior_sigmas": [.05, .05, .02], "motion_std": [.015, .015, .01],
                "odometry_sigmas": [.05, .05, .02], "observation_sigmas": [.02, .1],
                "sensor_min_range": .25, "revisit_standoff": .5}


class BeliefTask:
    """Actors: own pose/covariance, delivered factors, candidate descriptors.

    Known empty 4m square, unknown landmark coordinates, noisy pose estimates.
    Raw owner factors broadcast at each macro boundary; full UTF8 payload bytes
    are charged, including priors. No fused posterior becomes a new prior.
    """
    obs_dim, n_actions = 27, 5

    def __init__(self, config, scenario):
        if set(DEFAULT_TASK) - set(config):
            raise ValueError("Task config must resolve every DEFAULT_TASK field")
        if config["protocol"] != "belief-controller-v2":
            raise ValueError("Unknown task protocol")
        self.config, self.scenario = dict(config), dict(scenario)
        self.grid = np.array([(x, y) for x in np.linspace(1/3, 11/3, 6) for y in np.linspace(1/3, 11/3, 6)])

    def reset(self):
        c, s = self.config, self.scenario
        landmarks = generate_landmarks(s["map_seed"], c["landmark_count"], bounds=(.3, 3.7, .3, 3.7))
        self.sim = BoundedSimulator([[.5,.5,0],[3.5,3.5,0]], landmarks, motion_std=c["motion_std"],
                                   odom_std=c["odometry_sigmas"], sensor_std=list(reversed(c["observation_sigmas"])),
                                   sensor_range=c["sensor_range"], prior_std=c["prior_sigmas"], sample_initial_prior=True,
                                   seeds={"motion":s["noise_seed"],"odometry":s["noise_seed"]+1,"sensor":s["noise_seed"]+2,
                                          "relative":s["noise_seed"]+3,"prior":s["noise_seed"]+4}, scenario_id=s["scenario_id"])
        self.sim.minimum_range = c["sensor_min_range"]
        self.on_primitive = None
        self.graphs = [GTSAMSmoother(prior_sigmas=c["prior_sigmas"], odometry_sigmas=c["odometry_sigmas"],
                                     observation_sigmas=c["observation_sigmas"], backend="direct") for _ in range(2)]
        self.bytes, self.messages, self.collisions, self.distance = 0, 0, 0, 0.
        self.sent = [set(), set()]
        self.physical_seen = np.zeros(36, bool)
        self.goals = [None, None]
        for measurement in self.sim.priors + self.sim.observe():
            self.graphs[measurement["owner"]].add_measurement(measurement)
        self._broadcast()
        self._physical_sense()
        self.done = False
        return self.observe()

    def _broadcast(self):
        before = self.bytes
        packets = [[f for f in g.export_factors() if f.get("robot", f.get("robot_a")) == r and f["factor_id"] not in self.sent[r]]
                   for r, g in enumerate(self.graphs)]
        for r, packet in enumerate(packets):
            payload = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
            self.bytes += len(payload)
            self.messages += 1
            self.graphs[1-r].merge_factors(json.loads(payload))
            self.sent[r].update(f["factor_id"] for f in packet)
        for graph in self.graphs:
            graph.optimize()
        return self.bytes - before

    def _physical_sense(self):
        for r in range(2):
            position = self.sim.truth_trajectory(r)[-1, :2]
            distances = np.linalg.norm(self.grid-position, axis=1)
            self.physical_seen |= (distances <= self.config["sensor_range"]) & (distances >= self.config["sensor_min_range"])

    def belief(self, robot):
        graph = self.graphs[robot]
        pose = graph.pose(robot, self.sim.timestamp)
        covariance = graph.pose_covariance(robot, self.sim.timestamp)
        seen = np.zeros(36, bool)
        for other in range(2):
            for previous in graph.trajectory(other).values():
                distances = np.linalg.norm(self.grid-previous[:2], axis=1)
                seen |= (distances <= self.config["sensor_range"]) & (distances >= self.config["sensor_min_range"])
        teammate = list(graph.trajectory(1-robot).values())[-1]
        return pose, covariance, seen, teammate, graph.landmarks()

    def _candidates(self, robot, belief):
        pose, covariance, seen, teammate, landmarks = belief
        def footprint(point):
            distances = np.linalg.norm(self.grid-point, axis=1)
            return (distances <= self.config["sensor_range"]) & (distances >= self.config["sensor_min_range"])
        novelty = np.array([np.mean(footprint(point) & ~seen) for point in self.grid])
        scores = novelty - .002*np.linalg.norm(self.grid-pose[:2], axis=1)
        # Identity only breaks symmetric choices; it is not teammate state.
        order = sorted(range(36), key=lambda j: (-scores[j], (j+robot*18)%36))
        first = order[0]
        second = next((j for j in order[1:] if np.linalg.norm(self.grid[j]-self.grid[first])>=1.), order[1])
        candidates = [pose[:2] if self.goals[robot] is None else self.goals[robot], self.grid[first], self.grid[second], pose[:2], teammate[:2]]
        mask = np.ones(5, bool)
        mask[1:3] = [novelty[j] > 0 for j in [first, second]]
        mask[3] = bool(landmarks)
        if landmarks:
            identity = max(landmarks, key=lambda j: np.trace(self.graphs[robot].landmark_covariance(j)))
            direction = pose[:2]-landmarks[identity]
            direction /= max(np.linalg.norm(direction),1e-12)
            candidates[3] = landmarks[identity]+self.config["revisit_standoff"]*direction
        candidates = np.clip(np.asarray(candidates), .25, 3.75)
        gains = np.array([np.mean(footprint(point) & ~seen) for point in candidates])
        return candidates, mask, gains

    def observe(self):
        obs, masks, self.candidate_points, self.candidate_gains = [], [], [], []
        for r in range(2):
            belief = self.belief(r)
            pose, cov, seen, teammate, landmarks = belief
            candidates, mask, gains = self._candidates(r, belief)
            features = np.r_[pose[:2]/4, np.sin(pose[2]), np.cos(pose[2]), np.sqrt(np.maximum(np.diag(cov), 0)),
                             1-self.sim.timestamp/self.config["horizon"], seen.mean(), len(landmarks)/self.config["landmark_count"],
                             (teammate[:2]-pose[:2])/4, np.column_stack([(candidates-pose[:2])/4, gains]).ravel()]
            obs.append(features); masks.append(mask)
            self.candidate_points.append(candidates); self.candidate_gains.append(gains)
        return np.asarray(obs, dtype=np.float64), np.asarray(masks)

    def heuristic(self, mode, rng):
        _, masks = self.observe()
        if mode == "random":
            return np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])
        actions, selected = [], []
        for r in range(2):
            if mode == "uncertainty" and np.trace(self.belief(r)[1][:2,:2]) > .02 and masks[r,3]:
                action = 3
            else:
                scores = self.candidate_gains[r].copy()
                scores[[0,3,4]] = -1
                if mode == "reactive":
                    scores = np.array([0, 1, .9, -1, -1])
                for j, point in enumerate(self.candidate_points[r]):
                    if any(np.linalg.norm(point-other)<.8 for other in selected):
                        scores[j] -= 1
                scores[~masks[r]] = -np.inf
                action = int(np.argmax(scores))
            actions.append(action); selected.append(self.candidate_points[r][action])
        return np.array(actions)

    def step(self, actions, *, primitive_limit=None):
        if self.done:
            raise RuntimeError("reset required")
        _, masks = self.observe()
        actions = np.asarray(actions)
        if actions.shape != (2,) or not np.issubdtype(actions.dtype, np.integer) or np.any((actions<0)|(actions>=5)) or not all(masks[r,a] for r,a in enumerate(actions)):
            raise ValueError("infeasible joint action")
        self.goals = [self.candidate_points[r][a].copy() for r,a in enumerate(actions)]
        c = self.config
        maximum = min(c["macro_max_steps"], c["horizon"]-self.sim.timestamp,
                      primitive_limit if primitive_limit is not None else c["macro_max_steps"])
        if maximum < 1:
            raise ValueError("positive primitive budget required")
        rewards, trace = [], []
        for _ in range(maximum):
            commands = []
            for r in range(2):
                pose = self.graphs[r].pose(r, self.sim.timestamp)
                delta = self.goals[r] - pose[:2]
                delta *= min(1., c["speed"]/max(np.linalg.norm(delta), 1e-12))
                cs, sn = np.cos(pose[2]), np.sin(pose[2])
                commands.append([cs*delta[0]+sn*delta[1], -sn*delta[0]+cs*delta[1], 0])
            before = self.physical_seen.mean()
            measurements = self.sim.step(commands)
            if self.on_primitive is not None: self.on_primitive(1)
            information = 0.
            for r, graph in enumerate(self.graphs):
                own = [m for m in measurements if m["owner"] == r]
                for m in own:
                    if m["kind"] == "odometry": graph.add_measurement(m)
                predicted = graph.pose_covariance(r, self.sim.timestamp)
                for m in own:
                    if m["kind"] == "landmark": graph.add_measurement(m)
                posterior = graph.pose_covariance(r, self.sim.timestamp)
                information += np.linalg.slogdet(predicted)[1]-np.linalg.slogdet(posterior)[1]
            self._physical_sense()
            self.collisions += self.sim.last_collisions
            self.distance += self.sim.last_distance
            parts = {"coverage": c["coverage_weight"]*(self.physical_seen.mean()-before),
                     "information": c["information_weight"]*information,
                     "travel": -c["travel_weight"]*self.sim.last_distance,
                     "collision": -c["collision_weight"]*self.sim.last_collisions, "communication":0.}
            rewards.append(sum(parts.values())); trace.append(parts)
            if all(np.linalg.norm(self.goals[r]-self.graphs[r].pose(r,self.sim.timestamp)[:2])<c["goal_tolerance"] for r in range(2)):
                break
        fee = self._broadcast()*c["byte_weight"]
        rewards[-1] -= fee; trace[-1]["communication"] = -fee
        metrics = self.metrics()
        terminated = metrics["success"] or self.sim.timestamp>=c["horizon"]
        self.done = bool(terminated)
        duration = len(rewards)
        discounted = sum(c["gamma"]**i*r for i,r in enumerate(rewards))
        return *self.observe(), discounted, self.done, {"duration":duration,"reward_components":trace,
                                                       "undiscounted_reward":sum(rewards),"metrics":metrics}

    def metrics(self):
        truth = self.sim.truth_landmarks()
        estimated = self.graphs[0].landmarks()
        correct = sum(j in estimated and np.linalg.norm(estimated[j]-point)<=self.config["landmark_error_threshold"] for j,point in truth.items())
        coverage = float(self.physical_seen.mean())
        accuracy = correct/len(truth)
        current, ates, rpes = [], [], []
        for r in range(2):
            actual = self.sim.truth_trajectory(r)
            estimates = np.array([self.graphs[r].pose(r,t) for t in range(len(actual))])
            metrics = fixed_frame_metrics(actual, estimates)
            current.append(metrics["current_translation_error_m"])
            ates.append(metrics["translation_ate_rmse_m"])
            if metrics["translation_rpe_rmse_m"] is not None: rpes.append(metrics["translation_rpe_rmse_m"])
        return {"physical_coverage":coverage,"estimated_coverage":float(self.belief(0)[2].mean()),
                "fixed_landmark_accuracy":accuracy,"fixed_landmark_count":len(truth),"correct_landmark_count":int(correct),
                "pose_error_m":float(np.mean(current)),"trajectory_ate_m":float(np.mean(ates)),
                "trajectory_rpe_m":float(np.mean(rpes)) if rpes else None,
                "primitive_steps":self.sim.timestamp,"travel_distance_m":self.distance,"collisions":self.collisions,
                "communication_bytes":self.bytes,"messages":self.messages,
                "discarded_sensor_events":len(self.sim.sensor_events),
                "success":bool(coverage>=self.config["coverage_threshold"] and accuracy>=self.config["map_accuracy_threshold"]
                               and np.mean(current)<=self.config["pose_error_threshold"]),
                "localization_failure":bool(np.mean(current)>1.)}

    def close(self):
        for graph in self.graphs: graph.close()
