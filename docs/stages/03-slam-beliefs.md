# Stage 03 — Genuine 2D SLAM beliefs

Status: running. Dependency completion: `84bd6c8c150becb84ca5cdd65dcb855731d1db9c`.
Branch `stage/03-slam-beliefs` was created from that exact verified commit.

## Question and falsifier

Can a real pose/landmark smoother represent uncertainty, benefit from
informative revisits in known space, and revise historical estimates without
counting duplicate information twice? The smallest falsifier is a same-step
prediction/no-measurement graph and its measured counterpart for which an
observable known-landmark revisit cannot reduce the declared pose marginal.

## Estimator and information contract

Use full-history GTSAM Pose2/Point2 factors with a finite initial pose prior in
a fixed common frame, noisy body-frame odometry, and noisy bearing/range
measurements with known landmark IDs but unknown landmark coordinates. New
landmark initial guesses come from the measurement and estimated pose; they
are not extra coordinate priors. Retain all historical poses. Pose covariance
units/frame must be explicit. Ground truth is confined to sensor generation
and fixed-frame evaluation.

Every factor has a canonical ID, owner and hashed payload. Exact repeats have
no effect; conflicting reuse errors. Cooperation exchanges declared raw
factors once, rather than fusing correlated posteriors as independent priors.
Undelivered teammate factors cannot affect an owner's local belief. Delivered
factors and communication costs are recorded separately.

## Required gates and resource limits

Validate the actual backend installation, an anchored measurement graph and
finite marginals before pinning its lock. If NumPy ABI compatibility requires
isolation, use a separately locked process behind the same estimator adapter.

Run fixed straight and closed-loop trajectories; a controlled informative
revisit versus the same-step prediction; historical trajectory correction;
first observation of an unconstrained landmark; duplicate/conflicting factor
IDs; relative robot observation; ownership and delivered-factor isolation.
Report current pose error, trajectory ATE/RPE and landmark-map error in the
fixed reference frame. Persist the factor graph and replay it from its archive.

Diagnostic family cap: 3,600 seconds; cumulative stage cap: 86,400 seconds.
One CPU thread per process and no training, quantum calls or new paid compute.
The exact fixtures, seed streams and numeric gates are committed before the
source is published and reportable execution begins. Failures are retained.
