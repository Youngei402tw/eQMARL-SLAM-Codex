# Stage 03 — Genuine 2D SLAM beliefs

Status: passed. Dependency completion: `84bd6c8c150becb84ca5cdd65dcb855731d1db9c`.
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

## Observed evidence and limits

All13 estimator/replay gates passed in attempt02. The controlled same-step revisit reduced the current3x3 pose marginal trace by0.0082263 in the declared metre/radian convention and logdet by1.19214. Historical poses changed by0.14054 in the recorded pose-vector norm; fixed-frame trajectory translation ATE fell from0.25573m to0.16623m on the deliberately biased square. These are estimator correctness fixtures, not a trained-controller improvement.

An unknown landmark first sighting supplied no invented pose information. Exact duplicate observations changed no covariance (maximumdifference0), conflicting IDs were rejected, and duplicate packets added zero factors. Private teammate updates left the local covariance unchanged. Two508-byte raw-factor deliveries were charged1016bytes; the first added4 factors, the second added0. Persisted raw-factor graphs reproduced poses and marginals within the registered1e-8 tolerance. Straight and closed-loop noisy trajectories retained finite positive marginals and fixed-ID map/trajectory metrics.

Attempt01 failed mechanically after21 motion timestamps: the old YAML reader parsed JSON exponent tolerances as strings. Its152 raw records, failure traceback, raw sensors and partial graph archives are retained. The reader was corrected and attempt02 used exactly the same scientific values and thresholds, completing28 motion timestamps with195 raw records. Total recorded experiment wall across both attempts is0.41024s, CPU0.00011393 core-hours, zero ML updates/quantum calls/GPU use. Active setup envelopes are separately charged, excluding idle time between sessions.

The primary stack and native GTSAM4.3 wheel passed compatibility checks; the source suite passed96 tests, with2 additional parser regressions after the fix. All failed and completed artifacts were published and freshly retrieved with hash verification.

The posterior is a Gaussian linearization marginal conditional on the registered noise, gauge and known-association model. The single-trajectory NEES values are diagnostics and do not establish covariance calibration. Relative Pose2 sensing explicitly assumes heading-capable relative measurements. No occupancy-grid coverage task or learned controller is claimed here; Stage04 must freeze that task separately.
