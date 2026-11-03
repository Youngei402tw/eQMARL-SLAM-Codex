# Stage 26 — learned allocator benchmarks and mechanism ablations

## Preflight

**Question.** Under identical public information and training opportunities, how does the registered graph-conditioned Double DQN compare with ordinary DQN, feed-forward PPO, and recurrent PPO on non-test graphs and request catalogs?

**Falsifier.** The proposed method fails its development explanation if it does not improve on the strongest independently tuned learned comparator, or if removing budget/horizon/gain information has no measurable validation effect. Such an outcome remains publishable as a null development result and cannot be repaired by opening the final test.

**Deliverables.** Graph-size-independent environment and action interface; four equal-budget learned methods; all tuning checkpoints and raw validation rows; budget, horizon, binary-gain, and shuffled-gain ablations; throughput and resource records; selected configurations; and a locked Stage 27 input.

**Information boundary.** Every method observes only public certified gains, active public edges, agent count, remaining budget, and remaining horizon. Private request inputs, realized future states, and the transition matrix are excluded. The model-aware solution is evaluated only as a reference.

**Budget.** Five CPU cores and 17 GiB are available. One reportable job may use at most four threads, 1.5 million decisions, and three active wall-hours. The measured smoke rates must project below 80% of the registered wall cap before tuning begins.

**Gate.** Tests for matching feasibility, noise monotonicity, action features, tiny DQN/PPO training, and oracle dominance must pass. Each method receives three configurations, two seeds per configuration, 1,200 episodes per seed, 300 paired validation sequences per catalog, and final-checkpoint selection. All four ablations receive the same selected proposed-method configuration and two fresh seeds. Scientific superiority is not required to pass this implementation/development stage.

## Attempt 01 retained failure

All 24 tuning trials, eight feature-ablation trials, 30 non-test request certificates, and 32 final checkpoints completed. Bundle closure then failed because the direct JSON writer received a NumPy boolean in the decision-cap gate. The failed bundle is retained verbatim. Its raw metrics hash is `793f34959848482c8300aeb58cf13a01edff177b408325224944accd05ec02d6`. Attempt 02 is restricted to hash-locked analysis recovery and cannot retrain or access test data.
