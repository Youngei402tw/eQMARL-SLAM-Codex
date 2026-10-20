# Research direction and project ranking

These are planning judgments, not measured success probabilities or claims of novelty. Recheck the primary literature before describing a contribution as new. First execute the core 2D project; do not turn every extension into a simultaneous workstream.

| Priority | Project | Feasibility | Potential contribution | Evidence needed |
|---|---|---|---|---|
| 1 | Reproducible eQMARL active-SLAM benchmark with fair critic ablations | High | Medium to high: resolves confounded comparisons and establishes an honest baseline | Genuine uncertainty, strong classical baselines, matched no-entanglement intervention, complete multi-seed archives |
| 2 | Quantum interaction critic for revisit/rendezvous decisions | Medium | High if the interaction mechanism is supported; a useful negative result remains possible | Coupling-controlled tasks, pairwise classical/bilinear control, interaction ablation, held-out transfer |
| 3 | Training quantum resources versus execution communication | Medium to high | Medium to high: a practically useful resource frontier | Explicit bytes/delay/dropout at execution; circuit calls/shots/entanglement resources during training; equal information access |
| 4 | Finite-shot robustness and resource-efficient training | Medium | Medium to high when it improves measured cost and reliability | Fixed backend/differentiation semantics, actual backward-pass shots, matched training budget, strong analytic baseline |
| 5 | Larger teams with a sparse interaction graph | Medium to low | High if quality and total cost scale favorably | Independent small circuits per selected edge, graph classical baseline, generalization across team sizes, no hidden global state |
| 6 | ROS/Gazebo validation with a standard 2D SLAM backend | Medium | Medium: valuable external validity, usually limited novelty on its own | Same high-level policy interface, timestamp/frame consistency, estimator access contract, realistic sensor and control delays |
| 7 | 3D active SLAM with viewpoint and occlusion decisions | Low initially | Medium to high only for a specific new mechanism | A hypothesis unavailable in 2D, affordable simulator, common visual encoder, fair quantum/classical bottleneck and compute |
| 8 | Execution-time entangled policies or networked quantum robots | Low | Potentially high, with different assumptions and stronger prior-art risk | Explicit quantum network model and classical communication baseline; physical resource/error constraints; updated literature comparison |

## Why 2D first

A useful quantum critic experiment needs complementary local information and cooperative choices, not necessarily three-dimensional rendering. Two robots can already face coupled choices between visiting separate frontiers, revisiting landmarks to reduce uncertainty, and meeting to exchange relative constraints. A compact 2D environment makes those mechanisms observable and permits substantially more controls within the same research budget.

Use a MiniGrid-style task to test observation contracts, planning, reward, credit assignment, and logging. Label known-pose mapping as exploration. The core SLAM task adds noisy motion, observations of initially unknown landmarks, estimator uncertainty, and informative revisits. Known landmark identifiers simplify data association initially; they do not make the landmark positions known.

Move to a standard 2D library through an adapter before investing in a 3D stack. The first estimator target is GTSAM with Pose2 and landmark factors. A simple EKF is an explicitly labeled alternative, not a silent replacement when dependencies fail. Keep the training interface independent of the estimator implementation.

## A focused mechanism hypothesis

Hypothesis: when one robot's most useful high-level action depends on uncertainty or observations available to another robot, an interaction critic may improve value fitting or actor learning under a fixed representation and training budget.

This does not predict a universal quantum advantage. Separate low-coupling and high-coupling map families, vary only a preregistered coordination property, and test a treatment-by-coupling interaction. The paired critic must compete with a classical critic given the same graph, encoder features, unary terms, and residual path. If a bilinear model explains the gain, report the architectural lesson instead of attributing it to entanglement alone.

For a maximally entangled pair, the identity `<Phi_d|A ⊗ B|Phi_d> = Tr(A^T B)/d` motivates a classical bilinear control. It does not prove that every deeper quantum model has an efficient general classical simulation. Conversely, a smaller count of centralized trainable parameters is not a measured computational speedup.

## When to advance beyond the core

ROS/Gazebo is justified once the 2D result is reproducible and the estimator interface is stable. Three-dimensional evaluation is justified when viewpoint, occlusion, or vertical geometry changes the tested coordination mechanism. More difficult rendering or larger observations alone do not strengthen an entanglement claim.

Stage 06 can conclude that the proposed interaction model has no detectable benefit. Stage 07 can conclude that finite-shot cost removes a practical gain. Both are legitimate outcomes that still support a careful Stage 08 comparison and Stage 09 report.

## Primary starting references

- [eQMARL paper](https://arxiv.org/abs/2405.17486) and [official implementation](https://github.com/news-vt/eqmarl).
- [PennyLane Torch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).
- [GTSAM](https://github.com/borglab/gtsam).
- [Gymnasium documentation](https://gymnasium.farama.org/).
- [MiniGrid documentation](https://minigrid.farama.org/).
- [Official MAPPO implementation](https://github.com/marlbenchmark/on-policy).
- [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox).

The literature review for optional execution-time quantum policies must be refreshed at the stage start; the original eQMARL training-time critic does not establish that execution-time project as novel.
