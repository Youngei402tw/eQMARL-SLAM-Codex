# Stage 08 — Locked confirmatory comparison

Status: passed. All 24 preregistered outcomes completed, were retained, and were independently retrieved and verified from the remote stage branch.

## Locked design

The study asked whether eQMARL changed paired seed-level held-out physical coverage relative to its matched no-input-entanglement quantum critic under a fixed short learning budget. The centralized classical critic selected in Stage 05 was the strong control. No interaction-model, finite-shot-policy, or noise-robustness claim was tested.

The immutable source was `2ceeb16cae3c4d3bf9d611ef755e7c890ce7440c`. Three methods used eight paired training seeds (8801–8808), 10,000 training primitives per model, 30 validation episodes, and 100 fixed paired held-out test maps per final checkpoint. Training was sampled; validation and test were greedy. Test used the serialized actor only and verified that the checkpoint and actor state remained unchanged. Across all 2,400 test episodes, the critic and optimizer executed zero times.

The primary statistic was eQMARL minus quantum-noent physical coverage after averaging the 100 test episodes within each trained seed. Its interval used 10,000 paired-seed bootstrap resamples. Success and the classical comparisons were secondary descriptive contrasts. Training seeds, rather than test episodes, are the replication unit.

## Results

| Method | Seeds | Physical coverage, mean ± seed SD | Map accuracy | Success | ATE (m) | Collisions | Communication bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| centralized classical | 8 | 0.94868 ± 0.02573 | 0.95764 | 1.00000 | 0.08200 | 0.00000 | 10050.39 |
| quantum, no input entanglement | 8 | 0.94608 ± 0.02253 | 0.96097 | 1.00000 | 0.08360 | 0.00750 | 10736.33 |
| eQMARL | 8 | 0.95038 ± 0.01400 | 0.95875 | 1.00000 | 0.08392 | 0.00000 | 9788.70 |

The primary eQMARL-minus-quantum-noent coverage difference was 0.00431, with a 95% percentile interval of [-0.01458, 0.02615]. The eQMARL-minus-classical difference was 0.00170 [-0.01615, 0.02427], and quantum-noent-minus-classical was -0.00260 [-0.02188, 0.01406]. Every method had held-out success 1.0, so every success contrast was zero.

The complete locked sample therefore does not establish superiority. The interval spanning zero is also not an equivalence result. These are bounded-budget results for the fixed 2D landmark-SLAM task and fixed test manifest; they do not establish convergence, hardware speedup, or a general quantum advantage.

## Gates and resources

All 24 runs passed finite-loss, finite-gradient, final-checkpoint, frozen-test, split-lock, and source/configuration gates. There were no incomplete or failed Stage 08 outcomes to omit. The retained archive contains 24 checkpoints and 217 files. The locked analysis was regenerated from raw hashes in `analysis/08/confirmatory-analysis.json`.

Measured experiment resources were 261,879 primitives, 178,990 decisions, 13,226 optimizer updates, 166,628 analytic circuit evaluations, zero shots, 4,909.147 wall-seconds, and 1.36352 CPU-core-hours. The test phase accounted for 146.226 wall-seconds and 0.04062 CPU-core-hours. The stage remained below its 86,400-second active-wall cap; the documented inactive session gap after the batch is excluded.

Independent remote retrieval at results commit `e834c9f4d7c78e74bc82f5bea2d7e9625e0eb3c8` validated every manifest, resolved configuration, raw metrics file, diagnostic, and inventoried artifact. The machine-readable receipt is `docs/stages/08-publication.json`.
