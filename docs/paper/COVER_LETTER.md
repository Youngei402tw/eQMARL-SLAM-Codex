*Draft cover letter. The author must review, edit, and sign it before submission.*

---

To the Editors, *Quantum Information Processing*

Dear Editors,

Please consider the enclosed manuscript, **"Learning When and Where to Spend Scarce Entanglement: A Preregistered Benchmark for Budgeted Routing of Certified Bell Resources,"** for publication as a regular article in *Quantum Information Processing*.

Entanglement-assisted protocols are usually analysed one game at a time with an ideal Bell pair. Networks that serve many users must instead decide which of many competing requests should consume a scarce, imperfect pair, and when to save one. The manuscript formalises this as a sequential allocation problem over dynamic request graphs. Each request is a weighted XOR nonlocal game whose classical value is exact and whose entangled value is certified against the semidefinite-programming bound. It then asks how well model-free learning, using public information only, can perform this allocation.

The main findings come from a staged protocol in which the test split was hash-locked before being opened once:

- Learned allocators (PPO and DQN) recover about 86–88% of the surplus of the model-aware dynamic-programming optimum, versus about 52% for myopic greedy allocation. They transfer to an unseen eight-agent graph.
- PPO and DQN are practically equivalent within a preregistered margin. Input ablations show that the benefit comes from representing remaining budget, remaining time, and the magnitude and location of certified gains, not from a particular algorithm.
- Heralded generation loss reduces the benefit proportionally. Werner visibility at or below 0.8 removes it entirely, because no certified request in the sealed family keeps a quantum gain below visibility 0.849. We report this fidelity threshold as a central result rather than a footnote.

We think the work fits *Quantum Information Processing* for three reasons. It links certified nonlocal-game values to resource management in quantum networks. It gives a reproducible benchmark with complete raw records, checkpoints, and an independent audit. And it keeps its claims within what the evidence supports. The manuscript does not claim a new Bell violation, a computational speedup, hardware evidence, or general quantum advantage. It reports all null and failed preregistered outcomes, including a failed near-optimality gate and a null PPO-superiority test. It also discloses an overlap between two test catalogs and training dynamics that was found by the independent audit.

All code, configurations, data splits, raw results, checkpoints, and the LaTeX source are publicly available at https://github.com/Youngei402tw/eQMARL-SLAM-Codex. AI coding assistants were used for implementation and drafting under the author's direction, as stated in the manuscript.

The manuscript is not under consideration elsewhere. There are no competing interests.

Sincerely,

Yang-Che Liu
National University of Singapore
yangche.liu@u.nus.edu
