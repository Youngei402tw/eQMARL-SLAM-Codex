# Budgeted Entanglement Routing for Sequential Multi-Agent Reinforcement Learning

## Abstract

Entanglement-assisted multi-agent studies commonly assume that a fixed pair of agents can use entanglement whenever needed. We study a different control problem: several decentralized agents share a small, consumable supply of Bell pairs and must decide where and when to use them. Four agents face a changing graph of private-information coordination requests over ten time steps. A resource controller sees only public information and must route six Bell pairs across the episode. Each allocation must be a matching, because an agent cannot simultaneously share independent Bell pairs with multiple partners through one interface. Every selected edge consumes one pair.

We introduce **BER-MARL**, Budgeted Entanglement Routing for Multi-Agent Reinforcement Learning. It uses a masked Deep Q-Network to decide whether to spend or save Bell pairs. Each request is a three-input binary-output coordination game. We calculate its best zero-communication classical value by exhaustive enumeration and independently bound its entangled value with a semidefinite program. This separates the value of an individual Bell pair from the learned decision about when to use it.

In a locked evaluation with 16 independent training seeds and 1,000 paired held-out sequences per seed, BER-MARL achieved mean entanglement surplus `0.5064`, compared with `0.3084` for myopic greedy, `0.2268` for random, and `0.2248` for static allocation. The paired BER-MARL-minus-greedy difference was `0.1980`, with 95% bootstrap interval `[0.1914, 0.2043]`; every training seed was positive. BER-MARL recovered `88.66%` of the exact online dynamic-programming surplus. A preregistered requirement that the one-sided lower bound exceed 90% was not met: the bound was `87.68%`. The strongest near-oracle composite claim is therefore rejected. The supported contribution is a certified sequential benchmark and evidence that learned routing substantially improves the use of scarce entanglement under held-out task and transition distributions.

## 1. The problem in one picture

Suppose four agents have six Bell pairs for an episode lasting ten rounds. At every round, several pairs of agents receive coordination jobs. Some jobs benefit greatly from entanglement, some benefit only slightly, and some do not benefit at all. Jobs can also have different urgency.

The controller must answer:

> Which agent pairs should receive Bell pairs now, and how many pairs should be saved for later?

```mermaid
flowchart LR
    B[Six Bell pairs]
    G1[Round 1<br/>pairwise requests]
    D1{Spend or save?}
    G2[Round 2<br/>new request graph]
    D2{Spend or save?}
    GN[Rounds 3–10<br/>changing opportunities]
    R[Total coordination reward]

    B --> G1 --> D1 --> G2 --> D2 --> GN --> R
    D1 -- selected edges consume pairs --> RB[Smaller remaining budget]
    RB --> G2
```

A greedy controller spends on the most valuable requests visible now. That can be wasteful when rare, urgent requests may appear later. BER-MARL learns this timing tradeoff.

## 2. A concrete round

Consider four agents, `A`, `B`, `C`, and `D`. In one round, the public graph might be:

```mermaid
graph LR
    A((Agent A))
    B((Agent B))
    C((Agent C))
    D((Agent D))

    A -- "gain 0.04" --- B
    A -- "gain 0.01" --- C
    B -- "gain 0.12" --- D
    C -- "gain 0.09" --- D
```

The controller cannot select both `A-B` and `A-C`, because agent `A` would be used twice. It cannot select both `B-D` and `C-D`, because agent `D` would be used twice. Valid allocations include no edge, one active edge, `A-B` with `C-D`, or `A-C` with `B-D`.

The greatest immediate matching is `A-C` plus `B-D`, with gain `0.13`. Spending two Bell pairs may nevertheless be wrong if only two pairs remain and valuable future requests are likely.

```mermaid
flowchart TD
    O[Public graph, request values,<br/>urgency, time, budget]
    P[BER-MARL allocator]
    M[Feasible matching]
    Q[Selected edges use Bell pairs]
    C[Other edges use exact<br/>classical local policies]
    X[Private contexts arrive afterward]
    A[Agents produce local actions]
    Y[Reward and pair consumption]

    O --> P --> M
    M --> Q --> A
    M --> C --> A
    X --> A --> Y
```

Allocation occurs **before** private contexts arrive. This prevents the shared controller from solving the private coordination problem through ordinary communication.

## 3. One coordination request

Each endpoint privately observes one symbol from `{0,1,2}`. Call them `x` and `y`. Each endpoint produces one binary action, `a` and `b`. A public `3 × 3` table specifies the desired parity:

\[
a \oplus b = f(x,y).
\]

An example table is:

| Private `x` / private `y` | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 2 | 1 | 1 | 0 |

If `x=1` and `y=2`, the required parity is `1`, so the agents succeed when their actions differ. Context pairs have nonuniform weights, so some cells occur more often.

### 3.1 Exact classical value

Without communication, the agents use local functions:

\[
a=g(x), \qquad b=h(y).
\]

There are eight deterministic binary functions of a three-valued input. Therefore, only `8 × 8 = 64` deterministic strategy pairs exist. We enumerate all 64. Shared classical randomness cannot improve the expected optimum because it only mixes deterministic strategies and reward is linear in that mixture.

This gives the exact classical value `V_C`.

### 3.2 Bell-assisted value

For an entangled request, the endpoints share one Bell pair:

\[
|\Phi^+\rangle = \frac{|00\rangle+|11\rangle}{\sqrt{2}}.
\]

Each endpoint selects a local measurement using only its own private symbol. The binary measurement outcomes become `a` and `b`; no private symbol is communicated.

Optimizing six local angles gives a Bell-assisted value `V_Q`. An independent XOR-game semidefinite program, or SDP, supplies an upper bound. The optimized policies match those bounds to numerical tolerance.

The value of using a Bell pair on request `e` is:

\[
G_e=u_e\max(0,V_{Q,e}-V_{C,e}),
\]

where `u_e` is urgency. Some committed request types have `G_e=0`, so a good router must avoid wasting pairs on them.

## 4. The sequential benchmark

| Property | Registered value |
|---|---:|
| Agents | 4 |
| Episode length | 10 rounds |
| Initial Bell-pair budget | 6 |
| Active requests per round | 3–5 |
| Private contexts per endpoint | 3 |
| Actions per endpoint | 2 |
| Physical agent edges | 6 |
| Global matching actions | 10 |

The public graph follows a committed Markov process. Common low-value requests and rarer high-urgency requests make saving useful.

At round `t`, the allocator sees the active graph, certified gains, urgency, remaining budget `B_t`, and remaining horizon. It never sees private contexts or post-context messages.

For selected matching `M_t`, immediate entanglement surplus is:

\[
r_t=\sum_{e\in M_t}G_e,
\]

and the budget evolves as:

\[
B_{t+1}=B_t-|M_t|.
\]

The primary outcome is total surplus:

\[
S=\sum_{t=1}^{10}r_t.
\]

This equals total return minus the exact classical zero-message return on the same sequence.

## 5. BER-MARL

BER-MARL uses a masked Deep Q-Network, or DQN. It estimates the future value of every feasible allocation:

\[
Q(s,a)\approx \text{expected current and future surplus after action }a.
\]

Its 14 inputs are six edge gains, six activity bits, remaining budget, and remaining horizon. Two hidden layers contain 64 units each. Its ten outputs represent one empty matching, six single-edge matchings, and three disjoint two-edge matchings. Inactive and over-budget actions are masked.

```mermaid
flowchart LR
    F[14 public features]
    H1[64 units]
    H2[64 units]
    Q[10 long-term values]
    K[Feasibility mask]
    A[Selected matching]

    F --> H1 --> H2 --> Q --> K --> A
```

Training uses experience replay, Adam with learning rate `5×10⁻⁴`, batches of 64, and a target network updated every 200 optimization steps. Each final seed trains for 2,500 episodes on development and validation catalogs. Final evaluation uses a new request family, transition catalog, and sequence stream without adaptation.

## 6. Bell pair, DQN, PPO, DP, and SDP

| Term | Meaning | Role in this paper |
|---|---|---|
| Bell pair | Two entangled qubits | Consumable resource for one selected request |
| DQN | Deep Q-Network | Current BER-MARL allocator |
| PPO | Proximal Policy Optimization | Possible alternative learned allocator; not in the primary experiment |
| DP | Dynamic programming | Exact model-aware sequential reference |
| SDP | Semidefinite programming | Quantum upper bound for one request |

```mermaid
flowchart TB
    R[One private request]
    CE[Enumerate 64 classical strategies]
    BP[Optimize Bell measurements]
    SDP[Compute SDP upper bound]
    GAP[Certified request gain]
    SEQ[Ten-round sequence]
    DQN[DQN learns when to spend]
    DP[DP computes exact model-aware plan]

    R --> CE --> GAP
    R --> BP --> GAP
    R --> SDP --> GAP
    GAP --> SEQ
    SEQ --> DQN
    SEQ --> DP
```

The SDP asks, “How valuable can entanglement be for this request?” The DP asks, “When should a limited supply be spent across the full episode?”

The DP recurrence is:

\[
J_t(s,b)=\max_{M\in\mathcal{M}(s,b)}\left[G(s,M)+\sum_{s'}P(s'\mid s)J_{t+1}(s',b-|M|)\right].
\]

DP knows the true public transition matrix. BER-MARL learns approximate long-term reasoning from experience.

## 7. Comparison methods

1. **Classical exact:** exact local classical strategies and no Bell pairs.
2. **Quantum random:** random feasible matching.
3. **Quantum static:** fixed edge priorities from non-test frequencies.
4. **Quantum greedy:** greatest immediate certified gain.
5. **BER-MARL:** learned long-horizon allocation.
6. **Online DP oracle:** knows public transition probabilities but not future draws.
7. **Future-clairvoyant:** knows the realized future; descriptive ceiling.
8. **Message oracle:** exchanges private contexts; privileged reference.

The fair primary comparison is BER-MARL against greedy. Both receive identical sequences, Bell budgets, request policies, public information, and matching constraints.

## 8. Prospective experimental design

```mermaid
flowchart LR
    S20[Plan and claim gates]
    S21[Certify requests and task]
    S22[Bounded development and freeze]
    S23[Locked held-out evaluation]
    S24[Independent audit]

    S20 --> S21 --> S22 --> S23 --> S24
```

Development tested four DQN configurations with two seeds each, then four fresh pilots. Before final access, the protocol fixed 16 final training seeds, 2,500 episodes per seed, 1,000 paired held-out sequences, final-checkpoint evaluation, 20,000 bootstrap replicates, and five positive-result conditions. The independent training seed—not the episode—is the statistical unit.

## 9. Results

| Method | Mean surplus | Meaning |
|---|---:|---|
| Classical exact | 0.00000 | Surplus reference |
| Quantum static | 0.22478 | Fixed allocation priorities |
| Quantum random | 0.22685 | Random feasible allocation |
| Quantum greedy | 0.30839 | Best visible immediate gain |
| **BER-MARL** | **0.50642** | Learned long-horizon allocation |
| Exact online DP | 0.57122 | Model-aware online reference |
| Future-clairvoyant | 0.58326 | Realized future is known |

BER-MARL exceeded greedy by:

\[
0.50642-0.30839=0.19803.
\]

The paired-seed 95% bootstrap interval was `[0.19141,0.20432]`. All 16 seeds were positive. Relative to greedy surplus, the improvement was:

\[
\frac{0.19803}{0.30839}=64.2\%.
\]

```mermaid
flowchart LR
    C[Classical<br/>0.000]
    R[Random/static<br/>about 0.225]
    G[Greedy<br/>0.308]
    B[BER-MARL<br/>0.506]
    D[Online DP<br/>0.571]
    F[Clairvoyant<br/>0.583]
    C --> R --> G --> B --> D --> F
```

BER-MARL recovered `88.66%` of online-DP surplus. The one-sided 95% lower bound was `87.68%`, below the preregistered requirement of more than 90%.

### Five-part result gate

| Condition | Outcome |
|---|---|
| Paired BER-MARL-minus-greedy lower bound above zero | Passed |
| Improvement at least 5% of DP surplus | Passed |
| BER-MARL/DP one-sided lower bound above 90% | **Failed** |
| BER-MARL surplus lower bound above zero | Passed |
| Exact budget and matching constraints | Passed |

Four of five conditions passed. The full composite gate failed, while the learned-versus-greedy result remained strongly positive.

The 12 held-out games had Bell-minus-classical gaps from `0` to `0.03830`, mean `0.01760`. Zero-gap types remained in the test. The independent audit re-enumerated every classical optimum, resolved every SDP ceiling, and reproduced the analysis from raw rows.

## 10. What is and is not established

The evidence supports:

> Under the ideal Bell-pair and zero-message information model, BER-MARL learned to route a consumable entanglement budget across an unseen dynamic graph and achieved substantially greater held-out surplus than random, static, and myopic allocation.

It does not establish a new Bell inequality, a new theorem that entanglement can beat local strategies, quantum computational speedup, hardware advantage, performance above the registered 90%-of-DP confidence target, or superiority when private observations can be freely communicated.

The request-level quantum gaps rely on known nonlocal-game principles. The proposed novelty is the **sequential allocation of scarce entanglement over a changing multi-agent graph**, together with the certified benchmark, learning method, and held-out evidence.

## 11. Why the 90% condition failed

DP knows the exact transition matrix of the held-out catalog. BER-MARL trains on different catalogs and receives no adaptation period on the final one. It learned the broad rule—save pairs for valuable opportunities—but not every catalog-specific timing detail. This explains why it strongly beat greedy while missing the near-oracle criterion.

A natural next study would estimate public transitions online or use meta-RL, while continuing to hide private endpoint contexts.

## 12. Limitations

1. Bell policies use ideal analytic correlations; confirmatory hardware noise and finite shots were not tested.
2. Random parity tables enable exact certification but are less application-specific than networking or sensing tasks.
3. The benchmark has four agents; larger graphs create a much larger matching action space.
4. PPO, recurrent policies, model-based RL, and explicit threshold policies were not locked baselines.
5. Zero-shot transition shift is demanding and caused the failed near-oracle condition.
6. A bounded novelty search found no identical formulation, but a broader human literature review is required before claiming priority.

## 13. Publication contribution

The defensible contributions are:

1. a sequential MARL formulation for routing consumable entanglement;
2. a matching-constrained benchmark with heterogeneous certified opportunities;
3. BER-MARL, a public-information masked DQN allocator;
4. exact classical enumeration and independent quantum SDP certification;
5. a held-out 64.2% improvement over greedy across all 16 seeds;
6. an informative negative result for the stricter 90%-of-DP condition;
7. complete raw results, checkpoints, configurations, failures, hashes, and audit artifacts.

This supports a preprint and workshop submission. A stronger full-conference version should add broader related work, noisy finite-shot evaluation, larger graphs, PPO and model-based baselines, and online adaptation.

## 14. Reproducibility

The prospective plan is `docs/BUDGETED_ENTANGLEMENT_MARL_PLAN.md`. Authoritative artifacts are:

- `experiments/21/21-budgeted-entanglement-task_attempt-03`;
- `experiments/22/22-budgeted-entanglement-development_attempt-02`;
- `experiments/23/23-budgeted-entanglement-confirmatory_attempt-01`;
- `experiments/24/24-budgeted-entanglement-release-audit_attempt-01`.

The confirmatory bundle contains 16 checkpoints, 16,000 learned-policy episode rows, baseline rows, test certificates, bootstrap results, resource counters, and hashes. A fresh clone validated 281 reportable bundles. The repository suite had 128 passing tests, and the final completion workflow passed.

## 15. Final intuitive example

Suppose two Bell pairs remain with three rounds left:

| Round | Best visible gain | Future possibility | Greedy | Long-horizon policy |
|---|---:|---|---|---|
| 8 | 0.03 | Urgent state is likely | Spend | Save |
| 9 | 0.02 | Urgent state remains possible | Spend | Save or spend one |
| 10 | 0.12 + 0.09 | No future remains | No pairs left | Spend both |

Greedy earns `0.05`. A controller that understands the horizon can earn `0.21`. Learning this timing decision—not rediscovering the already-known existence of Bell correlations—is the central problem.

## References and positioning

1. *eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels*, arXiv:2405.17486.
2. Standard XOR-game semidefinite formulations for binary-output quantum correlations.
3. Standard DQN and experience-replay methods for sequential control.
4. Entanglement-allocation and quantum-networking literature identified in the prospective novelty screen.

Before submission, this section should be replaced by a systematic human-reviewed bibliography covering nonlocal games, entanglement allocation, constrained MARL, dynamic matching, and quantum networking.
