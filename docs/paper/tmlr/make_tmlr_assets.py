#!/usr/bin/env python3
"""Generate every Stage-30 number, figure-source CSV, and figure used by the TMLR manuscript.

Usage: python docs/paper/tmlr/make_tmlr_assets.py

Inputs (all hash-checked against their bundle inventories):
  experiments/30/30-tmlr-baselines_attempt-02/{baseline_analysis.json, episode_results.jsonl}
  experiments/28/28-qip-confirmatory_attempt-01/episode_results.jsonl
Outputs: generated/numbers30.tex, figure_data/*.csv (with figure_data/SHA256SUMS), figures/*.pdf.
Stage-28/29 numbers are reused from docs/paper/generated/numbers.tex (Stage-29 audit).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S30 = ROOT / "experiments/30/30-tmlr-baselines_attempt-02"
S28 = ROOT / "experiments/28/28-qip-confirmatory_attempt-01"
S31A = ROOT / "experiments/31/31-headroom-pilot_attempt-01"
S31C = ROOT / "experiments/31/31-regime-development_attempt-01"
S32 = ROOT / "experiments/32/32-learner-strength_attempt-01"
S33 = ROOT / "experiments/33/33-transition-robustness_attempt-01"
S34 = ROOT / "experiments/34/34-unified-confirmatory_attempt-01"
CATALOGS = ["qip-test-id-a4-complete", "qip-test-topology-a6-ring", "qip-test-scale-a8-sparse", "qip-test-horizon-a4-path"]
SHORT = {"qip-test-id-a4-complete": "4-agent\ncomplete$^\\dagger$", "qip-test-topology-a6-ring": "6-agent\nring",
         "qip-test-scale-a8-sparse": "8-agent\nsparse", "qip-test-horizon-a4-path": "4-agent path\n$H{=}24^\\dagger$"}
COLOR = {"random": "#c9c7c1", "static": "#9a9893", "greedy": "#1baf7a", "ppo": "#2a78d6", "dqn": "#eb6834",
         "bid_price": "#6250d6", "ce_dp_k0": "#e87ba4", "online": "#0b0b0b"}
NAME = {"random": "Random", "static": "Static", "greedy": "Greedy", "ppo": "PPO", "dqn": "DQN",
        "bid_price": "Bid-price (2 params)", "ce_dp_k0": "CE-DP, no data", "online": "Online DP (true $P$)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inventory(bundle, names):
    inventory = {row["path"]: row["sha256"] for row in json.loads((bundle / "artifact_manifest.json").read_text())["artifacts"]}
    for name in names:
        if sha(bundle / name) != inventory[name]:
            raise SystemExit(f"hash mismatch: {bundle.name}/{name}")


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True, "mathtext.fontset": "cm", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name):
    fig.savefig(HERE / "figures" / name, metadata={"CreationDate": None, "ModDate": None,
                                                    "Creator": "make_tmlr_assets.py", "Producer": None})
    plt.close(fig)


def write_csv(name, header, rows):
    with (HERE / "figure_data" / name).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n"); writer.writerow(header); writer.writerows(rows)


def main():
    check_inventory(S30, ["baseline_analysis.json", "episode_results.jsonl", "tuning_selection.json"])
    check_inventory(S28, ["episode_results.jsonl"])
    for folder in ("generated", "figures", "figure_data"):
        (HERE / folder).mkdir(exist_ok=True)
    analysis = json.loads((S30 / "baseline_analysis.json").read_text())
    manifest30 = json.loads((S30 / "manifest.json").read_text())
    freeze = json.loads((ROOT / "experiments/27/27-qip-robustness-scaling_attempt-01/stage28-freeze.json").read_text())
    seeds = freeze["training_seeds"]

    # Per-catalog sequence means: Stage-28 learned (per seed) and non-learned rows; Stage-30 comparators.
    learned = defaultdict(list); base = defaultdict(list)
    with (S28 / "episode_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "baseline_episode":
                base[(row["method"], row["catalog_id"])].append(row["surplus"])
            elif row.get("condition") == "ideal":
                learned[(row["method"], row["training_seed"], row["catalog_id"])].append(row["surplus"])
    with (S30 / "episode_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            base[(row["method"], row["catalog_id"])].append(row["surplus"])
    mean = {k: float(np.mean(v)) for k, v in base.items()}
    online = {c: mean[("online_reference", c)] for c in CATALOGS}
    assert all(abs(mean[("online_dp_recomputed", c)] - online[c]) < 1e-12 for c in CATALOGS)
    seed_cat = {(m, t, c): float(np.mean(learned[(m, t, c)])) for m in ("ppo", "dqn") for t in seeds for c in CATALOGS}

    def method_cat(m, c):
        if m in ("ppo", "dqn"):
            return float(np.mean([seed_cat[(m, t, c)] for t in seeds]))
        key = {"online": "online_reference", "greedy": "greedy", "future": "future_clairvoyant"}.get(m, m)
        return mean[(key, c)]

    methods = ["random", "static", "greedy", "ppo", "dqn", "bid_price", "ce_dp_k0", "online"]
    fresh = ["qip-test-topology-a6-ring", "qip-test-scale-a8-sparse"]

    def overall_frac(m, subset):
        return float(np.mean([method_cat(m, c) for c in subset]) / np.mean([online[c] for c in subset]))

    # ---- main results table: percentage of online DP first, mean surplus in parentheses ----------
    table_rows = [("Random", "random", False), ("Static", "static", False), ("Greedy", "greedy", False),
                  ("DQN", "dqn", False), ("PPO", "ppo", False), None,
                  ("Bid-price", "bid_price", True), ("CE-DP, $K{=}0$", "ce_dp_k0", False), None,
                  ("Online DP (true $P$)", "online", False), ("Future-clairvoyant", "future", False)]
    def cell(value, ref, bold):
        text = f"{100 * value / ref:.1f}"
        return (f"\\textbf{{{text}}}" if bold else text) + f"\\,({value:.3f})"
    tex = []
    for row in table_rows:
        if row is None:
            tex.append("\\midrule"); continue
        label, m, bold = row
        cells = [cell(method_cat(m, c), online[c], bold) for c in CATALOGS]
        overall = float(np.mean([method_cat(m, c) for c in CATALOGS]))
        cells.append(cell(overall, float(np.mean([online[c] for c in CATALOGS])), bold))
        tex.append(" & ".join([f"\\textbf{{{label}}}" if bold else label] + cells) + " \\\\")
    tex.append("\\bottomrule")  # kept inside the file: booktabs rules cannot follow \input in a tabular
    (HERE / "generated" / "main_table.tex").write_text("% Generated by make_tmlr_assets.py; do not edit.\n" + "\n".join(tex) + "\n")
    future_over_online = float(np.mean([method_cat("future", c) for c in CATALOGS]) / np.mean([online[c] for c in CATALOGS]))

    # ---- figure data ------------------------------------------------------------------------------
    rows = [[m, c, f"{method_cat(m, c):.10f}", f"{method_cat(m, c) / online[c]:.10f}"] for m in methods for c in CATALOGS]
    rows += [[m, "overall", f"{np.mean([method_cat(m, c) for c in CATALOGS]):.10f}", f"{overall_frac(m, CATALOGS):.10f}"] for m in methods]
    rows += [[m, "fresh", f"{np.mean([method_cat(m, c) for c in fresh]):.10f}", f"{overall_frac(m, fresh):.10f}"] for m in methods]
    write_csv("tmlr_fraction_online.csv", ["method", "catalog_id", "mean_surplus", "fraction_online_dp"], rows)
    seed_rows = []
    for m in ("ppo", "dqn"):
        for t in seeds:
            for scope, subset in (("all", CATALOGS), ("fresh", fresh)):
                s = float(np.mean([seed_cat[(m, t, c)] for c in subset]))
                seed_rows.append([m, t, scope, f"{s:.10f}"] + [f"{s - np.mean([method_cat(b, c) for c in subset]):.10f}"
                                                               for b in ("greedy", "bid_price", "ce_dp_k0")])
    write_csv("tmlr_seed_contrasts.csv", ["learner", "training_seed", "scope", "learner_mean", "minus_greedy", "minus_bid_price", "minus_ce_dp_k0"], seed_rows)
    contrast_rows = []
    for key, value in analysis["contrasts"].items():
        scope, pair = key.split(":"); learner, baseline = pair.split("-", 1)
        contrast_rows.append([scope, learner, baseline, f"{value['mean']:.10f}", f"{value['ci95'][0]:.10f}", f"{value['ci95'][1]:.10f}", value["seeds_above"]])
    write_csv("tmlr_contrasts.csv", ["scope", "learner", "baseline", "mean", "ci95_low", "ci95_high", "seeds_above"], contrast_rows)
    tuning = analysis["validation_tuning"]
    write_csv("tmlr_tuning_bid_price.csv", ["alpha", "prior_weight", "validation_mean"],
              [[r["alpha"], r["prior_weight"], f"{r['mean']:.10f}"] for r in tuning["bid_price"]])
    write_csv("tmlr_tuning_ce_dp.csv", ["warmup_episodes", "kappa", "validation_mean"],
              [[r["warmup_episodes"], "" if r["kappa"] is None else r["kappa"], f"{r['mean']:.10f}"] for r in tuning["certainty_equivalent"]])
    ce_rows = []
    for k in [int(x) for x in analysis["selection"]["certainty_equivalent"]["kappa_by_warmup"]]:
        mm = analysis["method_means"][f"ce_dp_k{k}"]
        ce_rows.append([k, f"{mm['all']:.10f}", f"{mm['fraction_online_all']:.10f}"])
    write_csv("tmlr_ce_dp_test.csv", ["warmup_episodes", "mean_surplus", "fraction_online_dp"], ce_rows)
    sums = "".join(f"{sha(p)}  {p.name}\n" for p in sorted((HERE / "figure_data").glob("*.csv")))
    (HERE / "figure_data" / "SHA256SUMS").write_text(sums)

    # ---- figures ----------------------------------------------------------------------------------
    setup()
    shown = ["greedy", "ppo", "dqn", "bid_price", "ce_dp_k0"]
    groups = CATALOGS + ["overall"]
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    width = 0.16
    for i, m in enumerate(shown):
        values = [method_cat(m, c) / online[c] for c in CATALOGS] + [overall_frac(m, CATALOGS)]
        x = np.arange(len(groups)) + (i - 2) * width
        ax.bar(x, values, width * 0.92, color=COLOR[m], label=NAME[m], zorder=2)
        if m in ("ppo", "dqn"):
            for j, c in enumerate(CATALOGS):
                pts = [seed_cat[(m, t, c)] / online[c] for t in seeds]
                ax.scatter(np.full(len(pts), x[j]) + np.linspace(-0.04, 0.04, len(pts)), pts, s=3, color=INK, zorder=3, linewidths=0)
    ax.axhline(1.0, color=INK, lw=0.9, zorder=1)
    ax.text(len(groups) - 0.45, 1.01, "online DP", fontsize=7, color=INK, ha="right", va="bottom")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([SHORT[c] for c in CATALOGS] + ["Overall\n(equal weight)"])
    ax.set_ylabel("Fraction of online-DP surplus"); ax.set_ylim(0, 1.12)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2), handlelength=1.2, columnspacing=1.0)
    save(fig, "fig_fraction_online.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15), sharey=True)
    labels = [("greedy", "Greedy"), ("bid_price", "Bid-price"), ("ce_dp_k0", "CE-DP, no data")]
    for ax, scope, title in zip(axes, ("all", "fresh"), ("All four sealed catalogs (registered)", "Two fully fresh catalogs")):
        for row_index, (b, label) in enumerate(labels):
            for offset, learner in ((0.13, "ppo"), (-0.13, "dqn")):
                c = analysis["contrasts"][f"{scope}:{learner}-{'greedy_recomputed' if b == 'greedy' else b}"]
                y = row_index + offset
                ax.plot(c["ci95"], [y, y], color=COLOR[learner], lw=2.2, solid_capstyle="butt")
                ax.plot([c["mean"]], [y], "o", ms=3.5, color="white", mec=COLOR[learner], mew=1.2)
                pts = [float(r[4 + [x for x, _ in labels].index(b)]) for r in seed_rows if r[0] == learner and r[2] == scope]
                ax.scatter(pts, np.full(len(pts), y - 0.07 * np.sign(offset) * -1), s=2.5, color=INK, linewidths=0, zorder=3)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels([f"Learner $-$ {label}" for _, label in labels])
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("Difference in mean surplus")
    axes[0].plot([], [], color=COLOR["ppo"], lw=2.2, label="PPO, 95% CI"); axes[0].plot([], [], color=COLOR["dqn"], lw=2.2, label="DQN, 95% CI")
    axes[0].legend(loc="lower left", fontsize=7)
    save(fig, "fig_contrasts.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    alphas = sorted({r["alpha"] for r in tuning["bid_price"]}); weights = sorted({r["prior_weight"] for r in tuning["bid_price"]})
    grid = np.array([[next(r["mean"] for r in tuning["bid_price"] if r["alpha"] == a and r["prior_weight"] == w) for a in alphas] for w in weights])
    image = axes[0].imshow(grid, cmap="Purples", aspect="auto", origin="lower")
    axes[0].grid(False)
    axes[0].set_xticks(range(len(alphas))); axes[0].set_xticklabels([f"{a:g}" for a in alphas])
    axes[0].set_yticks(range(len(weights))); axes[0].set_yticklabels([f"{w:g}" for w in weights])
    axes[0].set_xlabel(r"Pacing multiplier $\alpha$"); axes[0].set_ylabel("Prior weight $\\omega$")
    sel = analysis["selection"]["bid_price"]
    axes[0].add_patch(plt.Rectangle((alphas.index(sel["alpha"]) - 0.5, weights.index(sel["prior_weight"]) - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.05, pad=0.02); cbar.set_label("Validation mean surplus", fontsize=7.5); cbar.ax.tick_params(labelsize=6.5)
    axes[0].set_title("(a) Bid-price tuning on validation catalogs", fontsize=8.5)
    ks = [int(r[0]) for r in ce_rows]; xs = np.arange(len(ks))
    axes[1].plot(xs, [float(r[2]) for r in ce_rows], "o-", color=COLOR["ce_dp_k0"], ms=3.5, label="CE-DP (estimated $P$)")
    for m in ("bid_price", "dqn", "ppo", "greedy"):
        axes[1].axhline(overall_frac(m, CATALOGS), color=COLOR[m], lw=1.1, ls="--" if m != "bid_price" else "-", label=NAME[m])
    axes[1].set_xticks(xs); axes[1].set_xticklabels([str(k) for k in ks])
    axes[1].set_xlabel("Logged public sequences $K$ per test catalog"); axes[1].set_ylabel("Fraction of online-DP surplus")
    axes[1].set_ylim(0.45, 1.04); axes[1].legend(fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.4), ncol=2)
    axes[1].set_title("(b) Sealed catalogs: data needed by CE-DP", fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_tuning_ce.pdf")

    # ---- numbers ----------------------------------------------------------------------------------
    c = analysis["contrasts"]; mm = analysis["method_means"]; sel30 = analysis["selection"]
    def f3(x): return f"{x:.3f}"
    def f4(x): return f"{x:.4f}"
    def s4(x): return f"{x:+.4f}"
    def pct(x): return f"{100 * x:.1f}"
    ce_fracs = [float(r[2]) for r in ce_rows]
    macros = {
        "bpAlpha": f"{sel30['bid_price']['alpha']:g}", "bpPrior": f"{sel30['bid_price']['prior_weight']:g}",
        "bpValMean": f4(sel30["bid_price"]["validation_mean"]), "rhoDev": f3(sel30["certainty_equivalent"]["rho_development"]), "rhoFull": f3(rho_training()),
        "bpConfigs": str(len(tuning["bid_price"])), "ceConfigs": str(len(tuning["certainty_equivalent"])),
        "bpMean": f4(mm["bid_price"]["all"]), "bpFrac": pct(mm["bid_price"]["fraction_online_all"]),
        "bpFracFresh": pct(mm["bid_price"]["fraction_online_fresh"]),
        "ceZeroMean": f4(mm["ce_dp_k0"]["all"]), "ceZeroFrac": pct(mm["ce_dp_k0"]["fraction_online_all"]),
        "ceFracMin": pct(min(ce_fracs)), "ceFracMax": pct(max(ce_fracs)),
        "ppoFracFresh": pct(mm["ppo"]["fraction_online_fresh"]), "dqnFracFresh": pct(mm["dqn"]["fraction_online_fresh"]),
        "greedyFracFresh": pct(mm["greedy_recomputed"]["fraction_online_fresh"]),
        "gapClosed": f"{100 * (mm['bid_price']['all'] - mm['ppo']['all']) / (mm['online_dp_recomputed']['all'] - mm['ppo']['all']):.0f}",
        "reproErr": f"{analysis['evaluator_reproduction_max_abs_error']:.1g}",
        "decisionsThirty": f"{manifest30['resources_consumed']['decisions']:,}".replace(",", "{,}"),
        "wallThirty": f"{manifest30['resources_consumed']['wall_seconds']:.0f}",
        "cpuThirty": f"{manifest30['resources_consumed']['cpu_core_hours']:.3f}",
    }
    for scope, prefix in (("all", ""), ("fresh", "fr")):
        for learner in ("ppo", "dqn"):
            for b, tag in (("bid_price", "Bp"), ("ce_dp_k0", "CeZero"), ("greedy_recomputed", "Gr")):
                v = c[f"{scope}:{learner}-{b}"]; name = f"{prefix}{learner}Minus{tag}"
                macros[name] = s4(v["mean"]); macros[name + "Lo"] = s4(v["ci95"][0]); macros[name + "Hi"] = s4(v["ci95"][1])
                macros[name + "Wins"] = str(v["seeds_above"])
    for cat, tag in zip(CATALOGS, ("Cmp", "Ring", "Sparse", "Path")):
        for m, mtag in (("bid_price", "Bp"), ("ce_dp_k0", "CeZero")):
            macros[f"cat{tag}{mtag}"] = f3(method_cat(m, cat)); macros[f"frac{tag}{mtag}"] = pct(method_cat(m, cat) / online[cat])
    macros["foresightGain"] = f"{100 * (future_over_online - 1):.1f}"
    macros["onlineOfFuture"] = f"{100 / future_over_online:.1f}"
    macros.update(benchmark_assets())
    macros.update(stage31_assets())
    macros.update(stage32_assets())
    macros.update(stage33_assets())
    macros.update(stage34_assets())
    lines = ["% Generated by make_tmlr_assets.py from the Stage-30 to 34 bundles; do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (HERE / "generated" / "numbers30.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(macros, indent=1))


def rho_training():
    """Mean self-loop probability of the development + validation catalogs (the CE-DP prior used on the sealed catalogs)."""
    freeze = json.loads((ROOT / "experiments/27/27-qip-robustness-scaling_attempt-01/stage28-freeze.json").read_text())
    loops = []
    for ref in freeze["training_catalogs"]["development"] + freeze["training_catalogs"]["validation"]:
        if sha(ROOT / ref["path"]) != ref["sha256"]:
            raise SystemExit(f"hash mismatch: {ref['path']}")
        loops.append(np.mean(np.diag(np.asarray(json.loads((ROOT / ref["path"]).read_text())["transition"], float))))
    return float(np.mean(loops))


def _checked_json(ref):
    if sha(ROOT / ref["path"]) != ref["sha256"]:
        raise SystemExit(f"hash mismatch: {ref['path']}")
    return json.loads((ROOT / ref["path"]).read_text())


def benchmark_assets():
    """Benchmark description: XOR-game example, request-type statistics, catalog table, network figure."""
    freeze = json.loads((ROOT / "experiments/27/27-qip-robustness-scaling_attempt-01/stage28-freeze.json").read_text())
    certs = {r["type_id"]: r for r in _checked_json(freeze["non_test_certificate_bundle"])["certificates"]}
    lock = json.loads((ROOT / "configs/data/qip_v1/split-lock.json").read_text())
    types = {}
    for split in ("development", "validation"):
        for t in _checked_json(lock["splits"][split]["request_types"])["types"]:
            types[t["type_id"]] = t
    gaps = np.array([max(0.0, c["bell_value"] - c["classical_value"]) for c in certs.values()])
    positive = gaps[gaps > 1e-9]
    ex_id = "qip-development-type-000"
    ex, cert = types[ex_id], certs[ex_id]
    f, w = np.array(ex["predicate"]), np.array(ex["weights"])
    a, b = cert["classical_strategy"]["alice"], cert["classical_strategy"]["bob"]
    alpha, beta = cert["bell_angles"][:3], cert["bell_angles"][3:]
    total_w = int(w.sum())
    rows = ["\\begin{tabular}{@{}cc|ccc|cccc|c@{}}", "\\toprule",
            "\\multicolumn{2}{c|}{Questions} & Target & Weight & Probability & \\multicolumn{4}{c|}{Best classical strategy} & With a Bell pair \\\\",
            "$x$ & $y$ & $f(x,y)$ & $w_{xy}$ & $\\pi(x,y)$ & $a(x)$ & $b(y)$ & $a\\oplus b$ & wins? & win probability \\\\", "\\midrule"]
    for x in range(3):
        for y in range(3):
            won = (a[x] ^ b[y]) == f[x, y]
            q = 0.5 * (1 + (-1) ** int(f[x, y]) * np.cos(alpha[x] - beta[y]))
            rows.append(f"{x} & {y} & {f[x, y]} & {w[x, y]} & {w[x, y]}/{total_w} & {a[x]} & {b[y]} & {a[x] ^ b[y]} & "
                        + ("yes" if won else "\\textbf{no}") + f" & {q:.3f} \\\\")
    rows += ["\\midrule",
             f"\\multicolumn{{5}}{{l|}}{{Winning probability (weighted by $\\pi$)}} & \\multicolumn{{4}}{{c|}}{{$V_C={cert['classical_value']:.3f}$}} & $V_Q={cert['bell_value']:.3f}$ \\\\",
             "\\bottomrule", "\\end{tabular}"]
    worst = min(((x, y) for x in range(3) for y in range(3)), key=lambda xy: 0.5 * (1 + (-1) ** int(f[xy]) * np.cos(alpha[xy[0]] - beta[xy[1]])))
    qv = lambda xy: 0.5 * (1 + (-1) ** int(f[xy]) * np.cos(alpha[xy[0]] - beta[xy[1]]))
    lost = max(((x, y) for x in range(3) for y in range(3) if (a[x] ^ b[y]) != f[x, y]), key=lambda xy: w[xy])
    others = [qv((x, y)) for x in range(3) for y in range(3) if (x, y) not in (lost, worst)]
    xor_extra_more = {"xorRecX": str(lost[0]), "xorRecY": str(lost[1]), "xorRecW": str(int(w[lost])), "xorRecQ": f"{qv(lost):.3f}",
                      "xorOtherMin": f"{min(others):.3f}", "xorOtherMax": f"{max(others):.3f}"}
    xor_extra = {**xor_extra_more, "xorWorstX": str(worst[0]), "xorWorstY": str(worst[1]), "xorWorstW": str(int(w[worst])),
                 "xorWorstQ": f"{0.5 * (1 + (-1) ** int(f[worst]) * np.cos(alpha[worst[0]] - beta[worst[1]])):.3f}"}
    (HERE / "generated" / "xor_example.tex").write_text("% Generated by make_tmlr_assets.py; do not edit.\n" + "\n".join(rows) + "\n")
    # catalog table (all splits)
    refs = [("development", r) for r in freeze["training_catalogs"]["development"]] + \
           [("validation", r) for r in freeze["training_catalogs"]["validation"]] + [("test", r) for r in freeze["test_catalogs"]] + \
           [("confirm", r) for r in json.loads((ROOT / "configs/data/qip_v2/split-lock.json").read_text())["test"]["catalogs"]]
    cats, tex = [], []
    names = {"path": "path", "ring": "ring", "star": "star", "complete": "complete", "sparse_random": "sparse"}
    for split, ref in refs:
        c = _checked_json(ref); cats.append(c)
        P = np.asarray(c["transition"], float); n = len(P)
        tex.append(" & ".join([split.capitalize() if not tex or tex[-1].split(" & ")[0] != split.capitalize() else "",
                               names[c["topology"]] + ("$^\\dagger$" if c["catalog_id"] in ("qip-test-id-a4-complete", "qip-test-horizon-a4-path") else ""), str(c["agents"]), str(len(c["edges"])), str(c["horizon"]), str(c["bell_pair_budget"]),
                               f"{np.mean([len(s['requests']) for s in c['states']]):.1f}",
                               f"{np.mean(np.diag(P)):.2f}", f"{np.mean([P[i, (i + 1) % n] for i in range(n)]):.2f}"]) + " \\\\")
    out, prev = [], None
    for line, (split, _) in zip(tex, refs):
        if prev is not None and split != prev:
            out.append("\\midrule")
        label = {"development": "Dev.", "validation": "Val.", "test": "Earlier test", "confirm": "Test"}[split] if split != prev else ""
        out.append(label + line[line.index(" & "):]); prev = split
    head = ["\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
            "Split & Topology & Agents & Edges & $H$ & $B$ & Requests/state & Stay & Next \\\\", "\\midrule"]
    (HERE / "generated" / "catalog_table.tex").write_text("% Generated by make_tmlr_assets.py; do not edit.\n" + "\n".join(head + out + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    main_out, prev = [], None  # same table without the earlier test split (short version)
    for line, (split, _) in zip(tex, refs):
        if split == "test":
            continue
        if prev is not None and split != prev:
            main_out.append("\\midrule")
        label = {"development": "Dev.", "validation": "Val.", "confirm": "Test"}[split] if split != prev else ""
        main_out.append(label + line[line.index(" & "):]); prev = split
    (HERE / "generated" / "catalog_table_main.tex").write_text("% Generated by make_tmlr_assets.py; do not edit.\n" + "\n".join(head + main_out + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    training = cats[:12]
    stay = [np.mean(np.diag(np.asarray(c["transition"], float))) for c in training]
    nxt = [np.mean([np.asarray(c["transition"])[i][(i + 1) % 14] for i in range(14)]) for c in training]
    # network figure: five topologies, then three states of one catalog with their requests
    gap_of = lambda t: max(0.0, certs[t]["bell_value"] - certs[t]["classical_value"]) if t in certs else 0.0
    def layout(c):
        k = c["agents"]
        if c["topology"] == "path":
            return {i: (i / (k - 1), 0.5) for i in range(k)}
        if c["topology"] == "star":
            return {0: (0.5, 0.5), **{i: (0.5 + 0.45 * np.cos(2 * np.pi * (i - 1) / (k - 1)), 0.5 + 0.45 * np.sin(2 * np.pi * (i - 1) / (k - 1))) for i in range(1, k)}}
        return {i: (0.5 + 0.45 * np.cos(np.pi / 2 - 2 * np.pi * i / k), 0.5 + 0.45 * np.sin(np.pi / 2 - 2 * np.pi * i / k)) for i in range(k)}
    def draw(ax, c, state=None, title=""):
        pos = layout(c); ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 1.12); ax.set_aspect("equal"); ax.axis("off")
        active = {tuple(r["agents"]): (lambda g: g if g >= 5e-4 else 0.0)(r["urgency"] * gap_of(r["type_id"])) for r in (c["states"][state]["requests"] if state is not None else [])}
        for u, v in c["edges"]:
            g = active.get((u, v), active.get((v, u)))
            (x1, y1), (x2, y2) = pos[u], pos[v]
            if g is None:
                ax.plot([x1, x2], [y1, y2], color=GRID if state is not None else INK2, lw=0.9 if state is not None else 1.1, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], color=COLOR["dqn"] if g > 0 else INK2, lw=1.0 + 18 * g, solid_capstyle="round", zorder=1,
                        ls="-" if g > 0 else (0, (2, 1.5)))
                if g > 0:
                    through_centre = np.hypot((x1 + x2) / 2 - 0.5, (y1 + y2) / 2 - 0.5) < 0.1
                    frac = 0.64 if through_centre else 0.5
                    lx, ly = x1 + frac * (x2 - x1), y1 + frac * (y2 - y1)
                    ax.text(lx, ly, f"{g:.3f}", ha="center", va="center", fontsize=6.3, color=INK,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=3)
        for i, (x, y) in pos.items():
            ax.scatter([x], [y], s=55, color="white", edgecolor=INK, lw=0.8, zorder=2)
            ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=6, zorder=4)
        ax.set_title(title, fontsize=8, pad=2)
    pick = {c["catalog_id"]: c for c in cats}
    shown = [("qip-development-a6-path", "path, 6 agents"), ("qip-development-a4-ring", "ring, 4 agents"),
             ("qip-development-a6-star", "star, 6 agents"), ("qip-development-a4-complete", "complete, 4 agents"),
             ("qip-validation-a4-sparse", "sparse, 4 agents")]
    fig, axes = plt.subplots(2, 5, figsize=(6.6, 2.9))
    for ax, (cid, title) in zip(axes[0], shown):
        draw(ax, pick[cid], title=title)
    demo = pick["qip-development-a4-complete"]
    for k, ax in enumerate(axes[1][:4]):
        draw(ax, demo, state=k, title=f"state {k + 1}")
    axes[1][4].axis("off")
    axes[1][4].text(0.0, 0.5, "Orange: request with\npositive gain; label =\ncertified gain $G_e$.\nDashed: request with\nzero gain ($<10^{-3}$).\nGrey: no request.", fontsize=6.8, va="center", color=INK2)
    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    save(fig, "fig_network.pdf")
    return {**xor_extra, "xorClassical": f"{cert['classical_value']:.3f}", "xorBell": f"{cert['bell_value']:.3f}",
            "xorGap": f"{cert['bell_value'] - cert['classical_value']:.3f}", "xorWeightTotal": str(int(w.sum())),
            "xorLost": str(int(sum(w[x, y] for x in range(3) for y in range(3) if (a[x] ^ b[y]) != f[x, y]))),
            "nonTestTypes": str(len(certs)), "nonTestPositive": str(len(positive)),
            "gapMedian": f"{np.median(positive):.3f}", "gapMax": f"{positive.max():.3f}",
            "sdpMaxDev": f"{max(abs(c['sdp_gap']) for c in certs.values()):.0e}".replace("e-", "\\times 10^{-").replace("10^{-0", "10^{-") + "}",
            "genStayMin": f"{min(stay):.2f}", "genStayMax": f"{max(stay):.2f}", "genNextMin": f"{min(nxt):.2f}", "genNextMax": f"{max(nxt):.2f}"}


def stage31_assets():
    """Harder-variant pilot (31a) and hidden-regime development (31c): numbers, CSV, and figure."""
    check_inventory(S31A, ["headroom_analysis.json"]); check_inventory(S31C, ["development_analysis.json", "metrics.jsonl"])
    head = json.loads((S31A / "headroom_analysis.json").read_text())
    dev = json.loads((S31C / "development_analysis.json").read_text())
    refs = next(json.loads(l) for l in (S31C / "metrics.jsonl").read_text().splitlines() if '"kind": "references"' in l)
    belief = dev["belief_dp_validation_mean"]
    greedy = float(np.mean(list(refs["greedy"].values())))
    best = {}
    for key, value in sorted(dev["candidate_means"].items(), key=lambda kv: -kv[1]):
        best.setdefault(key.split(":")[0], (key.split(":")[1], value))
    trials = defaultdict(list)
    for t in dev["trials"]:
        trials[(t["method"], t["config"])].append(t["validation_mean"])
    names = {"ppo": "PPO", "recurrent_ppo": "Recurrent PPO", "dqn": "DQN"}
    rows = [["greedy", "", f"{greedy:.10f}", f"{greedy / belief:.10f}", ""]]
    for method in ("dqn", "ppo", "recurrent_ppo"):
        cfg, value = best[method]
        rows.append([method, cfg, f"{value:.10f}", f"{value / belief:.10f}", " ".join(f"{v:.10f}" for v in trials[(method, cfg)])])
    rows.append(["bid_price", f"alpha={dev['best_rule']['alpha']},omega={dev['best_rule']['prior_weight']}", f"{dev['best_rule']['mean']:.10f}", f"{dev['best_rule']['mean'] / belief:.10f}", ""])
    write_csv("tmlr_regime_validation.csv", ["method", "selected_config", "validation_mean", "fraction_belief_dp", "seed_means"], rows)
    sums = "".join(f"{sha(p)}  {p.name}\n" for p in sorted((HERE / "figure_data").glob("*.csv")))
    (HERE / "figure_data" / "SHA256SUMS").write_text(sums)
    fig, ax = plt.subplots(figsize=(4.6, 2.0))
    labels = ["Greedy", "DQN", "PPO", "Recurrent PPO", "Bid-price"]
    colors = [COLOR["greedy"], COLOR["dqn"], COLOR["ppo"], "#e87ba4", COLOR["bid_price"]]
    values = [float(r[3]) for r in rows]
    ax.barh(range(len(values)), values, color=colors, height=0.65, zorder=2)
    for i, r in enumerate(rows):
        if r[4]:
            pts = [float(x) / belief for x in r[4].split()]
            ax.scatter(pts, np.full(len(pts), i), s=6, color=INK, zorder=3, linewidths=0)
        right = max([values[i]] + ([float(x) / belief for x in r[4].split()] if r[4] else []))
        ax.text(right + 0.015, i, f"{100 * values[i]:.1f}%", va="center", fontsize=7)
    ax.axvline(1.0, color=INK, lw=0.9); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.15); ax.set_xlabel("Fraction of belief-state DP surplus (validation)")
    save(fig, "fig_regime.pdf")
    a, b = head["variants"]["A_generate_and_store"], head["variants"]["B_hidden_regime"]
    pct = lambda x: f"{100 * x:.1f}"
    m31 = json.loads((S31C / "manifest.json").read_text())["resources_consumed"]
    return {"hrStoreRule": pct(a["best_rule_fraction"]), "hrStoreGreedy": pct(a["greedy_fraction"]),
            "hrRegimeRule": pct(b["best_rule_fraction"]), "hrRegimeGreedy": pct(b["greedy_fraction"]),
            "hrRegimeOracle": pct(b["regime_oracle_over_belief"] - 1),
            "rgRuleFrac": pct(dev["rule_fraction_of_belief"]), "rgBestFrac": pct(dev["candidate_fraction_of_belief"]),
            "rgGreedyFrac": pct(greedy / belief), "rgBestName": names[dev["candidate"].split(":")[0]],
            "rgPPOFrac": pct(best["ppo"][1] / belief), "rgDQNFrac": pct(best["dqn"][1] / belief),
            "rgRPPOFrac": pct(best["recurrent_ppo"][1] / belief),
            "rgGap": f"{dev['best_rule']['mean'] - dev['candidate_validation_mean']:.4f}", "rgMargin": f"{0.05 * belief:.4f}",
            "rgModels": str(len(dev["trials"])), "rgWallHours": f"{m31['wall_seconds'] / 3600:.2f}", "rgCPU": f"{m31['cpu_core_hours']:.2f}"}


def stage32_assets():
    """Learner-strength check (Stage 32, v1 validation only): numbers and per-config CSV."""
    check_inventory(S32, ["strength_analysis.json", "metrics.jsonl"])
    d = json.loads((S32 / "strength_analysis.json").read_text())
    sel = {json.loads(l)["method"]: json.loads(l) for l in (S32 / "metrics.jsonl").read_text().splitlines() if '"kind": "selection"' in l}
    rows = []
    for method in ("ppo", "dqn"):
        for cfg in sel[method]["config_selection_means"]:
            seeds = [t for t in d["trials"] if t["method"] == method and t["config"] == cfg]
            rows.append([method, cfg, str(len(seeds)), f"{np.mean([t['selection_mean'] for t in seeds]):.10f}", f"{np.mean([t['comparison_mean'] for t in seeds]):.10f}",
                         "yes" if cfg == d["per_method"][method]["config"] else ""])
    rows.append(["bid_price", f"alpha={d['rule']['alpha']},omega={d['rule']['prior_weight']}", "", "", f"{d['rule']['comparison_mean']:.10f}", "yes"])
    tex = ["\\begin{tabular}{llrrr}", "\\toprule", "Method & Configuration & Seeds & Selection mean & Comparison: \\% of online DP (mean) \\\\", "\\midrule"]
    for r in rows:
        name = {"ppo": "PPO", "dqn": "DQN", "bid_price": "Bid-price"}[r[0]]
        cfg = r[1].replace("alpha=", "$\\alpha{=}").replace(",omega=", "$, $\\omega{=}") + ("$" if r[0] == "bid_price" else "")
        cell = lambda x: f"{float(x):.4f}" if x else "--"
        pct_cell = lambda x: f"{100 * float(x) / d['online_dp_mean']:.1f}\\,({float(x):.4f})"
        line = [name, cfg + (" (selected)" if r[5] and r[0] != "bid_price" else ""), r[2] or "--", cell(r[3]), pct_cell(r[4])]
        if r[0] == "dqn" and r[1] == rows[6][1] or r[0] == "bid_price":
            tex.append("\\midrule")
        tex.append(" & ".join(line) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (HERE / "generated" / "strength_table.tex").write_text("% Generated by make_tmlr_assets.py; do not edit.\n" + "\n".join(tex) + "\n")
    write_csv("tmlr_learner_strength.csv", ["method", "config", "seeds", "selection_mean", "comparison_mean", "selected"], rows)
    sums = "".join(f"{sha(p)}  {p.name}\n" for p in sorted((HERE / "figure_data").glob("*.csv")))
    (HERE / "figure_data" / "SHA256SUMS").write_text(sums)
    pct = lambda x: f"{100 * x:.1f}"
    ppo, dqn = d["per_method"]["ppo"], d["per_method"]["dqn"]
    ppo_trials = [t for t in d["trials"] if t["method"] == "ppo"]
    top = max(round(t["selection_mean"], 6) for t in ppo_trials)
    m32 = json.loads((S32 / "manifest.json").read_text())["resources_consumed"]
    curve = lambda m, i: f"{list(d['per_method'][m]['curve'].values())[i]:.3f}"
    episodes = lambda m: list(d["per_method"][m]["curve"])
    return {"lsDecision": d["decision"].replace("_", "\\_"), "lsRuleFrac": pct(d["rule"]["fraction_online_dp"]),
            "lsGreedyFrac": pct(d["greedy_fraction_online_dp"]),
            "lsPPOFrac": pct(ppo["fraction_online_dp"]), "lsPPOCmp": f"{ppo['comparison_mean']:.4f}", "lsDQNFrac": pct(dqn["fraction_online_dp"]),
            "lsPPOMinusRule": f"{ppo['minus_rule']:+.4f}", "lsPPOLo": f"{ppo['minus_rule_ci95'][0]:+.4f}", "lsPPOHi": f"{ppo['minus_rule_ci95'][1]:+.4f}",
            "lsDQNMinusRule": f"{dqn['minus_rule']:+.4f}", "lsDQNLo": f"{dqn['minus_rule_ci95'][0]:+.4f}", "lsDQNHi": f"{dqn['minus_rule_ci95'][1]:+.4f}",
            "lsPPOWins": str(ppo["seeds_above_rule"]), "lsDQNWins": str(dqn["seeds_above_rule"]), "lsSeeds": str(ppo["seeds"]),
            "lsPPOConfig": ppo["config"], "lsDQNConfig": dqn["config"],
            "lsPPOEpisodes": f"{int(episodes('ppo')[-1]):,}".replace(",", "{,}"), "lsDQNEpisodes": f"{int(episodes('dqn')[-1]):,}".replace(",", "{,}"),
            "lsPPOCurveA": curve("ppo", 0), "lsPPOCurveB": curve("ppo", 1), "lsPPOCurveC": curve("ppo", 2),
            "lsDQNCurveA": curve("dqn", 0), "lsDQNCurveB": curve("dqn", 1), "lsDQNCurveC": curve("dqn", 2),
            "lsRuleSel": f"{d['rule_tuning'] and max(r['selection_mean'] for r in d['rule_tuning']):.3f}",
            "lsPPOTrials": str(len(ppo_trials)), "lsPPOSameTop": str(sum(round(t["selection_mean"], 6) == top for t in ppo_trials)),
            "lsTrials": str(len(d["trials"])), "lsOnlineMean": f"{d['online_dp_mean']:.4f}", "lsWallHours": f"{m32['wall_seconds'] / 3600:.2f}", "lsCPU": f"{m32['cpu_core_hours']:.2f}"}


def stage33_assets():
    """Transition-matrix robustness (Stage 33, v1 validation only): numbers, CSV, and LaTeX table."""
    check_inventory(S33, ["robustness_analysis.json"])
    d = json.loads((S33 / "robustness_analysis.json").read_text())
    cols = ["greedy", "bid_price_frozen", "bid_price_retuned", "ce_dp_default", "ce_dp_matched_stay", "ce_dp_k10", "ce_dp_k100"]
    rows = [[r["variant"], f"{r['self_loop_mean']:.10f}"] + [f"{r['methods'][c]['fraction_online_dp']:.10f}" for c in cols] for r in d["table"]]
    write_csv("tmlr_transition_robustness.csv", ["variant", "mean_self_loop"] + cols, rows)
    sums = "".join(f"{sha(p)}  {p.name}\n" for p in sorted((HERE / "figure_data").glob("*.csv")))
    (HERE / "figure_data" / "SHA256SUMS").write_text(sums)
    pc = lambda x: f"{100 * x:.1f}"
    tex = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
           "& & & \\multicolumn{2}{c}{Bid-price} & \\multicolumn{3}{c}{CE-DP} \\\\ \\cmidrule(lr){4-5}\\cmidrule(l){6-8}",
           "Transition matrix & Stay & Greedy & frozen & retuned & default & $K{=}10$ & $K{=}100$ \\\\", "\\midrule"]
    for r in d["table"]:
        m = r["methods"]
        tex.append(" & ".join([r["variant"], f"{r['self_loop_mean']:.2f}"] + [pc(m[c]["fraction_online_dp"]) for c in ("greedy", "bid_price_frozen", "bid_price_retuned", "ce_dp_default", "ce_dp_k10", "ce_dp_k100")]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (HERE / "generated" / "transition_table.tex").write_text("% Generated by make_tmlr_assets.py; do not edit.\n" + "\n".join(tex) + "\n")
    by = {r["variant"]: r["methods"] for r in d["table"]}
    bp = [r["methods"]["bid_price_retuned"]["fraction_online_dp"] for r in d["table"]]
    ce_ok = [r["methods"]["ce_dp_default"]["fraction_online_dp"] for r in d["table"] if r["variant"] not in d["h2_failing_variants"]]
    m33 = json.loads((S33 / "manifest.json").read_text())["resources_consumed"]
    return {"trVariants": str(len(d["table"])), "trBpMin": pc(min(bp)), "trBpMax": pc(max(bp)),
            "trCeOkMin": pc(min(ce_ok)), "trCeSucc": pc(by["successor 0.8"]["ce_dp_default"]["fraction_online_dp"]),
            "trCeSuccKten": pc(by["successor 0.8"]["ce_dp_k10"]["fraction_online_dp"]),
            "trCeSuccKhundred": pc(by["successor 0.8"]["ce_dp_k100"]["fraction_online_dp"]),
            "trCeStayNine": pc(by["stay 0.9"]["ce_dp_default"]["fraction_online_dp"]),
            "trCeStayNineMatched": pc(by["stay 0.9"]["ce_dp_matched_stay"]["fraction_online_dp"]),
            "trGreedyMin": pc(min(r["methods"]["greedy"]["fraction_online_dp"] for r in d["table"])),
            "trGreedyMax": pc(max(r["methods"]["greedy"]["fraction_online_dp"] for r in d["table"])),
            "trWallSeconds": f"{m33['wall_seconds']:.0f}",
            "trMatchMaxDiff": f"{100 * max(abs(r['methods']['ce_dp_matched_stay']['fraction_online_dp'] - r['methods']['ce_dp_default']['fraction_online_dp']) for r in d['table']):.1f}"}


CF_CATALOGS = ["qip2-test-a4-complete", "qip2-test-a5-ring", "qip2-test-a8-sparse", "qip2-test-a6-path-h24"]
CF_SHORT = ["4-agent\ncomplete", "5-agent\nring", "8-agent\nsparse", "6-agent path\n$H{=}24$"]


def stage34_assets():
    """Confirmatory evaluation (Stage 34): every method fixed before a fresh sealed split was opened once."""
    check_inventory(S34, ["confirmatory_analysis.json", "episode_results.jsonl", "tuning_selection.json", "test_certificates.json"])
    a = json.loads((S34 / "confirmatory_analysis.json").read_text())
    m34 = json.loads((S34 / "manifest.json").read_text())["resources_consumed"]
    rule = defaultdict(list); learned = defaultdict(list)
    with (S34 / "episode_results.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["kind"] == "rule_episode":
                rule[(r["method"], r["catalog_id"])].append(r["surplus"])
            else:
                learned[(r["method"], r["training_seed"], r["catalog_id"])].append(r["surplus"])
    seeds = {m: sorted({t for (mm, t, _) in learned if mm == m}) for m in ("ppo", "dqn")}
    seed_cat = {k: float(np.mean(v)) for k, v in learned.items()}
    def cat_mean(m, c):
        if m in seeds:
            return float(np.mean([seed_cat[(m, t, c)] for t in seeds[m]]))
        return float(np.mean(rule[(m, c)]))
    online = {c: cat_mean("online_dp", c) for c in CF_CATALOGS}
    online_all = float(np.mean(list(online.values())))
    assert abs(online_all - a["online_dp_mean"]) < 1e-9
    for m in ("greedy", "bid_price", "ce_dp_k0", "ppo", "dqn", "future_clairvoyant"):  # re-derive the archived fractions
        assert abs(np.mean([cat_mean(m, c) for c in CF_CATALOGS]) / online_all - a["fraction_online_dp"][m]) < 1e-9, m
    rows = [("Greedy", "greedy", False), ("DQN", "dqn", False), ("PPO", "ppo", False), None,
            ("Bid-price", "bid_price", True), ("CE-DP, no data", "ce_dp_k0", False), None,
            ("Online DP (true $P$)", "online_dp", False), ("Future-clairvoyant", "future_clairvoyant", False)]
    def cell(v, ref, bold):
        t = f"{100 * v / ref:.1f}"
        return (f"\\textbf{{{t}}}" if bold else t) + f"\\,({v:.3f})"
    tex = []
    for row in rows:
        if row is None:
            tex.append("\\midrule"); continue
        label, m, bold = row
        cells = [cell(cat_mean(m, c), online[c], bold) for c in CF_CATALOGS]
        cells.append(cell(float(np.mean([cat_mean(m, c) for c in CF_CATALOGS])), online_all, bold))
        tex.append(" & ".join([f"\\textbf{{{label}}}" if bold else label] + cells) + " \\\\")
    tex.append("\\bottomrule")
    (HERE / "generated" / "confirm_table.tex").write_text("% Generated by make_tmlr_assets.py; do not edit.\n" + "\n".join(tex) + "\n")
    csv_rows = [[m, c, f"{cat_mean(m, c):.10f}", f"{cat_mean(m, c) / online[c]:.10f}"]
                for m in ("greedy", "dqn", "ppo", "bid_price", "ce_dp_k0", "online_dp", "future_clairvoyant") for c in CF_CATALOGS]
    csv_rows += [[m, t, c, f"{seed_cat[(m, t, c)]:.10f}", ""] for m in seeds for t in seeds[m] for c in CF_CATALOGS]
    write_csv("tmlr_confirmatory.csv", ["method", "catalog_or_seed", "catalog_or_mean", "mean_surplus_or_seed_mean", "fraction_online_dp"], csv_rows)
    sums = "".join(f"{sha(p)}  {p.name}\n" for p in sorted((HERE / "figure_data").glob("*.csv")))
    (HERE / "figure_data" / "SHA256SUMS").write_text(sums)

    setup()
    shown = ["greedy", "dqn", "ppo", "bid_price", "ce_dp_k0"]
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    width = 0.16
    for i, m in enumerate(shown):
        vals = [cat_mean(m, c) / online[c] for c in CF_CATALOGS] + [a["fraction_online_dp"][m]]
        x = np.arange(len(CF_CATALOGS) + 1) + (i - 2) * width
        ax.bar(x, vals, width * 0.92, color=COLOR[m], label=NAME[m], zorder=2)
        if m in seeds:
            for j, c in enumerate(CF_CATALOGS):
                pts = [seed_cat[(m, t, c)] / online[c] for t in seeds[m]]
                ax.scatter(np.full(len(pts), x[j]) + np.linspace(-0.04, 0.04, len(pts)), pts, s=3, color=INK, zorder=3, linewidths=0)
    ax.axhline(1.0, color=INK, lw=0.9, zorder=1)
    ax.text(len(CF_CATALOGS) + 0.55, 1.01, "online DP", fontsize=7, color=INK, ha="right", va="bottom")
    ax.set_xticks(np.arange(len(CF_CATALOGS) + 1)); ax.set_xticklabels(CF_SHORT + ["Overall\n(equal weight)"])
    ax.set_ylabel("Fraction of online-DP surplus"); ax.set_ylim(0, 1.12)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2), handlelength=1.2, columnspacing=1.0)
    save(fig, "fig_confirm.pdf")

    c = a["contrasts"]; f = a["fraction_online_dp"]; e = a["ppo_minus_dqn"]
    pc = lambda x: f"{100 * x:.1f}"; s3 = lambda x: f"{x:+.3f}"
    out = {"cfDecision": a["decision_primary"].replace("_", " "), "cfDqnDecision": a["dqn_vs_bid_price"].replace("_", " "),
           "cfAlpha": f"{a['bid_price_selection']['alpha']:g}", "cfOmega": f"{a['bid_price_selection']['prior_weight']:g}",
           "cfValMean": f"{a['bid_price_selection']['validation_mean']:.3f}", "cfRho": f"{a['rho']:.3f}",
           "cfOnlineMean": f"{online_all:.3f}", "cfForesight": f"{100 * (f['future_clairvoyant'] - 1):.1f}",
           "cfOnlineOfFuture": f"{100 / f['future_clairvoyant']:.1f}",
           "cfPpoMinusDqn": s3(e["mean"]), "cfPpoMinusDqnLoNinety": s3(e["ci90"][0]), "cfPpoMinusDqnHiNinety": s3(e["ci90"][1]),
           "cfEqMargin": f"{e['margin']:.3f}",
           "cfWallHours": f"{m34['wall_seconds'] / 3600:.1f}", "cfCPU": f"{m34['cpu_core_hours']:.1f}",
           "cfSdpMax": f"{max(abs(r['sdp_gap']) for r in json.loads((S34 / 'test_certificates.json').read_text())['certificates']):.1g}"}
    for m, tag in (("greedy", "Greedy"), ("dqn", "Dqn"), ("ppo", "Ppo"), ("bid_price", "Bp"), ("ce_dp_k0", "Ce")):
        out[f"cfFrac{tag}"] = pc(f[m])
        out[f"cfFracMin{tag}"] = pc(min(cat_mean(m, cc) / online[cc] for cc in CF_CATALOGS))
        out[f"cfFracMax{tag}"] = pc(max(cat_mean(m, cc) / online[cc] for cc in CF_CATALOGS))
    for key, tag in (("ppo_minus_bid_price", "PpoMinusBp"), ("ppo_minus_greedy", "PpoMinusGreedy"), ("dqn_minus_bid_price", "DqnMinusBp"),
                     ("dqn_minus_greedy", "DqnMinusGreedy"), ("ppo_minus_ce_dp_k0", "PpoMinusCe")):
        out[f"cf{tag}"] = s3(c[key]["mean"]); out[f"cf{tag}Lo"] = s3(c[key]["ci95"][0]); out[f"cf{tag}Hi"] = s3(c[key]["ci95"][1])
        out[f"cf{tag}Wins"] = str(c[key]["seeds_above"])
    ppo_scores = list(a["seed_scores"]["ppo"].values())
    out["cfPpoSame"] = str(sum(abs(v - max(ppo_scores)) < 1e-9 for v in ppo_scores))
    out["cfPpoBestSeedFrac"] = pc(max(ppo_scores) / online_all)
    out["cfGapClosed"] = f"{100 * (f['bid_price'] - f['ppo']) / (1 - f['ppo']):.0f}"
    return out


if __name__ == "__main__":
    main()
