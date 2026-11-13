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
        key = {"online": "online_reference", "greedy": "greedy"}.get(m, m)
        return mean[(key, c)]

    methods = ["random", "static", "greedy", "ppo", "dqn", "bid_price", "ce_dp_k0", "online"]
    fresh = ["qip-test-topology-a6-ring", "qip-test-scale-a8-sparse"]

    def overall_frac(m, subset):
        return float(np.mean([method_cat(m, c) for c in subset]) / np.mean([online[c] for c in subset]))

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
        "bpValMean": f4(sel30["bid_price"]["validation_mean"]), "rhoDev": f3(sel30["certainty_equivalent"]["rho_development"]),
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
    lines = ["% Generated by make_tmlr_assets.py from the Stage-30 bundle; do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (HERE / "generated" / "numbers30.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
