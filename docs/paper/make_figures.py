#!/usr/bin/env python3
"""Build every data figure of the QIP manuscript from the Stage 29 audit's figure-source CSVs.

Usage: python docs/paper/make_figures.py [--data DIR] [--out DIR]

The CSVs are copied verbatim into docs/paper/figure_data/ (with their SHA-256 checked against the
audit's paper_tables.json) so that every plotted number has a committed source file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / "experiments/29/29-qip-release-audit_attempt-01"

# Fixed entity colours (validated reference categorical palette, light mode); baselines are neutral.
COLOR = {"ppo": "#2a78d6", "dqn": "#eb6834", "greedy": "#1baf7a", "static": "#9a9893", "random": "#c9c7c1",
         "online_reference": "#0b0b0b", "future_clairvoyant": "#52514e", "double_dqn": "#6250d6", "recurrent_ppo": "#e87ba4"}
NAME = {"ppo": "PPO", "dqn": "DQN", "greedy": "Greedy", "static": "Static", "random": "Random",
        "online_reference": "Online model-aware DP", "future_clairvoyant": "Future-clairvoyant",
        "double_dqn": "Double DQN", "recurrent_ppo": "Recurrent PPO"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"


def setup_fonts():
    # Computer Modern (bundled with matplotlib) matches the manuscript's Latin Modern text face.
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True, "mathtext.fontset": "cm", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None, "Creator": "make_figures.py", "Producer": None})


def read(data: Path, name: str):
    with (data / name).open() as stream:
        return list(csv.DictReader(stream))


def fig_main(data, out):
    rows = read(data, "fig_main_by_catalog.csv")
    catalogs = list(dict.fromkeys(r["catalog_id"] for r in rows))
    labels = {r["catalog_id"]: r["label"] for r in rows}
    value = {(r["catalog_id"], r["method"]): float(r["mean_surplus"]) for r in rows}
    sd = {(r["catalog_id"], r["method"]): float(r["seed_sd"]) for r in rows}
    seeds = read(data, "fig_seed_catalog.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.45), sharey=False)
    bars = ["random", "static", "greedy", "dqn", "ppo"]
    for ax, c in zip(axes, catalogs):
        xs = range(len(bars))
        for x, m in zip(xs, bars):
            ax.bar(x, value[(c, m)], width=0.72, color=COLOR[m], edgecolor="white", linewidth=1.0, zorder=2)
            if m in ("dqn", "ppo"):
                pts = [float(r["mean_surplus"]) for r in seeds if r["catalog_id"] == c and r["method"] == m]
                ax.scatter([x + (i - 5.5) * 0.035 for i in range(len(pts))], pts, s=5, color=INK, linewidths=0, zorder=3)
        for m, style in (("online_reference", "-"), ("future_clairvoyant", ":")):
            ax.axhline(value[(c, m)], color=COLOR[m], linestyle=style, linewidth=1.1, zorder=4)
        ax.set_xticks(list(xs), [NAME[m] for m in bars], rotation=55, ha="right")
        ax.set_title(labels[c], fontsize=8)
        ax.set_ylim(0, value[(c, "future_clairvoyant")] * 1.12)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean surplus over classical optimum")
    handles = [plt.Line2D([], [], color=COLOR["online_reference"], lw=1.1, label=NAME["online_reference"]),
               plt.Line2D([], [], color=COLOR["future_clairvoyant"], lw=1.1, ls=":", label=NAME["future_clairvoyant"] + " (unattainable)"),
               plt.Line2D([], [], marker="o", color=INK, lw=0, ms=2.5, label="Individual training seeds (n=12)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    save(fig, out / "fig_main_by_catalog.pdf"); plt.close(fig)


def fig_intervals(data, out):
    rows = {r["test"]: r for r in read(data, "fig_primary_intervals.csv")}
    seeds = read(data, "fig_seed_contrasts.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.35, 1]})
    eq = rows["ppo_minus_dqn_equivalence"]; sup = rows["ppo_minus_dqn_superiority"]
    margin = float(eq["threshold_high"])
    a.axvspan(-margin, margin, color="#e8f1fb", zorder=0, label=r"Equivalence region $\pm 5\%$ of online DP")
    a.axvline(0, color=INK2, lw=0.6)
    for y, row, label in ((1.0, sup, "95% interval (superiority test)"), (0.0, eq, "90% interval (equivalence test)")):
        lo, hi, est = float(row["lower"]), float(row["upper"]), float(row["estimate"])
        a.plot([lo, hi], [y, y], color=COLOR["ppo"], lw=2.2, solid_capstyle="round")
        a.plot(est, y, "o", color="white", mec=COLOR["ppo"], mew=1.6, ms=6)
        a.text(hi + 0.002, y + 0.18, f"[{lo:+.4f}, {hi:+.4f}]", fontsize=7, color=INK2, va="bottom")
    diffs = [float(r["ppo_minus_dqn"]) for r in seeds]
    a.scatter(diffs, [-0.85] * len(diffs), s=9, color=INK, alpha=0.75, linewidths=0)
    a.set_yticks([1.0, 0.0, -0.85], ["95% CI", "90% CI", "Seeds"])
    a.set_ylim(-1.3, 1.6); a.set_xlim(-0.085, 0.085)
    a.set_xlabel("PPO $-$ DQN mean surplus"); a.grid(axis="y", visible=False)
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7)
    g = rows["ppo_minus_greedy_superiority"]
    lo, hi, est = float(g["lower"]), float(g["upper"]), float(g["estimate"])
    b.axvline(0, color=INK2, lw=0.6)
    b.plot([lo, hi], [1, 1], color=COLOR["greedy"], lw=2.2)
    b.plot(est, 1, "o", color="white", mec=COLOR["greedy"], mew=1.6, ms=6)
    b.text(est, 1.25, f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]", fontsize=7, color=INK2, ha="center")
    gd = [float(r["ppo_minus_greedy"]) for r in seeds]
    b.scatter(gd, [0.15] * len(gd), s=9, color=INK, alpha=0.75, linewidths=0)
    b.set_yticks([1, 0.15], ["95% CI", "Seeds"]); b.set_ylim(-0.4, 1.7); b.set_xlim(-0.03, 0.33)
    b.set_xlabel("PPO $-$ greedy mean surplus"); b.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, out / "fig_primary_intervals.pdf"); plt.close(fig)


def fig_noise(data, out):
    s27 = read(data, "fig_noise_stage27.csv"); s28 = read(data, "fig_noise_stage28.csv")
    thresholds = read(data, "fig_visibility_thresholds.csv"); positive = read(data, "fig_positive_fraction_test.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.55))
    ideal27 = {r["method"]: float(r["mean_surplus"]) for r in s27 if r["condition"] == "ideal"}
    ideal28 = next(float(r["ppo_mean_surplus"]) for r in s28 if r["condition"] == "ideal")
    for ax, prefix, key in ((a, "heralded", "heralded_success"), (b, "visibility", "visibility")):
        for m in ("dqn", "double_dqn", "ppo", "recurrent_ppo"):
            pts = [(1.0, 1.0)] + [(float(r["condition"].split("-")[1]), float(r["mean_surplus"]) / ideal27[m]) for r in s27
                                  if r["method"] == m and r["condition"].startswith(prefix)]
            pts.sort(); ax.plot(*zip(*pts), marker="o", ms=3.2, lw=1.1, color=COLOR[m], label=f"{NAME[m]}, Stage 27 validation")
        pts = sorted([(float(r[key]), float(r["ppo_mean_surplus"]) / ideal28) for r in s28
                      if r["condition"] == "ideal" or r["condition"].startswith(prefix)])
        ax.plot(*zip(*pts), marker="s", ms=5.5, lw=0, mfc="none", mec=INK, mew=1.2, label="PPO, Stage 28 sealed test")
    a.plot([0.45, 1.02], [0.45, 1.02], color=INK2, lw=0.6, ls="--", label="Proportional scaling $p_h$")
    a.set_xlabel("Heralded generation success $p_h$"); a.set_ylabel("Surplus / ideal surplus")
    a.set_xlim(0.45, 1.02); a.set_ylim(0.45, 1.02); a.set_title("(a) Heralded loss", fontsize=8.5)
    b.set_xlabel("Werner visibility $v$"); b.set_title("(b) Werner noise", fontsize=8.5)
    b.set_xlim(0.57, 1.02); b.set_ylim(-0.04, 1.04)
    grid = [0.80 + 0.0025 * i for i in range(81)]
    for split, color, style, label in (("test", INK, "-", "Sealed test types (16)"), ("nontest", COLOR["static"], "-", "Non-test types (30)")):
        vals = [float(r["visibility_threshold"]) for r in thresholds if (r["split"] == "test") == (split == "test")]
        c.plot(grid, [sum(t < v - 1e-12 for t in vals) / len(vals) for v in grid], color=color, lw=1.3, ls=style, drawstyle="steps-post", label=label)
    c.plot([float(r["visibility"]) for r in positive], [float(r["fraction_of_test_opportunities_with_positive_gain"]) for r in positive],
           color=COLOR["ppo"], lw=1.1, ls="--", drawstyle="steps-post", label="Sealed test opportunities (255)")
    c.axvline(0.8, color=INK2, lw=0.6, ls=":")
    c.set_xlabel("Werner visibility $v$"); c.set_ylabel("Share with positive quantum gain")
    c.set_xlim(0.8, 1.0); c.set_ylim(0, 1.0); c.legend(fontsize=6.2, loc="upper left")
    c.set_title("(c) Gain retained by requests", fontsize=8.5)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.36, -0.13), fontsize=6.6)
    fig.tight_layout(w_pad=1.2)
    save(fig, out / "fig_noise.pdf"); plt.close(fig)


def fig_ablation(data, out):
    rows = read(data, "fig_ablation_stage26.csv"); methods = read(data, "fig_stage26_methods.csv")
    names = {"full": "Full public resource state", "no_budget": "No remaining budget", "no_horizon": "No remaining time",
             "binary_gain": "Binary gain indicators", "shuffled_gain": "Gains on wrong edges"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [1.15, 1]})
    order = ["full", "no_budget", "binary_gain", "shuffled_gain", "no_horizon"]
    vals = {r["condition"]: float(r["mean_validation_surplus"]) for r in rows}
    ys = list(range(len(order)))[::-1]
    for y, k in zip(ys, order):
        a.barh(y, vals[k], color=COLOR["double_dqn"] if k == "full" else "#b9b2ee", height=0.62, edgecolor="white")
        a.text(vals[k] + 0.004, y, f"{vals[k]:.3f}", va="center", fontsize=7, color=INK2)
    a.set_yticks(ys, [names[k] for k in order]); a.set_xlim(0, 0.37); a.grid(axis="y", visible=False)
    a.set_xlabel("Mean validation surplus (Double DQN, 2 seeds)"); a.set_title("(a) Input ablations", fontsize=8.5)
    mo = ["dqn", "double_dqn", "ppo", "recurrent_ppo"]; mv = {r["method"]: float(r["mean_validation_surplus"]) for r in methods}
    ys = list(range(len(mo)))[::-1]
    for y, m in zip(ys, mo):
        b.barh(y, mv[m], color=COLOR[m], height=0.62, edgecolor="white")
        b.text(mv[m] + 0.004, y, f"{mv[m]:.4f}", va="center", fontsize=7, color=INK2)
    b.set_yticks(ys, [NAME[m] for m in mo]); b.set_xlim(0, 0.37); b.grid(axis="y", visible=False)
    b.set_xlabel("Mean validation surplus (2 seeds)"); b.set_title("(b) Learner comparison", fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, out / "fig_ablation.pdf"); plt.close(fig)


def fig_scaling(data, out):
    rows = read(data, "fig_scaling.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    offsets = {"greedy": -0.12, "dqn": 0.0, "ppo": 0.12}
    for m in ("greedy", "dqn", "ppo"):
        for split, marker, fill in (("stage27-validation", "o", "white"), ("stage28-test", "s", None)):
            pts = [(int(r["agents"]) + offsets[m], float(r[m]) / float(r["online_reference"])) for r in rows if r["split"] == split]
            ax.scatter(*zip(*pts), marker=marker, s=22, facecolors=fill or COLOR[m], edgecolors=COLOR[m], linewidths=1.1, zorder=3)
    for m in ("greedy", "dqn", "ppo"):
        ax.scatter([], [], marker="s", s=22, color=COLOR[m], label=NAME[m])
    ax.scatter([], [], marker="o", s=22, facecolors="white", edgecolors=INK2, label="Validation catalog (Stage 27)")
    ax.scatter([], [], marker="s", s=22, color=INK2, label="Sealed test catalog (Stage 28)")
    ax.set_xticks([3, 4, 6, 8]); ax.set_xlabel("Number of agents"); ax.set_ylabel("Fraction of online-DP surplus")
    ax.set_ylim(0.3, 1.0); ax.legend(fontsize=6.3, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.22), columnspacing=0.8)
    fig.tight_layout()
    save(fig, out / "fig_scaling.pdf"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=AUDIT / "figure_data")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    setup_fonts()
    args.out.mkdir(parents=True, exist_ok=True)
    local = HERE / "figure_data"
    if args.data.resolve() != local.resolve():
        local.mkdir(exist_ok=True)
        tables = json.loads((args.data.parent / "paper_tables.json").read_text())
        for path in sorted(args.data.glob("*.csv")):
            expected = tables["figure_data_sha256"].get(path.name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected != digest:
                raise SystemExit(f"figure source hash mismatch: {path.name}")
            shutil.copyfile(path, local / path.name)
    for build in (fig_main, fig_intervals, fig_noise, fig_ablation, fig_scaling):
        build(local, args.out)
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
