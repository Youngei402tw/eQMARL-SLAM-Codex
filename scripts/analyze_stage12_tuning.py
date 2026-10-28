"""Apply the registered Stage 12 learning-rate and structural-width rules."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments/12"


def main():
    groups = defaultdict(list)
    records = []
    for folder in sorted(RUN_ROOT.glob("12-tuning-*")):
        config = json.loads((folder / "resolved_config.yaml").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        diagnostics = json.loads((folder / "diagnostics.json").read_text())
        values = [row["return"] for row in diagnostics["summary"]["validation"]]
        if manifest["status"] != "completed" or len(values) != 30:
            raise RuntimeError(f"Incomplete tuning outcome: {folder.name}")
        seed_mean = statistics.fmean(values)
        key = (config["method"], config["learning_rate"])
        groups[key].append({"seed": config["actor_seed"], "mean_validation_return": seed_mean})
        records.append({"run_id": folder.name, "method": key[0], "learning_rate": key[1],
                        "actor_seed": config["actor_seed"], "validation_episodes": len(values),
                        "mean_validation_return": seed_mean, "status": manifest["status"]})
    methods = sorted({key[0] for key in groups})
    selections, table = {}, []
    for method in methods:
        candidates = []
        for rate in (0.001, 0.003):
            rows = groups[(method, rate)]
            if len(rows) != 2:
                raise RuntimeError(f"Expected two tuning seeds for {method} at {rate}")
            score = statistics.fmean(row["mean_validation_return"] for row in rows)
            table.append({"method": method, "learning_rate": rate,
                          "seed_mean_validation_returns": rows, "selection_score": score})
            candidates.append((score, -rate, rate))
        selections[method] = max(candidates)[2]
    # The width is selected structurally, without outcome access: width 16 gives
    # 250 parameters, closest to the registered 249-parameter central control.
    width_grid = [{"hidden_width": 8, "parameter_count": 154},
                  {"hidden_width": 16, "parameter_count": 250},
                  {"hidden_width": 32, "parameter_count": 442}]
    output = {
        "analysis_version": "mechanism-tuning-selection-v1",
        "rule": "Highest mean greedy validation return across two trained seeds; exact ties choose lower learning rate.",
        "test_accessed": False, "all_planned_runs_retained": True,
        "run_records": records, "rate_table": table, "selected_learning_rates": selections,
        "flexible_width_selection": {"rule": "Closest parameter count to central_classical before pilot outcomes",
                                     "central_parameter_count": 249, "grid": width_grid,
                                     "selected_hidden_width": 16, "selected_parameter_count": 250},
    }
    destination = ROOT / "analysis/12/tuning-selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
