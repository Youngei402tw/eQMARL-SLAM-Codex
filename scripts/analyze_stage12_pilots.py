"""Descriptive pilot phase diagram; no model, condition, or contrast selection."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    grouped=defaultdict(list); records=[]
    for folder in sorted((ROOT/"experiments/12").glob("12-pilot-*")):
        config=json.loads((folder/"resolved_config.yaml").read_text())
        manifest=json.loads((folder/"manifest.json").read_text())
        diagnostics=json.loads((folder/"diagnostics.json").read_text())
        values=diagnostics["summary"]["validation"]
        if manifest["status"]!="completed" or len(values)!=30:
            raise RuntimeError(f"Incomplete pilot: {folder.name}")
        row={"run_id":folder.name,"condition":config["condition"]["name"],"method":config["method"],
             "actor_seed":config["actor_seed"],"mean_physical_coverage":statistics.fmean(x["physical_coverage"] for x in values),
             "mean_return":statistics.fmean(x["return"] for x in values),
             "wall_seconds":manifest["resources_consumed"]["wall_seconds"],
             "circuit_evaluations":manifest["resources_consumed"]["circuit_evaluations"]}
        grouped[(row["condition"],row["method"])].append(row); records.append(row)
    table=[]
    for (condition,method),rows in sorted(grouped.items()):
        table.append({"condition":condition,"method":method,"trained_seeds":3,
          "mean_validation_physical_coverage":statistics.fmean(x["mean_physical_coverage"] for x in rows),
          "mean_validation_return":statistics.fmean(x["mean_return"] for x in rows),
          "seed_rows":rows})
    def value(condition,method):
        return next(x["mean_validation_physical_coverage"] for x in table if x["condition"]==condition and x["method"]==method)
    effects={condition:value(condition,"eqmarl")-value(condition,"quantum_noent") for condition in ("additive","low","high")}
    output={"analysis_version":"mechanism-pilot-v1","scope":"descriptive development/validation only",
      "test_accessed":False,"all_45_planned_pilots_retained":len(records)==45,
      "table":table,"eqmarl_minus_quantum_noent_validation_coverage":effects,
      "pilot_high_minus_additive_interaction":effects["high"]-effects["additive"],
      "interpretation":"Variance/feasibility diagnostic only; no confirmatory contrast, method, condition, seed, or budget was selected from these outcomes."}
    (ROOT/"analysis/12/pilot-phase-diagram.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__": main()
