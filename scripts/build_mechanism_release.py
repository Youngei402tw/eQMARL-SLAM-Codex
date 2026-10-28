"""Materialize paper-ready Stage 14 artifacts from the verified audit bundle."""
import json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experiments/14/14-mechanism-release-audit_attempt-01"
DEST=ROOT/"analysis/14"


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for source,target in (("reproduced_analysis.json","reproduced-analysis.json"),("paper_tables.json","paper-tables.json"),("claim_limitations.json","claim-limitations.json")):
        shutil.copyfile(SOURCE/source,DEST/target)
    analysis=json.loads((ROOT/"analysis/13/confirmatory-analysis.json").read_text())
    lines=["# Paper-ready mechanism-study tables","", "## Primary result","",
      "| Estimand | Estimate | 95% paired-seed bootstrap interval | Minimum effect | Outcome |",
      "|---|---:|---:|---:|---|",
      f"| `[eQMARL-noent]_high - [eQMARL-noent]_additive` | {analysis['primary']['estimate']:.5f} | [{analysis['primary']['interval_95'][0]:.5f}, {analysis['primary']['interval_95'][1]:.5f}] | 0.02000 | completed null |","",
      "## High-complementarity descriptive contrasts","",
      "| Contrast | Estimate | 95% unadjusted interval |","|---|---:|---:|"]
    for row in analysis["secondary_descriptive"]:
        lines.append(f"| {row['contrast']} | {row['estimate']:.5f} | [{row['interval_95_unadjusted'][0]:.5f}, {row['interval_95_unadjusted'][1]:.5f}] |")
    lines += ["","## Resource ledger","","| Resource | Total |","|---|---:|"]
    for key,value in analysis["resources"].items(): lines.append(f"| {key} | {value} |")
    lines += ["","Intervals use trained seeds as the replication unit. Secondary intervals are descriptive and unadjusted. Zero shots means analytic simulation; it does not mean zero quantum-circuit work.",""]
    (DEST/"paper-ready-tables.md").write_text("\n".join(lines))


if __name__=="__main__": main()
