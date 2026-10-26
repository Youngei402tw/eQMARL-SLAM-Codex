#!/usr/bin/env python3
"""Generate disjoint deterministic scenario manifests for the mechanism study."""

import argparse
import json
from pathlib import Path


SPECS = {
    "train": (64, 200000, 201000),
    "development": (2, 210000, 211000),
    "validation": (30, 220000, 221000),
    "test": (100, 330000, 331000),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split, (count, map_base, noise_base) in SPECS.items():
        manifest = {
            "split": split,
            "protocol": "belief-controller-mechanism-v1",
            "scenarios": [
                {
                    "scenario_id": f"mechanism-{split}-v1-{index:03d}",
                    "map_seed": map_base + index,
                    "noise_seed": noise_base + index,
                }
                for index in range(count)
            ],
            "purpose": split,
            "registration_note": (
                "Fresh namespace registered after the Stage08 final result. "
                "The test catalog is reserved for locked Stage13 actor-only evaluation."
            ),
        }
        path = args.output / f"{split}-mechanism-v1.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
