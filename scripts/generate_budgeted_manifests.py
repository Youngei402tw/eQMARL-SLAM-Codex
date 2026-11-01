"""Generate immutable split manifests without evaluating request values."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eqmarl_slam.budgeted_entanglement import generate_public_catalog, generate_request_types

SPECS = {
    "development": (20261101, 8, 20261201),
    "validation": (20261102, 8, 20261202),
    "test": (20261103, 12, 20261203),
}
for split, (type_seed, count, catalog_seed) in SPECS.items():
    types = generate_request_types(type_seed, count, split)
    type_path = ROOT / "configs/data" / f"{split}-budgeted-request-types-v1.json"
    type_path.write_text(json.dumps(types, indent=2) + "\n")
    catalog = generate_public_catalog(catalog_seed, split, [row["type_id"] for row in types["types"]])
    catalog_path = ROOT / "configs/data" / f"{split}-budgeted-public-catalog-v1.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
