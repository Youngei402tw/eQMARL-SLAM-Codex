"""Small explicit execution interface; importing this module is classical-only."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--stage", required=True)
    preflight.add_argument("--profile", default="diagnostic")
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--source-receipt", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            from .execution import preflight_stage
            result = preflight_stage(args.stage, args.profile)
        elif args.command == "run":
            from .execution import execute
            result = execute(args.config, args.source_receipt)
        else:
            from .archive import validate_bundle
            result = validate_bundle(args.run)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("passed", result.get("valid", True)) else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
