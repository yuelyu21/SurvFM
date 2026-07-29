"""Command-line diagnostics for the private SurvFM-RMST preview."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform

from . import __version__
from .backbones import list_backbones


def _backbones(as_json: bool) -> int:
    rows = [item.to_dict() for item in list_backbones()]
    if as_json:
        print(json.dumps(rows, indent=2))
        return 0
    print("NAME               ACCESS       INSTALL EXTRA  DESCRIPTION")
    for row in rows:
        print(
            f"{row['name']:<18} {row['access']:<12} {row['install_extra']:<14} "
            f"{row['description']}"
        )
    return 0


def _diagnose(as_json: bool) -> int:
    packages = ["numpy", "pandas", "sklearn", "tabpfn", "tabicl", "tabdpt", "autogluon", "requests"]
    payload = {
        "survfm_rmst": __version__,
        "python": platform.python_version(),
        "packages_available": {
            package: importlib.util.find_spec(package) is not None for package in packages
        },
        "credentials_configured": {
            "TABH2O_API_KEY": bool(os.environ.get("TABH2O_API_KEY", "").strip())
        },
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"survfm-rmst {payload['survfm_rmst']} | Python {payload['python']}")
        for package, available in payload["packages_available"].items():
            print(f"{package:<14} {'available' if available else 'not installed'}")
        configured = payload["credentials_configured"]["TABH2O_API_KEY"]
        print(f"TABH2O_API_KEY {'configured' if configured else 'not configured'} (value hidden)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="survfm-rmst")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("backbones", help="List registered backbone choices.")
    subparsers.add_parser("diagnose", help="Inspect optional dependencies without exposing credentials.")
    args = parser.parse_args(argv)
    if args.command == "backbones":
        return _backbones(args.json)
    return _diagnose(args.json)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
