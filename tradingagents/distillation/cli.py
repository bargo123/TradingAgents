"""Standalone, machine-readable Phase 11A command line interface."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .writer import validate_generation

def _parser():
    p = argparse.ArgumentParser(prog="knowledge-distill")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("plan", "distill", "validate", "inspect"):
        q = sub.add_parser(name)
        if name == "validate" or name == "inspect": q.add_argument("--generation", required=True)
        else: q.add_argument("--output", required=True)
        if name == "plan": q.add_argument("--phase7-root", required=True)
        if name == "distill": q.add_argument("--plan", required=True)
    return p

def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        report = validate_generation(args.generation)
        payload = report if isinstance(report, dict) else {"valid": getattr(report, "ok", False), "errors": list(getattr(report, "errors", ())) }
    elif args.command == "inspect":
        manifest = Path(args.generation) / "manifest.json"
        try: payload = json.loads(manifest.read_text(encoding="utf-8"))
        except OSError as exc: payload = {"status": "INVALID", "error": str(exc)}
        else: payload = {k: payload.get(k) for k in ("generation_id", "status", "counts", "source_fingerprints", "policy_versions")}
    elif args.command == "plan":
        payload = {"status": "PLAN_REQUIRES_EXPLICIT_SOURCE", "phase7_root": str(Path(args.phase7_root)), "output": str(Path(args.output))}
    else:
        payload = {"status": "DISTILLATION_TEACHER_NOT_CONFIGURED"}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload.get("valid", payload.get("status") not in {"INVALID"}) else 1

if __name__ == "__main__":
    raise SystemExit(main())
