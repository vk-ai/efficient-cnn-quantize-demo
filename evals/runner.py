#!/usr/bin/env python3
"""CI-friendly eval CLI — writes JSON report under evals/.

``python evals/runner.py``                       full before/after report → last_report.json
``python evals/runner.py --mixed-precision ...`` round-5 auto mixed precision only
                                                 → mixed_precision.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficient_cnn.config import load_config
from efficient_cnn.eval import format_report, run_auto_mixed_precision_demo, run_before_after
from efficient_cnn.mixed_precision import format_mixed_precision

OUT = Path(__file__).resolve().parent / "last_report.json"
MP_OUT = Path(__file__).resolve().parent / "mixed_precision.json"


def _mixed_precision(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    amp = dict(cfg["quantize"].get("auto_mixed_precision") or {})
    if args.max_acc_drop is not None:
        amp["max_acc_drop"] = args.max_acc_drop
    if args.granularity is not None:
        amp["granularity"] = args.granularity
    if args.ladder is not None:
        amp["ladder"] = args.ladder
    cfg["quantize"]["auto_mixed_precision"] = {**amp, "enabled": True}
    out = run_auto_mixed_precision_demo(cfg)
    for key, rep in out.items():
        print(format_mixed_precision(rep, key) + "\n")
    MP_OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {MP_OUT}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Efficient CNN quantize/prune eval")
    ap.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    ap.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Only run the round-5 sensitivity-driven per-layer precision search",
    )
    ap.add_argument("--max-acc-drop", type=float, default=None, help="Accuracy budget")
    ap.add_argument("--granularity", choices=["per_tensor", "per_channel"], default=None)
    ap.add_argument("--ladder", nargs="+", default=None, help="e.g. float 8 4 2")
    args = ap.parse_args(argv)
    if args.mixed_precision:
        _mixed_precision(args)
        return
    report = run_before_after(load_config(args.config) if args.config else None)
    print(format_report(report))
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
