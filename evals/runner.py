#!/usr/bin/env python3
"""CI-friendly eval CLI — writes JSON report under evals/."""

from __future__ import annotations

import json
from pathlib import Path

from efficient_cnn.eval import format_report, run_before_after

OUT = Path(__file__).resolve().parent / "last_report.json"


def main() -> None:
    report = run_before_after()
    print(format_report(report))
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
