#!/usr/bin/env python3
"""Quickstart: train tiny efficient CNN, then int8 fake-quant + magnitude prune."""

from __future__ import annotations

from efficient_cnn.eval import format_report, run_before_after


def main() -> None:
    report = run_before_after()
    print(format_report(report))
    print()
    print("Tips: edit configs/default.yaml for width, sparsity, bits, epochs.")


if __name__ == "__main__":
    main()
