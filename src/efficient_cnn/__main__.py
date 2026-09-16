"""python -m efficient_cnn"""

from __future__ import annotations

from efficient_cnn.eval import format_report, run_before_after


def main() -> None:
    report = run_before_after()
    print(format_report(report))


if __name__ == "__main__":
    main()
