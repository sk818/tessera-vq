"""Phase 5 analysis (downstream linear probes); filename keeps ``phase4`` per docs/spec.md.

Stub at Phase 0: defines the CLI surface only. The probe orchestration is implemented in
Phase 5 of docs/spec.md.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the downstream-probes phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    return parser


def main() -> None:
    """Entry point. Not yet implemented at Phase 0."""
    args = build_parser().parse_args()
    raise SystemExit(
        f"phase4_downstream is a stub (seed={args.seed}, config={args.config}); "
        "implemented in Phase 5 of docs/spec.md."
    )


if __name__ == "__main__":
    main()
