"""Phase 4 analysis (index map compression, pipelines A-G); filename keeps ``phase3``.

Stub at Phase 0: defines the CLI surface only. The pipelines are implemented in Phase 4
of docs/spec.md.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the index-compression phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    return parser


def main() -> None:
    """Entry point. Not yet implemented at Phase 0."""
    args = build_parser().parse_args()
    raise SystemExit(
        f"phase3_index_compression is a stub (seed={args.seed}, config={args.config}); "
        "implemented in Phase 4 of docs/spec.md."
    )


if __name__ == "__main__":
    main()
