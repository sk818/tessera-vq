"""WS-4 synthesis: the compression-vs-downstream-accuracy Pareto figure.

Joins the index-compression bytes/px (WS-2, ``*_index_compression.parquet``) with the
downstream F1 retention (WS-3, ``*_downstream.parquet``) and plots accuracy retention
(reconstructed / raw macro-F1) against compression ratio over int8, one point per
``(t, k1)`` cell. The recommended operating point (t=512, k1=20) is starred.

Writes ``figures/phase4_pareto.png`` and ``.pdf``.

Run::

    uv run python scripts/phase4_pareto.py --downstream results/phase4/austria_downstream.parquet \
        --bytes results/phase3/idx_v2_index_compression.parquet --tag austria
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the two input parquets and the output tag."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--downstream", default="results/phase4/austria_downstream.parquet")
    p.add_argument("--bytes", default="results/phase3/idx_v2_index_compression.parquet")
    p.add_argument("--tag", default="austria", help="dataset name for the title")
    p.add_argument("--out-dir", default="figures")
    return p.parse_args()


def _load(downstream: str, bytes_path: str) -> pd.DataFrame:
    """Merge downstream F1 with byte cost on (t, k1, k2); add retention column."""
    f1 = pd.read_parquet(downstream)
    by = pd.read_parquet(bytes_path)
    m = f1.merge(
        by[["t", "k1", "k2", "total_compressed_Bpx", "x_int8_compressed"]],
        on=["t", "k1", "k2"],
    )
    m["retention"] = m["f1_recon_mean"] / m["f1_raw_mean"]
    return m


def _plot(m: pd.DataFrame, tag: str, out_dir: str) -> Path:
    """Scatter retention vs compression, colour by t, annotate k1, star the pick."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colours = {512: "#1f77b4", 1024: "#d62728"}
    for t, grp in m.groupby("t"):
        ax.scatter(
            grp["x_int8_compressed"],
            grp["retention"],
            s=90,
            color=colours.get(int(t), "#555"),
            label=f"t={int(t)}",
            zorder=3,
        )
        for _, r in grp.iterrows():
            ax.annotate(
                f"k1={int(r.k1)}",
                (r.x_int8_compressed, r.retention),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
    pick = m[(m.t == 512) & (m.k1 == 20)]  # noqa: PLR2004
    if not pick.empty:
        ax.scatter(
            pick["x_int8_compressed"],
            pick["retention"],
            s=320,
            marker="*",
            edgecolor="black",
            facecolor="gold",
            zorder=4,
            label="recommended",
        )
    ax.axhline(1.0, ls="--", c="grey", lw=1, label="lossless (recon = raw)")
    ax.set_xlabel("compression ratio over int8 (higher = smaller files)")
    ax.set_ylabel("downstream F1 retention  (recon / raw macro-F1)")
    ax.set_title(f"VQ compression vs downstream accuracy — {tag} (4-fold spatial CV)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "phase4_pareto.png"
    fig.savefig(png, dpi=150)
    fig.savefig(out / "phase4_pareto.pdf")
    plt.close(fig)
    return png


def main() -> None:
    """Build and save the Pareto figure."""
    args = parse_args()
    m = _load(args.downstream, args.bytes)
    path = _plot(m, args.tag, args.out_dir)
    print(f"wrote {path} (+ .pdf)")


if __name__ == "__main__":
    main()
