"""Shared plumbing for the experiment scripts: arguments, tables, figure style."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hedging import config

RESULTS = Path(__file__).resolve().parent.parent / "results"

TYPE_COLOURS = {"night-only": "#1f4e79", "flexible": "#c0504d", "early": "#6a994e"}

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def setup(description, seasons=600):
    """Parse the common options and load the market."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seasons", type=int, default=seasons,
                        help="simulated seasons per data point (default %(default)s)")
    parser.add_argument("--config", default=config.DEFAULT_PATH, help="market description (yaml)")
    args = parser.parse_args()
    return config.load(args.config), args


def save_table(rows, name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    return path


def save_figure(fig, name):
    RESULTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png")
    plt.close(fig)


def print_table(rows, columns=None):
    columns = columns or list(rows[0])
    cells = [[_fmt(r[c]) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in cells:
        print("  ".join(x.ljust(w) for x, w in zip(row, widths)))


def _fmt(x):
    if isinstance(x, float):
        return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3f}"
    return str(x)


def describe(profile):
    """Compact text for a strategy mix, e.g. 'night-only: hedge2-timely 0.95'."""
    parts = []
    for kind, mix in profile.items():
        main = [f"{name} {share:.2f}" for name, share in mix.items() if share >= 0.05]
        parts.append(f"{kind}: {', '.join(main)}")
    return "; ".join(parts)
