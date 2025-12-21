import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from ruamel.yaml import YAML


def main():
    if len(sys.argv) < 2:
        print("Usage: python chart.py <results.yml>")
        sys.exit(1)

    yaml = YAML()
    with open(sys.argv[1]) as f:
        data = yaml.load(f)

    attrs = [
        a
        for a in data["attributes"]
        if a["diff"]["mean"] != 0 and abs(a["diff"]["mean"]) >= 1.96 * a["diff"]["sem"]
    ]
    names = [a["name"] for a in attrs]
    diffs = [a["diff"]["mean"] for a in attrs]
    sems = [a["diff"]["sem"] for a in attrs]
    _fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    colors = ["#a8d5ba" if d >= 0 else "#f4a9a8" for d in diffs]
    ax.barh(
        names,
        diffs,
        xerr=sems,
        color=colors,
        edgecolor="black",
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1},
    )
    ax.axvline(0, color="black", linewidth=1)
    outline = [pe.withStroke(linewidth=2, foreground="white")]
    for i, d in enumerate(diffs):
        ax.text(
            d / 2,
            i,
            f"{d:+.1f}",
            va="center",
            ha="center",
            fontsize=8,
            path_effects=outline,
        )
    max_abs = max(abs(d) + s for d, s in zip(diffs, sems, strict=False)) * 1.1
    ax.set_xlim(-max_abs, max_abs)
    ax.set_xlabel("Difference (B - A)")
    ax.annotate(
        "Error bars: ±1 SEM",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
    )
    ax.set_title(
        f"{data['variants'][1]} vs {data['variants'][0]} (n={data['num_runs']})"
    )
    ax.invert_yaxis()

    plt.tight_layout()
    output_path = Path(sys.argv[1]).with_suffix(".png")
    plt.savefig(output_path, dpi=200)
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    main()
