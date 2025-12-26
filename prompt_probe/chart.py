from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


def generate_chart(data: dict, output_path: Path):
    criteria = [
        c
        for c in data["criteria"]
        if c["diff"]["mean"] != 0 and abs(c["diff"]["mean"]) >= 1.96 * c["diff"]["sem"]
    ]
    names = [c["name"] for c in criteria]
    diffs = [c["diff"]["mean"] for c in criteria]
    sems = [c["diff"]["sem"] for c in criteria]
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
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Chart saved to {output_path}")
