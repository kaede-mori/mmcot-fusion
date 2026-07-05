"""Regenerate the stage-1 loss comparison figure (human vs teacher rationales) for the report.

Usage: uv run --group dev python scripts/make_loss_figure.py
Requires experiments_hpc/e{1,8}-rationale_*/train_log.json.
"""

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series(tag, label):
    d = glob.glob(f"experiments_hpc/{tag}-rationale_*")[0]
    log = json.load(open(f"{d}/train_log.json"))
    entries = [l for l in log["log_history"] if "loss" in l]
    return [l["step"] for l in entries], [l["loss"] for l in entries], label


e1 = series("e1", "Human rationales")
e8 = series("e8", "GPT-4o teacher rationales")

fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
ax.plot(e1[0], e1[1], color="#2f6fb2", lw=1.6, ls="-", label=e1[2])
ax.plot(e8[0], e8[1], color="#c1662f", lw=1.6, ls="--", label=e8[2])
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("stage-1 training loss (log)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.5)
ax.legend(frameon=False, fontsize=8)
for xs, ys, _c, dy, ha, fmt in (
    (e1[0], e1[1], "#444444", -11, "right", ".4f"),
    (e8[0], e8[1], "#444444", 6, "right", ".2f"),
):
    ax.annotate(
        f"{ys[-1]:{fmt}}",
        xy=(xs[-1], ys[-1]),
        fontsize=7,
        color="#444444",
        xytext=(-4, dy),
        textcoords="offset points",
        ha=ha,
    )
fig.tight_layout()
fig.savefig("report/figures/loss_curves.pdf")
print("saved; final losses:", e1[1][-1], e8[1][-1])
