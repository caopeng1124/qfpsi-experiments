"""Publication figure for Section 6: measured overlap-vs-similarity (5 seeds).

Single-column USENIX figure (3.4 in wide), so the fonts stay readable at 1:1.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/predicate_multiseed.json")
out_stem = Path(sys.argv[2] if len(sys.argv) > 2 else "figures/fig_predicate_v2")
d = json.loads(src.read_text())
rho = np.array(d["rho"])
agg = d["aggregate"]

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6})
fig, ax = plt.subplots(figsize=(3.4, 2.15))

sel = [("B1w2", "B=1, w=2", "C0", "-", "o"),
       ("B2w2", "B=2, w=2", "C2", "-", "s"),
       ("B2w4", "B=2, w=4", "C1", "--", "^"),
       ("B4w8", "B=4, w=8 (previous default)", "C3", ":", "x")]
for key, lab, col, ls, mk in sel:
    m = np.array(agg[key]["mean"])
    s = np.array(agg[key]["mean_std"])
    ax.plot(rho, m, ls=ls, color=col, marker=mk, ms=2.2, lw=0.9, label=lab)
    ax.fill_between(rho, m - s, m + s, color=col, alpha=0.18, lw=0)

ax.set_xlabel("cosine similarity of the pair")
ax.set_ylabel(r"token overlap (out of $T{=}128$)")
ax.grid(alpha=0.25, lw=0.4)
ax.set_ylim(-4, 112)
ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.3)
plt.tight_layout(pad=0.2)
fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
print("wrote", out_stem.with_suffix(".pdf"), "and .png")
