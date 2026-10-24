"""Transfer curve figure: release probability vs cosine, threshold at 99% recall."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/transfer.json")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures/fig_transfer_v2")
d = json.loads(src.read_text())
rho = np.array(d["rho"])

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6})
fig, ax = plt.subplots(figsize=(3.4, 2.15))

style = {"B2_w2_T128": ("B=2, w=2, T=128", "C0", "-", "o"),
         "B2_w4_T256": ("B=2, w=4, T=256", "C1", "--", "^"),
         "B2_w2_T512": ("B=2, w=2, T=512", "C2", ":", "s")}
for key, (lab, col, ls, mk) in style.items():
    if key not in d["configs"]:
        continue
    m = 100 * np.array(d["configs"][key]["mean"])
    s = 100 * np.array(d["configs"][key]["std"])
    ax.plot(rho, m, ls=ls, color=col, marker=mk, ms=2.2, lw=0.9, label=lab)
    ax.fill_between(rho, m - s, m + s, color=col, alpha=0.15, lw=0)

ax.axvline(0.85, color="k", lw=0.7, ls="--", alpha=0.7)
ax.text(0.838, 55, r"target $\tau=0.85$", fontsize=5.6, rotation=90, ha="right")
ax.axhline(50, color="gray", lw=0.5, ls=":", alpha=0.8)
ax.set_xlabel("cosine similarity of the pair")
ax.set_ylabel("release probability (%)")
ax.set_ylim(-3, 107)
ax.set_xlim(0.49, 0.96)
ax.grid(alpha=0.25, lw=0.4)
ax.legend(frameon=False, loc="upper left", handlelength=1.6)
plt.tight_layout(pad=0.2)
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
print("wrote", out.with_suffix(".pdf"))
