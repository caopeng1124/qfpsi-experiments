"""
QFPSI pilot: does the paper's token construction produce the claimed overlap scale?

Builds pairs with controlled cosine similarity, applies the construction from
Section 4 of paper.tex (normalize -> random rotation -> B-bit uniform quantization
-> T random windows of size w -> hash -> overlap), and reports
  * mean/std overlap for close (cos = 0.85) and far pairs
  * overlap-vs-cosine curves for B in {1,2,4}, w in {1,2,4,8}
  * best achievable FMR/FNMR trade-off (threshold sweep)
  * a "soft" overlap variant (per-coordinate agreement) for comparison

Everything is measured here; nothing is projected.
"""
import json
import numpy as np

RNG = np.random.default_rng(20260719)
D = 128
T = 128
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
N_PER_RHO = 1000


def make_pairs(rho, n):
    """Return (q, x) unit vectors with <q,x> = rho exactly."""
    q = RNG.normal(size=(n, D))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    u = RNG.normal(size=(n, D))
    u -= (u * q).sum(1, keepdims=True) * q          # orthogonalize ...
    u /= np.linalg.norm(u, axis=1, keepdims=True)   # ... and renormalize
    x = rho * q + np.sqrt(max(0.0, 1 - rho * rho)) * u
    return q, x


def random_rotation(d):
    a = RNG.normal(size=(d, d))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def quantize(z, B, zmin, zmax):
    lv = (1 << B) - 1
    u = np.rint((z - zmin) / (zmax - zmin) * lv)
    return np.clip(u, 0, lv).astype(np.uint8)


def mix64(x):
    x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def tokens(u, W, B):
    """Pack the w codes of each window into one integer, then mix."""
    n = u.shape[0]
    tok = np.zeros((n, W.shape[0]), dtype=np.uint64)
    for k in range(W.shape[1]):
        tok |= (u[:, W[:, k]].astype(np.uint64) << np.uint64(B * k))
    return mix64(tok)


P = random_rotation(D)
ZMIN, ZMAX = -3.0 / np.sqrt(D), 3.0 / np.sqrt(D)

# --- build all pairs -------------------------------------------------------
vecs, labels = [], []
for rho in RHO_GRID:
    q, x = make_pairs(rho, N_PER_RHO)
    # interleave so that rows 2i / 2i+1 are the two halves of pair i
    vecs.append(np.stack([q, x], axis=1).reshape(-1, D))
    labels += [rho] * N_PER_RHO * 2
V = np.vstack(vecs)
V = V / np.linalg.norm(V, axis=1, keepdims=True)
Z = V @ P                      # (n, D) rotated (no centering: anchor = 0)
Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

idx_q = np.arange(0, 2 * len(RHO_GRID) * N_PER_RHO, 2)
idx_x = idx_q + 1

results = {"D": D, "T": T, "rho_grid": RHO_GRID, "configs": {}}
fig_curves, fig_dists = [], []

for B in (1, 2, 4):
    U = quantize(Z, B, ZMIN, ZMAX)
    for w in (1, 2, 4, 8):
        W = np.stack([RNG.choice(D, w, replace=False) for _ in range(T)])  # (T,w)
        tok = tokens(U, W, B)
        ov = (tok[idx_q] == tok[idx_x]).sum(1).reshape(len(RHO_GRID), N_PER_RHO)
        means = ov.mean(1).round(2).tolist()
        stds = ov.std(1).round(2).tolist()
        # soft variant: per-coordinate agreement averaged over windows
        agree = (U[idx_q][:, W] == U[idx_x][:, W]).mean((1, 2)).reshape(len(RHO_GRID), N_PER_RHO)
        soft = agree.mean(1).round(3).tolist()
        key = f"B{B}_w{w}"
        results["configs"][key] = {"mean": means, "std": stds, "soft_mean": soft}
        fig_curves.append((B, w, np.array(means), np.array(stds)))
        if (B, w) in ((1, 2), (2, 2)):
            fig_dists.append((B, w, ov))
        print(f"B={B} w={w:2d} | close(rho=.85) {means[RHO_GRID.index(0.85)]:6.1f}"
              f"  far(rho=.3) {means[RHO_GRID.index(0.3)]:6.1f}"
              f"  far(rho=0) {means[0]:6.1f}  soft(.85)={soft[RHO_GRID.index(0.85)]:.3f}")

# --- FMR/FNMR separability for the paper's default claim -------------------
def fmr_fnmr(close_scores, far_scores):
    ts = np.arange(0, T + 1)
    fmr = np.array([(far_scores >= t).mean() for t in ts])
    fnmr = np.array([(close_scores < t).mean() for t in ts])
    return ts, fmr, fnmr

close_scores = results["configs"]["B4_w8"]["mean"]  # placeholder, real arrays below
print("\n--- threshold sweep for selected configs (close: rho>=0.85, far: rho<=0.5) ---")
sep = {}
for B in (1, 2, 4):
    U = quantize(Z, B, ZMIN, ZMAX)
    for w in (1, 2, 4, 8):
        W = np.stack([RNG.choice(D, w, replace=False) for _ in range(T)])
        tok = tokens(U, W, B)
        ov = (tok[idx_q] == tok[idx_x]).sum(1).reshape(len(RHO_GRID), N_PER_RHO)
        cs = ov[RHO_GRID.index(0.85)].astype(float).tolist() + ov[RHO_GRID.index(0.9)].astype(float).tolist() \
             + ov[RHO_GRID.index(0.95)].astype(float).tolist()
        fs = np.concatenate([ov[i] for i in range(RHO_GRID.index(0.5) + 1)]).astype(float).tolist()
        ts, fmr, fnmr = fmr_fnmr(np.array(cs), np.array(fs))
        # smallest threshold that still meets FMR <= 1e-3 / 1e-2
        best = {}
        for eps in (1e-2, 1e-3):
            ok = np.where(fmr <= eps)[0]
            if len(ok):
                i = ok[0]
                best[f"eps{eps:g}"] = {"t": int(ts[i]), "fmr": float(fmr[i]), "fnmr": float(fnmr[i])}
            else:
                best[f"eps{eps:g}"] = None
        sep[f"B{B}_w{w}"] = {"close_mean": float(np.mean(cs)), "close_std": float(np.std(cs)),
                             "far_mean": float(np.mean(fs)), "far_std": float(np.std(fs)),
                             "dprime": float((np.mean(cs) - np.mean(fs)) / np.sqrt((np.var(cs) + np.var(fs)) / 2)),
                             "fp_at_fmr1e-3": best}
        print(f"B={B} w={w:2d} | close {np.mean(cs):6.1f}±{np.std(cs):4.1f}  far {np.mean(fs):5.1f}±{np.std(fs):4.1f}"
              f"  d'={sep[f'B{B}_w{w}']['dprime']:5.2f}  op@1e-2={best['eps0.01']}  op@1e-3={best['eps0.001']}")
results["separability"] = sep

with open("experiments/pilot/results.json", "w") as f:
    json.dump(results, f, indent=1)

# --- figures ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for B, w, m, s in fig_curves:
    ls = {1: "-", 2: "--", 4: ":"}[B]
    col = {1: "C0", 2: "C2", 4: "C3"}[B]
    ax[0].plot(RHO_GRID, m, ls=ls, color=col, marker="o", ms=3, label=f"B={B}, w={w}")
ax[0].set_xlabel("cosine similarity of the pair")
ax[0].set_ylabel("token overlap (out of T=128)")
ax[0].axhline(90, color="k", lw=0.8, alpha=.6)
ax[0].text(0.02, 92, "paper's t=90", fontsize=8)
ax[0].legend(fontsize=6, ncol=2)
ax[0].set_title("measured overlap vs similarity (synthetic Gaussian, D=128)")

for B, w, ov in fig_dists:
    ax[1].hist(ov[RHO_GRID.index(0.85)], bins=40, alpha=.55, label=f"close cos=.85, B={B},w={w}", density=True)
    ax[1].hist(ov[0], bins=40, alpha=.35, label=f"far cos=0, B={B},w={w}", density=True)
ax[1].set_xlabel("token overlap")
ax[1].set_ylabel("density")
ax[1].legend(fontsize=6)
ax[1].set_title("close vs far overlap distributions")
plt.tight_layout()
plt.savefig("experiments/pilot/overlap_pilot.png", dpi=160)
print("\nwrote experiments/pilot/results.json and experiments/pilot/overlap_pilot.png")
