"""Paper figures from data/<run>/*.json.  Run from the paper folder:  python scripts/make_figures.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)
EXP = ["null_s0", "null_s1", "null_s2"]
ABL = ["nonull_s0", "notime_s0"]
PL = ["powerlaw_s0", "powerlaw_s1", "powerlaw_s2"]

# validated categorical slots 1-3 (all-pairs CVD-safe); references in neutral ink; lines also differ in style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(run, name):
    return json.load(open(DATA / run / f"{name}.json"))


def grid(ax):
    ax.grid(True, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Figure 1: forecasting quality
def pooled_excess(runs):
    rows = [load(r, "baseline")["rows"] for r in runs]
    ms = [x["context"] for x in rows[0]]
    out = {}
    for k in ("model", "bayes", "ml", "poisson"):
        w = np.array([[x["n"] for x in rr] for rr in rows], float)
        v = np.array([[x[k] - x["oracle"] for x in rr] for rr in rows])
        out[k] = (v * w).sum(0) / w.sum(0)
    return np.array(ms), out


fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.55})
for ax, runs, title in ((axes[0], EXP, "Exponential kernel"), (axes[1], PL, "Power-law kernel")):
    m, ex = pooled_excess(runs)
    ax.plot(m, ex["poisson"], color=INK2, ls=":", lw=1.2, label="Poisson rate")
    ax.plot(m, ex["ml"], color=AQUA, ls="-.", marker="^", ms=3.2, label="ML plug-in")
    ax.plot(m, ex["bayes"], color=ORANGE, ls="--", lw=1.6, label="Bayes-optimal")
    ax.plot(m, ex["model"], color=BLUE, marker="o", ms=3.2, lw=1.1, label="Transformer")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(m)
    ax.set_xticklabels([str(v) for v in m])
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel("events seen $m$")
    ax.set_title(title, color=INK)
    grid(ax)
axes[0].set_ylabel("excess cross-entropy\nover oracle (nats)")
axes[0].legend(loc="lower left", handlelength=2.2, borderaxespad=0.2)

ax = axes[2]
for runs, color, mk, lab in ((EXP + ABL, BLUE, "o", "exponential (5 models)"), (PL, ORANGE, "s", "power law (3 models)")):
    for j, r in enumerate(runs):
        rows = load(r, "baseline")["rows"]
        m = np.array([x["context"] for x in rows], float) * 2 ** ((j - (len(runs) - 1) / 2) * 0.07 + (0.12 if color == ORANGE else -0.12))
        d = np.array([x["model_minus_bayes"] for x in rows])
        se = np.array([x["model_minus_bayes_se"] for x in rows])
        ax.errorbar(m, d, yerr=2 * se, fmt=mk, ms=2.4, color=color, ecolor=color, elinewidth=0.6, capsize=0,
                    label=lab if j == 0 else None, alpha=0.9)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xscale("log", base=2)
ms_ = [2, 4, 8, 16, 32, 64, 128]
ax.set_xticks(ms_)
ax.set_xticklabels([str(v) for v in ms_])
ax.set_ylim(-0.009, 0.009)
ax.set_xlabel("events seen $m$")
ax.set_ylabel("model $-$ Bayes (nats)", labelpad=1)
ax.set_title("Gap to Bayes ($\\pm$2 s.e.)", color=INK)
ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
grid(ax)
fig.savefig(FIG / "fig_bayes.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: posterior probes by layer


def layer_curves(runs, k):
    P = [load(r, "posterior") for r in runs]
    L = [row["layer"] for row in P[0]["rows"]]
    get = lambda t: np.array([[row[f"{k}__{t}"] for row in p["rows"]] for p in P])  # noqa: E731
    return np.array(L), get("posterior_mean"), get("true"), get("posterior_sd"), np.array([p["ceiling"][k] for p in P])


fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
panels = ((axes[0], EXP, "alpha", "$\\alpha$ (exponential)"), (axes[1], EXP, "log_beta", "$\\log\\beta$ (exponential)"),
          (axes[2], PL, "gamma", "$\\gamma$ (power law)"))
for ax, runs, k, title in panels:
    L, pm, tr, sd, ce = layer_curves(runs, k)
    for v, color, ls, mk, lab in ((pm, BLUE, "-", "o", "posterior mean"), (sd, AQUA, "-.", "^", "posterior s.d."),
                                  (tr, ORANGE, "--", "s", "true value")):
        ax.errorbar(L, v.mean(0), yerr=v.std(0, ddof=1), color=color, ls=ls, marker=mk, ms=3, lw=1.2, elinewidth=0.6,
                    capsize=0, label=lab)
    ax.axhline(ce.mean(), color=INK2, ls=(0, (1, 1.5)), lw=1.0)
    ax.text(-0.15, ce.mean() + 0.02, "ceiling", color=INK2, fontsize=6.5, ha="left", va="bottom")
    ax.set_xticks(L)
    ax.set_xticklabels(["emb"] + [str(v) for v in L[1:]])
    ax.set_xlabel("after block")
    ax.set_title(title, color=INK)
    ax.set_ylim(-0.05, 1.04)
    grid(ax)
axes[0].set_ylabel("probe $R^2$ (held-out sequences)")
axes[0].legend(loc="lower right", handlelength=2.0, borderaxespad=0.2)
fig.savefig(FIG / "fig_posterior.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Figure 3: whose prior, and a causal posterior
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85), gridspec_kw={"width_ratios": [1.15, 1, 0.9], "wspace": 0.85})
ax = axes[0]
order = EXP + ABL + PL
for i, r in enumerate(order):
    d = next(x for x in load(r, "r8_misspec_ci")["data"].values() if x["misspecified"])
    ax.plot(i, d["kl_true"], "s", color=ORANGE, ms=3.2, label="true family's prior" if i == 0 else None)
    ax.plot(i, d["kl_own"], "o", color=BLUE, ms=3.2, label="model's own prior" if i == 0 else None)
    ax.plot([i, i], [d["kl_own"], d["kl_true"]], color=GRID, lw=0.8, zorder=0)
ax.axvline(len(EXP + ABL) - 0.5, color=INK2, lw=0.5, ls=":")
ax.set_yscale("log")
ax.set_ylim(8e-5, 1e-2)
ax.set_xticks([2, 6])
ax.set_xticklabels(["exponential\nmodels", "power-law\nmodels"], fontsize=6.3)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("KL(Bayes $\\|$ model) (nats)")
ax.set_title("Whose posterior predictive?", color=INK)
ax.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1, fontsize=6.2)
grid(ax)

blocks = [0, 1, 2, 3]
curves = {k: [] for k in ("das3", "das8", "encoder3", "decoder3", "random8", "full")}
ovl_e, ovl_d = [], []
for r in EXP + ABL:
    B = {(b["block"], b["mode"]): b for b in load(r, "r8_das")["blocks"]}
    for k in curves:
        curves[k].append([B[(li, "prefix")][k] for li in blocks])
    ovl_e.append([B[(li, "prefix")]["das8_overlap_encoder"] for li in blocks])
    ovl_d.append([B[(li, "prefix")]["das8_overlap_decoder"] for li in blocks])
x = np.array(blocks) + 1
ax = axes[1]
for k, color, ls, mk, lab in (("das3", BLUE, "-", "o", "DAS (3)"), ("encoder3", AQUA, "-.", "^", "encoder"),
                               ("full", INK2, ":", "D", "whole"), ("random8", ORANGE, "--", "s", "random (8)"),
                               ("decoder3", INK, "-", "x", "probe")):
    v = np.array(curves[k])
    ax.errorbar(x, v.mean(0), yerr=v.std(0, ddof=1), color=color, ls=ls, marker=mk, ms=3, lw=1.1, elinewidth=0.6, capsize=0)
    yl = {"das3": 0.93, "encoder3": 0.37, "full": 0.53, "random8": 0.13, "decoder3": 0.01}[k]
    ax.text(4.18, yl, lab, color=color, fontsize=5.9, va="center", ha="left")
ax.set_xticks(x)
ax.set_xlim(0.8, 4.2)
ax.set_ylim(-0.05, 1.03)
ax.set_xlabel("patched after block")
ax.set_ylabel("transfer to Bayes transplant")
ax.set_title("Swapping the posterior", color=INK)
grid(ax)

ax = axes[2]
for v, color, ls, mk, lab in ((np.array(ovl_e), AQUA, "-.", "^", "encoder directions"),
                              (np.array(ovl_d), INK, "-", "x", "probe directions")):
    ax.errorbar(x, v.mean(0), yerr=v.std(0, ddof=1), color=color, ls=ls, marker=mk, ms=3, lw=1.1, elinewidth=0.6, capsize=0,
                label=lab)
ax.axhline(8 / 128, color=INK2, lw=0.8, ls=":")
ax.text(4.05, 8 / 128 + 0.02, "chance", color=INK2, fontsize=6.3, ha="right", va="bottom")
ax.set_xticks(x)
ax.set_ylim(-0.03, 1.0)
ax.set_xlabel("after block")
ax.set_ylabel("share inside DAS subspace")
ax.set_title("Which directions are used?", color=INK)
ax.legend(loc="center right", fontsize=6.0, borderaxespad=0.1)
grid(ax)
fig.savefig(FIG / "fig_causal.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Figure 4: likelihood pooling
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8), gridspec_kw={"wspace": 0.48})
ax = axes[0]
for runs, color, mk, lab in ((EXP + ABL, BLUE, "o", "exponential"), (PL, ORANGE, "s", "power law")):
    for j, r in enumerate(runs):
        w = {x["layer"] + 1: x["delta_nats"] for x in load(r, "mech")["windows"] if x["window"] == 4}
        b = np.array(sorted(w)) + (j - (len(runs) - 1) / 2) * 0.06 + (0.14 if color == ORANGE else -0.1)
        ax.plot(b, [w[k] for k in sorted(w)], mk, color=color, ms=3, alpha=0.9, label=lab if j == 0 else None)
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("block limited to the last 4 events")
ax.set_ylabel("loss increase (nats)")
ax.set_title("Which block needs long context?", color=INK)
ax.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.2)
grid(ax)

ax = axes[1]
for j, r in enumerate(EXP + ABL):
    A = load(r, "r8_pooled")["attention"]
    for a in A:
        xx = a["block"] + 1 + (j - 2) * 0.07
        pool = a["tv_uniform"] < 0.25
        ax.plot(xx, a["tv_uniform"], "o" if pool else ".", color=BLUE if pool else INK2, ms=3.2 if pool else 2.6,
                alpha=0.95 if pool else 0.6)
ax.axhline(0.25, color=INK2, lw=0.5, ls=":")
ax.set_xticks([1, 2, 3, 4])
ax.set_ylim(-0.03, 1.0)
ax.set_xlabel("block")
ax.set_ylabel("distance from uniform\nattention over past events", fontsize=6.8)
ax.set_title("Pooling heads", color=INK)
grid(ax)

ax = axes[2]
for k, color, ls, mk, lab in (("recovered_bank_pooled", BLUE, "-", "o", "traces + pooled log-traces"),
                              ("recovered_bank", ORANGE, "--", "s", "traces at the current event")):
    v = np.array([[x[k] for x in load(r, "r8_pooled")["reconstruction"]] for r in EXP + ABL], float)
    xb = np.arange(v.shape[1]) + 1
    ax.errorbar(xb, v.mean(0), yerr=v.std(0, ddof=1), color=color, ls=ls, marker=mk, ms=3, lw=1.1, elinewidth=0.6,
                capsize=0, label=lab)
ax.set_xticks(xb)
ax.set_ylim(0.3, 1.02)
ax.set_xlabel("block output replaced")
ax.set_ylabel("loss recovered")
ax.set_title("Replacing a block by features", color=INK)
ax.legend(loc="lower left", fontsize=6.0, borderaxespad=0.1, handlelength=1.8)
grid(ax)
fig.savefig(FIG / "fig_mechanism.pdf")
plt.close(fig)

# ------------------------------------------------------------------ appendix figure: parameter edits (round 3)
fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8))
params = [("alpha", "$\\alpha$"), ("log_mu", "$\\log\\mu$"), ("log_beta", "$\\log\\beta$")]
for i, (p, lab) in enumerate(params):
    own, other, rnd = [], [], []
    for r in EXP + ABL:
        rows = load(r, "transplant")
        best = max((x for x in rows if x["param"] == p and x["method"] == "param_map"),
                   key=lambda x: x["pattern_own"] - x["pattern_best_other"])
        rr = next(x for x in rows if x["param"] == p and x["layer"] == best["layer"] and x["method"] == "ctrl_random")
        own.append(best["pattern_own"]); other.append(best["pattern_best_other"]); rnd.append(rr["pattern_own"])
    for off, vals, color, mk, name in ((-0.22, own, BLUE, "o", "edit $\\to$ own pattern"),
                                       (0.0, other, ORANGE, "s", "edit $\\to$ other parameters' pattern"),
                                       (0.22, rnd, INK2, "x", "random edit $\\to$ own pattern")):
        ax.plot(np.full(len(vals), i + off) + np.linspace(-0.05, 0.05, len(vals)), vals, mk, color=color, ms=3,
                label=name if i == 0 else None)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks(range(len(params)))
ax.set_xticklabels([f"edit {lab}" for _, lab in params])
ax.set_ylabel("within-sequence correlation")
ax.set_ylim(-1.3, 0.85)
ax.set_yticks([-0.5, 0.0, 0.5])
ax.legend(loc="lower center", ncol=1, handletextpad=0.2, borderaxespad=0.1, fontsize=6.2)
grid(ax)
fig.savefig(FIG / "fig_edits.pdf")
plt.close(fig)
print("figures written to", FIG)
