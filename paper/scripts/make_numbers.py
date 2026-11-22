"""Every number in the paper comes from here: reads data/<run>/*.json, writes numbers.tex (macros) and tables/*.tex.

Run from the paper folder:  python scripts/make_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import re

import numpy as np

_MINUS = re.compile(r"(?<![\w$\-])-(\d)")


def tex_minus(text: str) -> str:
    """Typeset negative numbers with a real minus sign (leaves en-dash ranges like 0.1--0.2 alone)."""
    return _MINUS.sub(r"$-$\1", text)

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tables").mkdir(exist_ok=True)
DATA = ROOT / "data"
EXP = ["null_s0", "null_s1", "null_s2"]            # main exponential setting, 3 seeds
ABL = ["nonull_s0", "notime_s0"]                   # ablations (seed 0)
ALL_EXP = EXP + ABL
PL = ["powerlaw_s0", "powerlaw_s1", "powerlaw_s2"]  # power-law kernel, 3 seeds
ALL = ALL_EXP + PL
NEAR_CRIT = "a0.90-0.98 (unseen, near-critical)"


def load(run, name):
    return json.load(open(DATA / run / f"{name}.json"))


def late_key(ev):
    return [k for k in ev if k.endswith("_gap_frac")][-1].replace("_gap_frac", "")


def ms(values, fmt="{:.3f}"):
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(v.mean())} $\\pm$ {fmt.replace('+', '').format(v.std(ddof=1))}"


def rng(values, fmt="{:.2f}"):
    v = [x for x in values if x is not None]
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else f"{lo}--{hi}"


macros: dict[str, str] = {}


def M(name, value):
    macros[name] = value


# ---------------------------------------------------------------- behaviour: gap, Bayes, ML
def run_stats(r):
    ev = load(r, "eval")
    lk = late_key(ev["all"])
    bl = load(r, "baseline")["rows"]
    mech = load(r, "mech")
    return {
        "gap": ev["all"][f"{lk}_gap_frac"],
        "excess": ev["all"][f"{lk}_model"] - ev["all"][f"{lk}_oracle"],
        "gap_crit": ev[NEAR_CRIT][f"{lk}_gap_frac"],
        "gap_low": ev["a0.0-0.3"][f"{lk}_gap_frac"],
        "maxbayes": max(abs(x["model_minus_bayes"]) for x in bl),
        "maxbayes_se": max(abs(x["model_minus_bayes"]) / x["model_minus_bayes_se"] for x in bl),
        "ml2": next(x["model_minus_ml"] for x in bl if x["context"] == 2),
        "ml128": next(x["model_minus_ml"] for x in bl if x["context"] == 128),
        "rho": mech["intensity"]["spearman"],
        "rho_within": mech["intensity"]["within_seq_spearman_median"],
        "late": lk,
    }


S = {r: run_stats(r) for r in ALL}
late = S["null_s0"]["late"]  # e.g. ctx64-159
lo_, hi_ = late[3:].split("-")
M("lateFrom", lo_)
M("lateTo", str(int(hi_)))
M("gapExp", ms([S[r]["gap"] for r in EXP]))
M("gapPL", ms([S[r]["gap"] for r in PL]))
M("gapAllRange", rng([S[r]["gap"] for r in ALL], "{:.3f}"))
M("excessAllRange", rng([S[r]["excess"] for r in ALL], "{:.3f}"))
M("maxBayesAll", f"{max(S[r]['maxbayes'] for r in ALL):.4f}")
M("maxBayesExp", f"{max(S[r]['maxbayes'] for r in ALL_EXP):.4f}")
M("maxBayesPL", f"{max(S[r]['maxbayes'] for r in PL):.4f}")
M("maxBayesSE", f"{max(S[r]['maxbayes_se'] for r in ALL):.2f}")
zs = [abs(x["model_minus_bayes"]) / x["model_minus_bayes_se"] for r in ALL for x in load(r, "baseline")["rows"]]
M("nBayesTests", str(len(zs)))
M("nOverTwoSE", str(sum(z > 1.96 for z in zs)))
M("mlTwoRange", rng([-S[r]["ml2"] for r in ALL], "{:.2f}"))
M("gapCritRange", rng([S[r]["gap_crit"] for r in ALL], "{:.3f}"))
M("rhoRange", rng([S[r]["rho"] for r in ALL], "{:.2f}"))
M("rhoWithinRange", rng([S[r]["rho_within"] for r in ALL], "{:.2f}"))
bl0 = load("null_s0", "baseline")
M("nPrior", f"{bl0['n_prior']:,}".replace(",", "{,}"))
M("nBaseExp", str(bl0["rows"][0]["n"]))
M("nBasePL", str(load("powerlaw_s0", "baseline")["rows"][0]["n"]))
ess_exp = [x["median_ess"] for r in ALL_EXP for x in load(r, "baseline")["rows"] if x["context"] == 128]
ess_pl = [x["median_ess"] for r in PL for x in load(r, "baseline")["rows"] if x["context"] == 128]
M("essMinExp", f"{min(ess_exp):.0f}")
M("essMinPL", f"{min(ess_pl):.0f}")

SETTINGS = [EXP, ["nonull_s0"], ["notime_s0"], PL]
METRICS = [("gap fraction (late context)", "gap", "{:.3f}"), ("excess over oracle (nats)", "excess", "{:.3f}"),
           ("gap fraction, unseen $\\alpha \\ge 0.9$", "gap_crit", "{:.3f}"),
           ("max $|$model $-$ Bayes$|$ (nats)", "maxbayes", "{:.4f}"), ("model $-$ ML at $m{=}2$ (nats)", "ml2", "{:+.2f}"),
           ("intensity tracking (Spearman)", "rho_within", "{:.2f}")]
rows = [" & ".join([lab] + [ms([S[r][key] for r in runs], fmt) for runs in SETTINGS]) + r" \\" for lab, key, fmt in METRICS]
(ROOT / "tables" / "summary.tex").write_text(tex_minus("\n".join(rows) + "\n"))

# full baseline table (appendix)
rows = []
for r in ALL:
    for x in load(r, "baseline")["rows"]:
        rows.append(" & ".join([r.replace("_", r"\_") if x["context"] == 2 else "", str(x["context"]),
                                f"{x['model']:.4f}", f"{x['bayes']:.4f}", f"{x['ml']:.4f}", f"{x['oracle']:.4f}",
                                f"{x['poisson']:.4f}", ("0.0000" if abs(x['model_minus_bayes']) < 5e-5 else f"{x['model_minus_bayes']:+.4f}") + f" $\\pm$ {x['model_minus_bayes_se']:.4f}",
                                f"{x['median_ess']:.0f}"]) + r" \\")
    rows.append(r"\midrule")
(ROOT / "tables" / "baseline_full.tex").write_text(tex_minus("\n".join(rows[:-1]) + "\n"))

# ---------------------------------------------------------------- posterior probes
NAMES = {"log_mu": r"$\log\mu$", "alpha": r"$\alpha$", "log_beta": r"$\log\beta$", "gamma": r"$\gamma$"}
NAMES_PL = {**NAMES, "log_beta": r"$-\log c$"}


def post_stats(r):
    p = load(r, "posterior")
    out = {}
    for k in p["params"]:
        best_t = max(row[f"{k}__true"] for row in p["rows"])
        out[k] = {"true": best_t, "pm": max(row[f"{k}__posterior_mean"] for row in p["rows"]),
                  "sd": max(row[f"{k}__posterior_sd"] for row in p["rows"]), "ceil": p["ceiling"][k],
                  "share": best_t / p["ceiling"][k]}
    out["_ess"] = p["median_ess"]
    return out


P = {r: post_stats(r) for r in ALL}
rows = []
for kernel, runs, names in (("Exponential", EXP, NAMES), ("Power law", PL, NAMES_PL)):
    params = [k for k in P[runs[0]] if not k.startswith("_")]
    for i, k in enumerate(params):
        rows.append(" & ".join([kernel if i == 0 else "", names[k],
                                ms([P[r][k]["ceil"] for r in runs], "{:.2f}"),
                                ms([P[r][k]["true"] for r in runs], "{:.2f}"),
                                ms([P[r][k]["pm"] for r in runs], "{:.3f}"),
                                ms([P[r][k]["sd"] for r in runs], "{:.2f}"),
                                ms([100 * P[r][k]["share"] for r in runs], "{:.0f}") + r"\%"]) + r" \\")
    rows.append(r"\midrule")
(ROOT / "tables" / "posterior.tex").write_text(tex_minus("\n".join(rows[:-1]) + "\n"))
three = ["log_mu", "alpha", "log_beta"]
M("pmAlphaMuRange", rng([P[r][k]["pm"] for r in ALL for k in ("alpha", "log_mu")], "{:.2f}"))
M("pmBetaRange", rng([P[r]["log_beta"]["pm"] for r in ALL], "{:.2f}"))
M("pmGammaRange", rng([P[r]["gamma"]["pm"] for r in PL], "{:.2f}"))
M("pmMin", f"{min(P[r][k]['pm'] for r in ALL for k in three):.2f}")
M("trueRange", rng([P[r][k]["true"] for r in ALL for k in three], "{:.2f}"))
M("sdRange", rng([P[r][k]["sd"] for r in ALL for k in P[r] if not k.startswith("_")], "{:.2f}"))
M("shareExpRange", rng([100 * P[r][k]["share"] for r in ALL_EXP for k in three], "{:.0f}"))
M("sharePLRange", rng([100 * P[r][k]["share"] for r in PL for k in ("log_mu", "alpha")], "{:.0f}"))
M("sharePLBetaRange", rng([100 * P[r]["log_beta"]["share"] for r in PL], "{:.0f}"))
M("shareGammaRange", rng([100 * P[r]["gamma"]["share"] for r in PL], "{:.0f}"))
M("ceilGammaRange", rng([P[r]["gamma"]["ceil"] for r in PL], "{:.2f}"))
M("trueGammaRange", rng([P[r]["gamma"]["true"] for r in PL], "{:.2f}"))
M("essPostRange", rng([P[r]["_ess"] for r in ALL], "{:.0f}"))
n_verd = sum(1 for r in ALL for k in P[r] if not k.startswith("_"))
n_pm_pass = sum(1 for r in ALL for k in P[r] if not k.startswith("_") and P[r][k]["pm"] >= 0.85 and P[r][k]["pm"] >= P[r][k]["true"] + 0.05)
n_share_pass = sum(1 for r in ALL for k in P[r] if not k.startswith("_") and P[r][k]["share"] >= 0.9)
M("nPostVerdicts", str(n_verd))
M("nPmPass", str(n_pm_pass))
M("nSharePass", str(n_share_pass))
pr0 = load("null_s0", "posterior")
M("nPostSeqs", "400")
M("nPostRows", f"{len(pr0['rows'])}")

# where the posterior is assembled vs which block needs long context (exploratory)
match = 0
detail = []
for r in ALL:
    p = load(r, "posterior")
    w4 = {x["layer"] + 1: x["delta_nats"] for x in load(r, "mech")["windows"] if x["window"] == 4}
    Lstar = max(w4, key=w4.get)
    params = p["params"] if r in ALL_EXP else ["log_mu", "alpha", "log_beta"]
    final = {k: p["rows"][-1][f"{k}__posterior_mean"] for k in params}
    after = {k: p["rows"][Lstar][f"{k}__posterior_mean"] for k in params}
    before = {k: p["rows"][Lstar - 1][f"{k}__posterior_mean"] for k in params}
    done = all(after[k] >= final[k] - 0.03 for k in params)
    not_before = any(before[k] < final[k] - 0.03 for k in params)
    match += int(done and not_before)
    detail.append((r, Lstar, done, not_before))
M("nAssembledAtLong", str(match))
late3 = [r for r in ALL if max(w := {x["layer"] + 1: x["delta_nats"] for x in load(r, "mech")["windows"] if x["window"] == 4}, key=w.get) == 3]
M("lateRuns", " and ".join(r.replace("_", r"\_") for r in late3))
M("lateAlphaTwo", " and ".join(f"{load(r, 'posterior')['rows'][2]['alpha__posterior_mean']:.2f}" for r in late3))
M("lateAlphaThree", " and ".join(f"{load(r, 'posterior')['rows'][3]['alpha__posterior_mean']:.2f}" for r in late3))
early = [r for r in ALL if r not in late3 and r != "notime_s0"]
M("earlyAlphaTwo", rng([load(r, "posterior")["rows"][2]["alpha__posterior_mean"] for r in early], "{:.2f}"))
M("earlyAlphaOne", rng([load(r, "posterior")["rows"][1]["alpha__posterior_mean"] for r in early], "{:.2f}"))
M("nRuns", str(len(ALL)))
w4all = {r: {x["layer"] + 1: x["delta_nats"] for x in load(r, "mech")["windows"] if x["window"] == 4} for r in ALL}
M("longCostRange", rng([max(v.values()) for v in w4all.values()], "{:.3f}"))
M("lastBlockMax", f"{max(v[4] for v in w4all.values()):.3f}")
M("longBlockTwo", str(sum(max(v, key=v.get) == 2 for v in w4all.values())))
rows = []
for r in ALL:
    w = {(x["layer"] + 1, x["window"]): x["delta_nats"] for x in load(r, "mech")["windows"]}
    f3 = lambda v: "0.000" if abs(v) < 5e-4 else f"{v:+.3f}"  # noqa: E731
    rows.append(" & ".join([r.replace("_", r"\_")] + [f3(w[(b, 4)]) for b in (1, 2, 3, 4)]
                           + [f3(w[(b, 16)]) for b in (1, 2, 3, 4)]) + r" \\")
(ROOT / "tables" / "windows.tex").write_text(tex_minus("\n".join(rows) + "\n"))

# ---------------------------------------------------------------- transplant
TP = ["alpha", "log_mu", "log_beta"]


def tp_stats(r):
    rows_ = load(r, "transplant")
    out = {}
    for p in TP:
        best = max((x for x in rows_ if x["param"] == p and x["method"] == "param_map"),
                   key=lambda x: x["pattern_own"] - x["pattern_best_other"])
        rnd = next(x for x in rows_ if x["param"] == p and x["layer"] == best["layer"] and x["method"] == "ctrl_random")
        margin = best["pattern_own"] - best["pattern_best_other"]
        rnd_pref = rnd["pattern_own"] - rnd["pattern_best_other"]
        ok = best["pattern_own"] >= 0.5 and best["slope"] >= 0.3 and margin >= 0.2 and margin >= rnd_pref + 0.2
        out[p] = {"block": best["layer"] + 1, "slope": best["slope"], "own": best["pattern_own"], "other": best["pattern_best_other"],
                  "ratio": best["mean_abs_model_change"] / max(rnd["mean_abs_model_change"], 1e-12),
                  "rnd_own": rnd["pattern_own"], "pass": ok}
    return out


T = {r: tp_stats(r) for r in ALL_EXP}
rows = []
for p in TP:
    rows.append(" & ".join([NAMES[p],
                            rng([T[r][p]["block"] for r in ALL_EXP], "{:d}"),
                            ms([T[r][p]["slope"] for r in ALL_EXP], "{:.2f}"),
                            ms([T[r][p]["ratio"] for r in ALL_EXP], "{:.1f}"),
                            ms([T[r][p]["own"] for r in ALL_EXP], "{:.2f}"),
                            ms([T[r][p]["other"] for r in ALL_EXP], "{:.2f}"),
                            ms([T[r][p]["rnd_own"] for r in ALL_EXP], "{:.2f}"),
                            f"{sum(T[r][p]['pass'] for r in ALL_EXP)}/{len(ALL_EXP)}"]) + r" \\")
(ROOT / "tables" / "transplant.tex").write_text(tex_minus("\n".join(rows) + "\n"))
M("tpSlopeRange", rng([T[r][p]["slope"] for r in ALL_EXP for p in TP], "{:.1f}"))
M("tpRatioRange", rng([T[r][p]["ratio"] for r in ALL_EXP for p in TP], "{:.0f}"))
M("tpOwnRange", rng([T[r][p]["own"] for r in ALL_EXP for p in TP], "{:.2f}"))
M("tpOtherMax", f"{max(T[r][p]['other'] for r in ALL_EXP for p in TP):.2f}")
M("tpPassBeta", str(sum(T[r]["log_beta"]["pass"] for r in ALL_EXP)))
M("tpPassMu", str(sum(T[r]["log_mu"]["pass"] for r in ALL_EXP)))
M("tpPassAlpha", str(sum(T[r]["alpha"]["pass"] for r in ALL_EXP)))
M("tpOwnMeans", rng([float(np.mean([T[r][p]["own"] for r in ALL_EXP])) for p in TP], "{:.2f}"))
M("tpOtherMeans", rng([float(np.mean([T[r][p]["other"] for r in ALL_EXP])) for p in TP], "{:.2f}").replace("--", " to ").replace("-0", "$-$0"))
M("tpAlphaOwnRange", rng([T[r]["alpha"]["own"] for r in ALL_EXP], "{:.3f}"))
M("tpAlphaOtherMax", f"{max(T[r]['alpha']['other'] for r in ALL_EXP):.2f}")
M("nExpRuns", str(len(ALL_EXP)))

# ---------------------------------------------------------------- heads / kernel shape


def shape(r):
    k = [x for x in load(r, "kernel") if max(x["median_r2"], x["median_r2_pow"]) >= 0.3]
    return float(np.median([x["median_r2_pow"] - x["median_r2"] for x in k]))


M("shapeExp", ms([shape(r) for r in EXP], "{:+.2f}"))
M("shapePL", ms([shape(r) for r in PL], "{:+.2f}"))
kmax = {r: max(x["median_r2"] for x in load(r, "kernel")) for r in ALL}
adaptive = {r: max((x["spearman_decay_vs_beta"] for x in load(r, "kernel") if x["median_r2"] >= 0.7), default=None) for r in ALL}
M("expHeadMaxR", rng([kmax[r] for r in EXP + ["nonull_s0"] + PL], "{:.2f}"))
M("notimeHeadMaxR", f"{kmax['notime_s0']:.2f}")
M("nAdaptiveKernel", str(sum(1 for r in ALL if adaptive[r] is not None and adaptive[r] >= 0.5)))

# ---------------------------------------------------------------- probes of true values (earlier rounds)
pr = {r: load(r, "probe") for r in ALL}
M("probeIntensityRange", rng([max(x["log_intensity"] for x in pr[r]) for r in ALL], "{:.2f}"))

# ---------------------------------------------------------------- config
cfg = load("null_s0", "config")
M("nEvents", str(cfg["data"]["n_events"]))
M("nSteps", f"{cfg['train']['steps']:,}".replace(",", "{,}"))
M("batchSize", str(cfg["data"]["batch_size"]))
M("dModel", str(cfg["model"]["d_model"]))
M("nLayers", str(cfg["model"]["n_layers"]))
M("nHeads", str(cfg["model"]["n_heads"]))

# ---------------------------------------------------------------- misspecification (rounds 7-8)
MS = {r: load(r, "r8_misspec_ci") for r in ALL}
mis = {r: next(d for d in MS[r]["data"].values() if d["misspecified"]) for r in ALL}
ins7 = {r: next(d for d in load(r, "r7_misspec")["data"].values() if not d["misspecified"]) for r in ALL}
mis7 = {r: next(d for d in load(r, "r7_misspec")["data"].values() if d["misspecified"]) for r in ALL}
M("msOwnRange", rng([mis[r]["kl_own"] for r in ALL], "{:.4f}"))
M("msTrueRange", rng([mis[r]["kl_true"] for r in ALL], "{:.4f}"))
M("msRatioRange", rng([mis[r]["ratio_own_over_true"] for r in ALL], "{:.2f}"))
M("msRatioCIMax", f"{max(mis[r]['ratio_ci95'][1] for r in ALL):.2f}")
M("msFoldRange", rng([1 / mis[r]["ratio_own_over_true"] for r in ALL], "{:.0f}"))
M("msShareMin", f"{100 * min(mis[r]['share_seqs_closer_to_own'] for r in ALL):.0f}")
M("msInRange", rng([ins7[r]["kl_own"] for r in ALL], "{:.4f}"))
M("msNSeqs", str(mis["null_s0"]["n_seqs"]))
M("msESSMin", f"{min(min(mis[r]['ess_median_true_prior']) for r in ALL):.0f}")
M("msPLonExpModel", ms([mis7[r]["ce_model"] for r in PL], "{:.3f}").split(" ")[0])
M("msPLonExpOwn", ms([mis7[r]["ce_bayes_own"] for r in PL], "{:.3f}").split(" ")[0])
M("msPLonExpTrue", ms([mis7[r]["ce_bayes_true"] for r in PL], "{:.3f}").split(" ")[0])
M("msAlphaOwnCorr", rng([mis7[r]["alpha_corr_decoded_own"] for r in ALL], "{:.2f}"))
M("msAlphaTrueCorr", rng([mis7[r]["alpha_corr_decoded_true"] for r in ALL], "{:.2f}"))
rows = []
for r in ALL:
    d = mis[r]
    rows.append(" & ".join([r.replace("_", r"\_"), MS[r]["own_family"].replace("exp", "exponential").replace("powerlaw", "power law"),
                            f"{d['kl_own']:.4f}", f"{d['kl_true']:.4f}", f"{d['ratio_own_over_true']:.3f}",
                            f"[{d['ratio_ci95'][0]:.3f}, {d['ratio_ci95'][1]:.3f}]", f"{100 * d['share_seqs_closer_to_own']:.0f}\\%"]) + r" \\")
(ROOT / "tables" / "misspec.tex").write_text(tex_minus("\n".join(rows) + "\n"))

# ---------------------------------------------------------------- interchange / DAS (rounds 7-8)
DAS = {r: load(r, "r8_das") for r in ALL_EXP}


def das_best(r, key="das3"):
    return max(DAS[r]["blocks"], key=lambda b: b[key])


M("dasThreeRange", rng([das_best(r)["das3"] for r in ALL_EXP], "{:.2f}"))
M("dasEightRange", rng([das_best(r, "das8")["das8"] for r in ALL_EXP], "{:.2f}"))
M("dasRandRange", rng([das_best(r, "das8")["random8"] for r in ALL_EXP], "{:.2f}"))
M("dasRandThreeRange", rng([das_best(r)["random3"] for r in ALL_EXP], "{:.2f}"))
M("dasDecMax", f"{max(max(b['decoder3'] for b in DAS[r]['blocks']) for r in ALL_EXP):.2f}")
M("dasEncRange", rng([max(b["encoder3"] for b in DAS[r]["blocks"]) for r in ALL_EXP], "{:.2f}"))
M("dasFullRange", rng([das_best(r)["full"] for r in ALL_EXP], "{:.2f}"))
last = {r: next(b for b in DAS[r]["blocks"] if b["block"] == max(x["block"] for x in DAS[r]["blocks"]) and b["mode"] == "query") for r in ALL_EXP}
M("dasLastRange", rng([last[r]["das3"] for r in ALL_EXP], "{:.2f}"))
M("dasOvlDecRange", rng([das_best(r, "das8")["das8_overlap_decoder"] for r in ALL_EXP], "{:.2f}"))
M("dasOvlEncRange", rng([das_best(r, "das8")["das8_overlap_encoder"] for r in ALL_EXP], "{:.2f}"))
M("dasOvlChance", f"{das_best('null_s0', 'das8')['das8_overlap_random_expected']:.2f}")
M("dasBlocks", ", ".join(sorted({str(das_best(r)['block'] + 1) for r in ALL_EXP})))
M("dasTrainTestGap", f"{max(abs(b['das3'] - b['das3_train']) for r in ALL_EXP for b in DAS[r]['blocks']):.2f}")
M("dasNTest", str(DAS["null_s0"]["n_test"]))
IC7 = {r: load(r, "r7_interchange") for r in ALL_EXP}
M("icSevenMax", f"{max(v['transfer_posterior'] for r in ALL_EXP for v in IC7[r]['layers'].values()):.3f}")
rows = []
for r in ALL_EXP:
    for b in DAS[r]["blocks"]:
        rows.append(" & ".join([r.replace("_", r"\_"), str(b["block"] + 1), b["mode"]] +
                               [f"{b[k]:.2f}" for k in ("decoder3", "encoder3", "das3", "das8", "random8", "full")] +
                               [f"{b['das8_overlap_encoder']:.2f} / {b['das8_overlap_decoder']:.2f}"]) + r" \\")
(ROOT / "tables" / "das_full.tex").write_text(tex_minus("\n".join(rows) + "\n"))

# ---------------------------------------------------------------- likelihood pooling (rounds 7-8)
PO = {r: load(r, "r8_pooled") for r in ALL_EXP}
BK = {r: load(r, "r7_bank") for r in ALL}
P3 = ["log_mu", "alpha", "log_beta"]
for key, src in (("suffBank", "bank_mlp"), ("suffPool", "bank_pooled_mlp"), ("suffResid", "residual_best_linear"),
                 ("suffCount", "count_time_mlp")):
    M(key, " / ".join(rng([PO[r]["sufficiency"][p][src] for r in ALL_EXP], "{:.2f}") for p in P3))
M("suffBankPL", " / ".join(rng([BK[r]["sufficiency"][p]["bank_mlp"] for r in PL], "{:.2f}") for p in P3))
pool_heads = {r: [a for a in PO[r]["attention"] if a["tv_uniform"] < 0.25] for r in ALL_EXP}
M("poolTVRange", rng([min(a["tv_uniform"] for a in pool_heads[r]) for r in ALL_EXP], "{:.2f}"))
M("poolTVMax", f"{max(a['tv_uniform'] for r in ALL_EXP for a in pool_heads[r] if a['tv_uniform'] < 0.1):.2f}")
M("poolNHeads", rng([len(pool_heads[r]) for r in ALL_EXP], "{:d}"))
tv_other = [a["tv_uniform"] for r in ALL_EXP for a in PO[r]["attention"] if a["tv_uniform"] >= 0.25 and a["block"] >= 1]
M("poolTVOtherMedian", f"{np.median(tv_other):.2f}")
pool_blocks = {r: sorted({a["block"] + 1 for a in pool_heads[r]}) for r in ALL_EXP}
M("poolBlocks", "; ".join(f"{r.replace('_', '-')}: {', '.join(map(str, b))}" for r, b in pool_blocks.items()))
long_block = {r: max((x for x in load(r, "mech")["windows"] if x["window"] == 4), key=lambda x: x["delta_nats"])["layer"] + 1
              for r in ALL_EXP}
M("poolMatchLong", str(sum(1 for r in ALL_EXP if long_block[r] in pool_blocks[r])))
rep = {r: [x["pooled_r2_mean"] for x in PO[r]["representation"]] for r in ALL_EXP}
M("pooledEmb", rng([rep[r][0] for r in ALL_EXP], "{:.2f}"))
M("pooledAfterPool", rng([rep[r][max(pool_blocks[r])] for r in ALL_EXP], "{:.2f}"))
M("pooledBefore", rng([rep[r][min(pool_blocks[r]) - 1] for r in ALL_EXP], "{:.2f}"))
rec = {r: {x["block"] + 1: x for x in PO[r]["reconstruction"]} for r in ALL_EXP}
later = [x for r in ALL_EXP for b, x in rec[r].items() if b >= 2 and x["recovered_bank"] is not None]
M("reconBankLater", rng([x["recovered_bank"] for x in later], "{:.2f}"))
M("reconPoolLater", rng([x["recovered_bank_pooled"] for x in later], "{:.2f}"))
M("reconBankFirst", rng([BK[r]["reconstruction"][0]["recovered"] for r in ALL if BK[r]["reconstruction"][0].get("recovered") is not None], "{:.2f}"))
M("reconPoolFirst", rng([rec[r][1]["recovered_bank_pooled"] for r in ALL_EXP], "{:.2f}"))
M("nRates", str(len(PO["null_s0"]["rates"])))
M("nOffsets", str(len(PO["null_s0"]["cs"])))

# ---------------------------------------------------------------- FIM-PP audit (exploratory)
F = json.load(open(DATA / "fimpp" / "fimpp.json"))
frows = {r["K"]: r for r in F["rows"]}
M("fimKLOne", f"{frows[1]['kl_bayes_fimpp']:.2f}")
M("fimKLFour", f"{frows[4]['kl_bayes_fimpp']:.2f}")
M("fimKLSixteen", f"{frows[16]['kl_bayes_fimpp']:.2f}")
M("fimBetaCorr", rng([frows[k]["log_beta_corr_est_posterior"] for k in (1, 4, 16)], "{:.2f}"))
M("fimMuCorr", rng([frows[k]["log_mu_corr_est_posterior"] for k in (1, 4, 16)], "{:.2f}"))
M("fimExcessLarge", rng([frows[k]["ce_fimpp"] - frows[k]["ce_oracle"] for k in (64, 256)], "{:.3f}"))

# ---------------------------------------------------------------- round 9: head ablation and DAS controls
AB = {r: load(r, "r9_ablate") for r in ALL_EXP}
TIMED = [r for r in ALL_EXP if r != "notime_s0"]
d_pool = {r: AB[r]["pooling"]["ce"] - AB[r]["baseline"]["ce"] for r in ALL_EXP}
d_rand = {r: [x["ce"] - AB[r]["baseline"]["ce"] for x in AB[r]["random"]] for r in ALL_EXP}
M("ablPoolRange", rng([d_pool[r] for r in TIMED], "{:.3f}"))
M("ablOtherMeanRange", rng([float(np.mean(d_rand[r])) for r in TIMED], "{:.3f}"))
M("ablOtherMaxRange", rng([max(d_rand[r]) for r in TIMED], "{:.3f}"))
M("ablNotime", f"{d_pool['notime_s0']:.2f}")
M("ablNotimeOther", f"{max(d_rand['notime_s0']):.3f}")
M("ablPassA", str(sum(1 for r in ALL_EXP if d_pool[r] >= 3 * max(np.mean(d_rand[r]), 1e-4) and d_pool[r] > max(d_rand[r]))))
M("ablPooledAfter", rng([AB[r]["pooling"]["pooled_r2"] for r in TIMED if r != "null_s2"], "{:.2f}"))
M("ablPooledOther", rng([x["pooled_r2"] for r in TIMED for x in AB[r]["random"]], "{:.2f}"))
M("ablPostAfter", " / ".join(rng([AB[r]["pooling"]["posterior_r2_final"][i] for r in TIMED], "{:.2f}") for i in range(3)))
DC = {r: load(r, "r9_das_controls") for r in ALL_EXP}
best_dc = {r: max(DC[r]["blocks"], key=lambda b: b["right_target_transfer"]) for r in ALL_EXP}
M("dcRightRange", rng([best_dc[r]["right_target_transfer"] for r in ALL_EXP], "{:.2f}"))
M("dcWrongRange", rng([b["wrong_target_transfer"] for r in ALL_EXP for b in DC[r]["blocks"]], "{:.2f}"))
M("dcCrossRange", rng([b["right_subspace_on_wrong_target"] for r in ALL_EXP for b in DC[r]["blocks"]], "{:.2f}"))
M("dcRandRange", rng([b["random_subspace_right_target"] for r in ALL_EXP for b in DC[r]["blocks"]], "{:.2f}"))
b2 = {r: next(b for b in DC[r]["blocks"] if b["block"] == 2) for r in ALL_EXP}
M("dcCoordRange", rng([min(b2[r]["posterior_r2_from_das_coords"]) for r in ALL_EXP if r != "notime_s0"], "{:.2f}"))
M("dcCoordMax", rng([max(b2[r]["posterior_r2_from_das_coords"]) for r in ALL_EXP], "{:.2f}"))
M("dcCoordNotimeBeta", f"{b2['notime_s0']['posterior_r2_from_das_coords'][2]:.2f}")
M("dcRandCoordWorst", rng([min(b2[r]["posterior_r2_from_random_coords"]) for r in ALL_EXP], "{:.2f}"))
M("dcRandCoordRange", rng([x for r in ALL_EXP for x in b2[r]["posterior_r2_from_random_coords"]], "{:.2f}"))
M("dcPassTwo", str(sum(1 for r in ALL_EXP if min(max(DC[r]["blocks"], key=lambda b: b["right_target_transfer"])["posterior_r2_from_das_coords"]) >= 0.8)))
al = [x["overlap"] for r in ALL_EXP for x in DC[r]["alignment"]]
M("dcAlign", f"{np.mean(al):.2f}")
M("dcAlignChance", f"{DC['null_s0']['alignment_random_expected']:.2f}")
rows = []
for r in ALL_EXP:
    for b in DC[r]["blocks"]:
        rows.append(" & ".join([r.replace("_", r"\_"), str(b["block"] + 1), f"{b['right_target_transfer']:.2f}",
                                f"{b['wrong_target_transfer']:.2f}", f"{b['right_subspace_on_wrong_target']:.2f}",
                                f"{b['random_subspace_right_target']:.2f}",
                                " / ".join(f"{x:.2f}" for x in b["posterior_r2_from_das_coords"]),
                                " / ".join(f"{x:.2f}" for x in b["posterior_r2_from_random_coords"])]) + r" \\")
(ROOT / "tables" / "das_controls.tex").write_text(tex_minus("\n".join(rows) + "\n"))
rows = []
for r in ALL_EXP:
    a = AB[r]
    rows.append(" & ".join([r.replace("_", r"\_"), ", ".join(f"{b + 1}.{h + 1}" for b, h in a["heads"]),
                            f"{d_pool[r]:+.3f}", f"{np.mean(d_rand[r]):+.3f}", f"{max(d_rand[r]):+.3f}",
                            f"{a['pooling']['pooled_r2']:.2f}"]) + r" \\")
(ROOT / "tables" / "ablation.tex").write_text(tex_minus("\n".join(rows) + "\n"))

lines = ["% generated by scripts/make_numbers.py -- do not edit by hand"]
for k, v in macros.items():
    lines.append(f"\\newcommand{{\\{k}}}{{{tex_minus(v)}}}")
(ROOT / "numbers.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("assembled-at-long-context detail:", detail)

# ---------------------------------------------------------------- appendix: by branching ratio
BINS = [("a0.0-0.3", "$[0, 0.3)$"), ("a0.3-0.6", "$[0.3, 0.6)$"), ("a0.6-0.9", "$[0.6, 0.9)$"), (NEAR_CRIT, "$[0.90, 0.98]$ (unseen)")]
rows = []
for b, lab in BINS:
    cells = [lab]
    for runs in (EXP, PL):
        g, x = [], []
        for r in runs:
            ev = load(r, "eval")
            lk = late_key(ev["all"])
            g.append(ev[b][f"{lk}_gap_frac"])
            x.append(ev[b][f"{lk}_model"] - ev[b][f"{lk}_oracle"])
        cells += [ms(g), ms(x, "{:.4f}")]
    rows.append(" & ".join(cells) + r" \\")
(ROOT / "tables" / "alpha_bins.tex").write_text(tex_minus("\n".join(rows) + "\n"))
print("alpha bins table written")
