"""Report figures, generated from the result JSONs only (no model runs).

  python scripts/report/make_figures.py                       # everything
  python scripts/report/make_figures.py --only heatmap        # name filter
  python scripts/report/make_figures.py --paper report/final_report_v2/figures

Output: results/figures/main/*.png are the figures used in the final report
(MAIN_FIGURES); results/figures/extra/*.png is everything else. --paper
additionally writes suptitle-less 200-dpi copies of the MAIN figures into DIR
for direct \\includegraphics use (captions live in LaTeX). Figures skip
gracefully when their inputs are missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
CAL = RES / "calibrated"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

METHODS = ("static", "cqr", "aci", "input_tau")
METHOD_LABEL = {"static": "static τ", "cqr": "CQR", "aci": "ACI (online)",
                "input_tau": "input-τ (offline)"}
METHOD_COLOR = {"raw": "#7f8c8d", "static": "#c0392b", "cqr": "#e67e22",
                "aci": "#1f4e79", "input_tau": "#27ae60"}

# Paper-facing display names (the report uses these; raw keys stay in JSONs).
MODEL_LABEL = {"deepar": "DeepAR", "qtransformer": "QT-uni",
               "qtransformer_multi": "QT-multi", "qlstm": "qLSTM",
               "qdlinear": "qDLinear", "lgbm": "LightGBM-q", "qrf": "QRF",
               "lstm": "LSTM", "gru": "GRU", "dlinear": "DLinear",
               "naive_seasonal": "Seasonal-naive", "arima": "ARIMA",
               "sarima": "SARIMA", "naive": "Seasonal-naive", "qt": "QT-uni",
               "qlstm_robust": "qLSTM (robust)",
               "qtransformer_robust": "QT-uni (robust)"}


def _fault_label(f: str) -> str:
    return "FGSM" if f == "fgsm" else f.replace("_", " ")


def _load_cal(method: str, model: str) -> dict | None:
    p = CAL / method / f"{model}.json"
    return json.loads(p.read_text()) if p.exists() else None


# The figures used in the final report (v2 outline, F1-F7) go to figures/main;
# everything else goes to figures/extra. Membership here is the single switch.
MAIN_FIGURES = {
    "input_value.png",                      # F1  §3.2 covariate strength + leakage
    "covariate_importance_independent.png", # F2  §3.2 (optional)
    "error_breakdowns_full.png",            # F3  §5.2 horizon · year · temp · season
    "fault_catalog_heatmap.png",            # F4  §7.3 degradation taxonomy
    "calibration_picp_vs_intensity.png",    # F5  §8.1 interval-side repair
    "robust_plus_cal.png",                  # F6  §8.3 four-corner recipe
    "detect_then_adapt.png",                # F7  §8.4 gated repair
}


# --paper <dir>: additionally write a copy of each MAIN figure WITHOUT the
# narrative suptitle into <dir> for direct inclusion in the LNCS report
# (captions live in LaTeX there). Panel titles are kept.
PAPER_DIR: Path | None = None


def _save(fig, name: str) -> None:
    out_dir = RES / "figures" / ("main" if name in MAIN_FIGURES else "extra")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.savefig(out)
    print(f"  wrote {out}")
    if PAPER_DIR is not None and name in MAIN_FIGURES:
        if fig._suptitle is not None:
            fig._suptitle.remove()
        # tight_layout breaks figures with fig.colorbar(ax=...) axes;
        # constrained-layout figures manage themselves.
        if not fig.get_constrained_layout():
            try:
                fig.tight_layout()
            except Exception:
                pass
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_DIR / name, dpi=200)
        print(f"  wrote {PAPER_DIR / name} (paper copy)")
    plt.close(fig)


# --------------------------------------------------- calibration story
def fig_picp_vs_intensity() -> None:
    """The money plot: coverage vs corruption intensity, per repair method."""
    models = ("deepar", "qtransformer")
    kinds = ("level_shift", "fgsm")
    fig, axes = plt.subplots(len(kinds), len(models), figsize=(7.4, 5.2),
                             sharex=True, sharey=True)
    xs = [0.0, 1.0, 2.0, 4.0]
    for i, kind in enumerate(kinds):
        for j, model in enumerate(models):
            ax = axes[i, j]
            base = _load_cal("static", model)
            if base is None:
                continue

            def series(method=None):
                ys = []
                src = _load_cal(method or "static", model)
                for x in xs:
                    s = src["settings"]["clean" if x == 0 else f"{kind}_{x:.1f}"]
                    ys.append(s["before" if method is None else "after"]["picp"])
                return ys

            ax.plot(xs, series(None), "o-", color=METHOD_COLOR["raw"], label="uncalibrated")
            for m in METHODS:
                if _load_cal(m, model):
                    ax.plot(xs, series(m), "o-", color=METHOD_COLOR[m],
                            label=METHOD_LABEL[m])
            ax.axhline(0.9, color="k", lw=0.7, ls=":")
            ax.set_title(f"{MODEL_LABEL.get(model, model)} — {_fault_label(kind)}")
            ax.set_ylim(0, 1.0)
            if i == len(kinds) - 1:
                ax.set_xlabel("intensity (× local σ)")
            if j == 0:
                ax.set_ylabel("PICP (target 0.90)")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Coverage vs corruption intensity: static repair collapses, adaptive holds", y=1.01)
    _save(fig, "calibration_picp_vs_intensity.png")


def fig_mis_ls4() -> None:
    models = ("deepar", "qtransformer")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    labels = ["uncalibrated"] + [METHOD_LABEL[m] for m in METHODS]
    width = 0.38
    for k, model in enumerate(models):
        vals = []
        base = _load_cal("static", model)
        if base is None:
            return
        vals.append(base["settings"]["level_shift_4.0"]["before"]["mis"])
        for m in METHODS:
            vals.append(_load_cal(m, model)["settings"]["level_shift_4.0"]["after"]["mis"])
        x = np.arange(len(vals)) + (k - 0.5) * width
        ax.bar(x, vals, width, label=model,
               color="#1f4e79" if k == 0 else "#27ae60")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("MIS (level shift 4×)")
    ax.set_title("Interval score: adaptive repair halves MIS")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "calibration_mis_ls4.png")


def fig_significance() -> None:
    p = RES / "base" / "significance.json"
    if not p.exists():
        return
    sig = json.loads(p.read_text())
    pairs = [("lstm__vs__qtransformer", "LSTM − QT"),
             ("ensemble_lstm_qt__vs__lstm", "Ensemble − LSTM"),
             ("ensemble_lstm_qt__vs__qtransformer", "Ensemble − QT")]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    for i, (key, label) in enumerate(pairs):
        if key not in sig["pairs"]:
            continue
        bs = sig["pairs"][key]["bootstrap"]
        lo, hi = bs["ci95"]
        ax.errorbar(bs["delta_rmse"], i, xerr=[[bs["delta_rmse"] - lo], [hi - bs["delta_rmse"]]],
                    fmt="o", color="#1f4e79", capsize=3)
        ax.text(hi + 0.004, i, f"DM p={sig['pairs'][key]['dm']['p_value']:.3f}",
                va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([l for _, l in pairs])
    ax.set_xlabel("ΔRMSE (negative = left model better), 95% bootstrap CI")
    ax.set_title("Significance: LSTM≈QT tie; ensemble gain is real")
    _save(fig, "significance_headline.png")


# -------------------------------------------------------- roster story
FAMILY = {  # model -> (family, deterministic?)
    "naive_seasonal": ("classical", True), "arima": ("classical", True),
    "sarima": ("classical", True), "lstm": ("recurrent", True),
    "gru": ("recurrent", True), "dlinear": ("linear", True),
    "lgbm": ("tree", False), "qrf": ("tree", False),
    "qlstm": ("recurrent", False), "qdlinear": ("linear", False),
    "deepar": ("AR-likelihood", False), "qtransformer": ("transformer", False),
    "qtransformer_multi": ("transformer", False),
}
FAM_COLOR = {"classical": "#7f8c8d", "recurrent": "#1f4e79", "linear": "#8e44ad",
             "tree": "#27ae60", "AR-likelihood": "#c0392b", "transformer": "#e67e22"}


def _base_json(model: str) -> dict | None:
    p = RES / "base" / f"{model}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_roster() -> None:
    rows = []
    for m in FAMILY:
        d = _base_json(m)
        if d is None:
            continue
        rmse = d.get("rmse")
        rows.append((m, rmse, d.get("crps"), FAMILY[m][0]))
    rows.sort(key=lambda r: r[1])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    names = [MODEL_LABEL.get(r[0], r[0]) for r in rows]
    axes[0].barh(names, [r[1] for r in rows],
                 color=[FAM_COLOR[r[3]] for r in rows])
    axes[0].set_xlabel("RMSE °C (point / median)")
    axes[0].set_title("Roster — point accuracy")
    axes[0].invert_yaxis()
    prob = [r for r in rows if r[2] is not None]
    prob.sort(key=lambda r: r[2])
    axes[1].barh([MODEL_LABEL.get(r[0], r[0]) for r in prob], [r[2] for r in prob],
                 color=[FAM_COLOR[r[3]] for r in prob])
    axes[1].set_xlabel("CRPS")
    axes[1].set_title("Probabilistic — CRPS")
    axes[1].invert_yaxis()
    fig.suptitle("13-model roster (colour = family)", y=1.02)
    _save(fig, "roster_overview.png")


def fig_paired_families() -> None:
    pairs = [("lstm", "qlstm", "recurrent\n(p=0.024 *)"),
             ("dlinear", "qdlinear", "linear\n(p=0.081)"),
             ("lgbm", "qrf", "tree (point vs QRF med)")]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    width = 0.36
    for i, (det, prob, label) in enumerate(pairs):
        d1, d2 = _base_json(det), _base_json(prob)
        if d1 is None or d2 is None:
            continue
        det_rmse = d1.get("point_rmse", d1.get("rmse"))
        prob_rmse = d2.get("rmse")
        ax.bar(i - width / 2, det_rmse, width, color="#7f8c8d",
               label="deterministic" if i == 0 else None)
        ax.bar(i + width / 2, prob_rmse, width, color="#1f4e79",
               label="probabilistic (median)" if i == 0 else None)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p[2] for p in pairs])
    ax.set_ylabel("RMSE °C")
    ax.set_ylim(2.0, None)
    ax.set_title("Paired families: the pinball head costs no point accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "paired_families.png")


def fig_raw_robustness() -> None:
    models = ("deepar", "qtransformer", "qtransformer_multi", "qlstm",
              "qdlinear", "lgbm", "qrf")
    vals, names, colors = [], [], []
    for m in models:
        d = _load_cal("static", m)
        if d is None:
            continue
        vals.append(d["settings"]["level_shift_4.0"]["before"]["picp"])
        names.append(m)
        colors.append(FAM_COLOR[FAMILY[m][0]])
    order = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar([MODEL_LABEL.get(names[i], names[i]) for i in order],
           [vals[i] for i in order],
           color=[colors[i] for i in order])
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_ylabel("uncalibrated PICP (level shift 4×)")
    ax.set_title("Raw robustness: trees do not 'follow' the shift")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "raw_robustness_ls4.png")


def fig_permutation_importance() -> None:
    p = RES / "base" / "feature_importance.json"
    if not p.exists():
        return
    fi = json.loads(p.read_text())
    rows = sorted(fi["splits"]["test"]["permuted"].items(),
                  key=lambda kv: kv[1]["delta_crps"])
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh([k for k, _ in rows], [v["delta_crps"] for _, v in rows], color="#1f4e79")
    ax.set_xlabel("ΔCRPS (channel shuffled across windows)")
    ax.set_title("Permutation importance (QT-multi, test)")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, "permutation_importance.png")


def fig_covariate_full() -> None:
    p = RES / "base" / "covariate_importance_full.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    rows = sorted(d["splits"]["test"]["permuted"].items(),
                  key=lambda kv: kv[1]["delta_crps"])
    color = {True: "#c0392b", False: "#1f4e79"}  # temp-derived red, independent blue
    cols, vals, cs = [], [], []
    for n, v in rows:
        cols.append(n)
        vals.append(v["delta_crps"])
        if v["is_calendar"]:
            cs.append("#7f8c8d")
        elif v["is_temp_derived"]:
            cs.append("#c0392b")
        else:
            cs.append("#27ae60")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(cols, vals, color=cs)
    ax.set_xlabel("ΔCRPS (channel shuffled across windows)")
    ax.set_title("Full 13-covariate importance (QT, test)\n"
                 "red = temperature-derived (partial leakage) · green = independent sensor · grey = calendar")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, "covariate_importance_full.png")


def fig_covariate_independent() -> None:
    p = RES / "base" / "covariate_importance_independent.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    # report version: drop only the unmeasurable hour channels (stride-24
    # artefact, exactly 0); doy_sin stays (it pairs with doy_cos to encode
    # day-of-year, marginal but real at +0.04)
    SKIP = {"hour_sin", "hour_cos"}
    rows = sorted((kv for kv in d["splits"]["test"]["permuted"].items()
                   if kv[0] not in SKIP),
                  key=lambda kv: kv[1]["delta_crps"])
    names, vals, cs = [], [], []
    for n, v in rows:
        names.append(n)
        vals.append(v["delta_crps"])
        if v.get("is_target"):
            cs.append("#c0392b")       # T — dominant
        elif v["is_calendar"]:
            cs.append("#7f8c8d")
        else:
            cs.append("#27ae60")       # independent sensor
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(names, vals, color=cs)
    for i, v in enumerate(vals):
        if v > 0.02:
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_xlabel("ΔCRPS (channel shuffled across windows)")
    # narrative title as suptitle so the --paper copy drops it (caption in LaTeX)
    fig.suptitle("Independent covariate importance (leakage-free)\n"
                 "red = target T (dominant) · green = independent sensor · grey = calendar",
                 y=1.06)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, "covariate_importance_independent.png")


def fig_input_value() -> None:
    p = RES / "base" / "exogenous_only.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    e, r = d["exogenous_only"], d["references"]
    pi = RES / "base" / "exogenous_only_independent.json"
    indep = json.loads(pi.read_text())["exogenous_only"]["rmse"] if pi.exists() else None
    bars = [
        ("seasonal-naive\n(reference)", r.get("naive_seasonal", {}).get("rmse"), "#7f8c8d"),
        ("independent\nsensors only", indep, "#c0392b"),
        ("exogenous +\nT-proxies (leaky)", e["rmse"], "#e67e22"),
        ("target T\nonly", r.get("target_only", {}).get("rmse"), "#1f4e79"),
        ("T + all 13\ncovariates", r.get("full_T_plus_cov", {}).get("rmse"), "#27ae60"),
    ]
    bars = [b for b in bars if b[1] is not None]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.03, f"{b[1]:.2f}", ha="center", fontsize=8)
    ax.axhline(r.get("naive_seasonal", {}).get("rmse", 3.21), color="k", lw=0.7, ls=":")
    ax.set_ylabel("test RMSE °C")
    ax.set_ylim(2.0, None)
    # narrative title as suptitle so the --paper copy drops it (caption in LaTeX)
    fig.suptitle("Temperature information is indispensable\n"
                 "without T-proxies (red) the model falls below naive; "
                 "the saturation-vapor-pressure leak brings T back", y=1.08)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "input_value.png")


def fig_error_breakdowns() -> None:
    """Error anatomy for Sect. 5.2: horizon, year, temperature, season."""
    p = RES / "base" / "error_tables_full.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    cvp = RES / "base" / "cv_forward_chaining.json"
    cv = json.loads(cvp.read_text()) if cvp.exists() else None
    # roster models only: robust-trained variants belong to the repair story
    models = sorted((m for m in d["per_horizon"] if m in FAMILY),
                    key=lambda m: np.mean(d["per_horizon"][m]))
    best, worst = models[0], models[-1]
    n = 4 if cv else 3
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4),
                             layout="constrained")

    # panel 1: per-horizon curves (highlight best + worst, grey the rest)
    ax = axes[0]
    for m in models:
        ph = d["per_horizon"][m]
        hl = m in (best, worst)
        ax.plot(range(1, len(ph) + 1), ph, lw=1.6 if hl else 0.7,
                color="#27ae60" if m == best else ("#c0392b" if m == worst else "#bbb"),
                label=MODEL_LABEL.get(m, m) if hl else None,
                zorder=3 if hl else 1)
    ax.set_xlabel("forecast step (hours)")
    ax.set_ylabel("RMSE °C")
    ax.set_title("Per-horizon (full roster)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)

    # panel 2: year-based forward-chaining CV
    if cv:
        axc = axes[1]
        for m, block in cv["models"].items():
            years = sorted(block["folds"])
            axc.plot([int(y) for y in years],
                     [block["folds"][y]["rmse"] for y in years],
                     "o-", ms=3, label=MODEL_LABEL.get(m, m))
        axc.set_xticks([int(y) for y in years])
        axc.set_xlabel("test year (expanding train)")
        axc.set_title("By year (forward-chaining CV)")
        axc.legend(frameon=False, fontsize=6.5)
        axc.grid(alpha=0.25)

    # panels 3+4: by-temperature and by-season heatmaps (shared model axis)
    names = [MODEL_LABEL.get(m, m) for m in models]
    for ax_h, key, order, cmap in ((axes[-2], "by_temperature", d["temp_order"], "YlOrRd"),
                                   (axes[-1], "by_season", d["season_order"], "YlGnBu")):
        mat = np.array([[d[key][m].get(c, np.nan) for c in order] for m in models])
        im = ax_h.imshow(mat, aspect="auto", cmap=cmap)
        ax_h.set_xticks(range(len(order)))
        ax_h.set_xticklabels(order, fontsize=7)
        ax_h.set_yticks(range(len(models)))
        ax_h.set_yticklabels(names if ax_h is axes[-2] else [""] * len(models),
                             fontsize=6)
        ax_h.set_title("RMSE by temperature range" if key == "by_temperature"
                       else "RMSE by season")
        fig.colorbar(im, ax=ax_h, fraction=0.046)

    fig.suptitle("Full-roster error anatomy (horizon · year · temperature · season)")
    _save(fig, "error_breakdowns_full.png")


def fig_natural_extremes() -> None:
    p = RES / "base" / "natural_extremes.json"
    if not p.exists():
        return
    ne = json.loads(p.read_text())
    names, full_w, drop_w, picps = [], [], [], []
    for m, block in sorted(ne["models"].items()):
        names.append(m)
        full_w.append(block["input_tau"]["full"]["mpiw"])
        drop_w.append(block["input_tau"]["sharp_drop"]["mpiw"])
        picps.append(block["input_tau"]["sharp_drop"]["picp"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - 0.19, full_w, 0.38, label="full test", color="#7f8c8d")
    ax.bar(x + 0.19, drop_w, 0.38, label="cold-front slice", color="#1f4e79")
    for i, pi in enumerate(picps):
        ax.text(i + 0.19, drop_w[i] + 0.15, f"{pi:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in names], rotation=20)
    ax.set_ylabel("MPIW °C (input-conditional τ)")
    ax.set_title("Low false-alarm cost on natural extremes (label = slice PICP)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "natural_extremes_falsealarm.png")


# -------------------------------------------------- optimization story
def fig_hpo() -> None:
    p = RES / "base" / "hpo.json"
    if not p.exists():
        return
    hpo = json.loads(p.read_text())
    models = list(hpo["models"])
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    width = 0.36
    for i, m in enumerate(models):
        b = hpo["models"][m]
        metric = "test_crps" if b.get("test_crps_default") is not None else "test_rmse"
        ax.bar(i - width / 2, b[f"{metric}_default"], width, color="#7f8c8d",
               label="default" if i == 0 else None)
        ax.bar(i + width / 2, b[f"{metric}_best"], width, color="#1f4e79",
               label="HPO" if i == 0 else None)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("test CRPS / RMSE")
    ax.set_title("HPO: default vs optimised (selection on validation only)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "hpo_default_vs_best.png")


def fig_multiseed() -> None:
    p = RES / "base" / "multiseed.json"
    if not p.exists():
        return
    ms = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    names = list(ms["models"])
    for i, m in enumerate(names):
        vals = [r["rmse"] for r in ms["models"][m]["runs"]]
        ax.errorbar(i, np.mean(vals), yerr=np.std(vals), fmt="o", capsize=4,
                    color="#1f4e79")
        ax.scatter([i] * len(vals), vals, s=10, color="#7f8c8d", zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in names], rotation=15)
    ax.set_ylabel("RMSE °C (3 seeds)")
    ax.set_title("Seed sensitivity (mean ± std; dots = individual seeds)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "multiseed_rmse.png")


def fig_cv() -> None:
    p = RES / "base" / "cv_forward_chaining.json"
    if not p.exists():
        return
    cv = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for m, block in cv["models"].items():
        years = sorted(block["folds"])
        ax.plot([int(y) for y in years], [block["folds"][y]["rmse"] for y in years],
                "o-", label=MODEL_LABEL.get(m, m))
    ax.set_xlabel("test year (expanding train)")
    ax.set_ylabel("RMSE °C")
    ax.set_title("Year-based forward-chaining CV (fold variance)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "cv_fold_variance.png")


def fig_robust_training() -> None:
    p = RES / "base" / "qlstm_robust_compare.json"
    if not p.exists():
        return
    rc = json.loads(p.read_text())
    settings = rc["settings_order"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(settings))
    ax.bar(x - 0.19, [rc["normal"][s]["picp"] for s in settings], 0.38,
           label="normal qLSTM", color="#7f8c8d")
    ax.bar(x + 0.19, [rc["robust"][s]["picp"] for s in settings], 0.38,
           label="robust-trained qLSTM", color="#27ae60")
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in settings], fontsize=7)
    ax.set_ylabel("uncalibrated PICP")
    ax.set_title("Robust (anomaly-augmented) training: normal vs robust-trained")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "robust_training_picp.png")


def fig_horizon() -> None:
    p = RES / "base" / "horizon_ablation.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    hs = sorted(d["results"], key=lambda h: int(h[:-1]))
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = [int(h[:-1]) for h in hs]
    ax.plot(x, [d["results"][h]["rmse"] for h in hs], "o-", color="#1f4e79", label="RMSE")
    ax.plot(x, [d["results"][h]["crps"] for h in hs], "s-", color="#e67e22", label="CRPS")
    for h in hs:
        ax.annotate(f"{d['results'][h]['rmse']:.2f}", (int(h[:-1]), d["results"][h]["rmse"]),
                    textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.axvline(24, color="grey", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("error")
    ax.set_title("Horizon ablation: error grows with horizon (24h choice)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "horizon_ablation.png")


def fig_extreme_quantiles() -> None:
    p = RES / "base" / "qt_extreme_quantiles.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    blocks = ["full", "cold_decile", "hot_decile"]
    levels = ["90pct", "95pct", "98pct"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(levels))
    w = 0.26
    colors = {"full": "#1f4e79", "cold_decile": "#2980b9", "hot_decile": "#c0392b"}
    for k, b in enumerate(blocks):
        if b not in d:
            continue
        picps = [d[b][l]["picp"] for l in levels]
        ax.bar(x + (k - 1) * w, picps, w, color=colors[b], label=b.replace("_decile", ""))
    for l, lev in enumerate(levels):
        ax.axhline(float(lev[:2]) / 100, color="k", lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["90%", "95%", "98%"])
    ax.set_ylabel("PICP")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Extreme quantiles: 90/95/98% intervals (full + extreme deciles)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "extreme_quantiles.png")


def fig_ensemble_intervals() -> None:
    p = RES / "base" / "ensemble_intervals.json"
    if not p.exists():
        return
    ei = json.loads(p.read_text())
    rows = [(k, v["pinball"], v["picp"], v["mis"]) for k, v in ei["candidates"].items()]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    names = [r[0] for r in rows]
    colors = ["#c0392b" if "ensemble" in n else "#1f4e79" for n in names]
    ax.barh(names, [r[1] for r in rows], color=colors)
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.002, i, f"PICP {r[2]:.3f}", va="center", fontsize=7)
    ax.set_xlabel("pinball ≈ CRPS/2 (clean test) — red = ensemble")
    ax.set_title("Interval combination: ensemble cannot beat the strongest member")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, "ensemble_intervals.png")


# ------------------------------------------------------- anomaly story
# Catalog order follows the mechanism grouping of the report: additive,
# structural, temporal, adversarial (separator lines between groups).
V2_FAULTS = ("point_spike", "contextual_outlier", "noise_burst",
             "level_shift", "drift", "flatline",
             "gap_imputation", "clock_skew",
             "fgsm")
FAULT_GROUP_SEPS = (2.5, 5.5, 7.5)  # after additive / structural / temporal

HEATMAP_MODELS = (("qtransformer_multi", "QT-multi"),
                  ("lgbm", "LightGBM-quantile (tree)"))


def fig_fault_heatmap() -> None:
    """Fault catalog x intensity -> raw PICP, headline model vs tree contrast.

    Two panels: the multivariate quantile Transformer (gradient model, worst
    degradation) next to LightGBM-quantile (tree, saturating splits, mildest
    degradation). White-box FGSM is undefined for the gradient-free tree, so
    that row renders as hatched n/a cells.
    """
    p = RES / "base" / "report_tables.json"
    if not p.exists():
        return
    rob = json.loads(p.read_text())["robustness_matrix"]
    panels = [(m, label) for m, label in HEATMAP_MODELS if m in rob]
    if not panels:
        return
    intens = (1.0, 2.0, 4.0)
    faults = [f for f in V2_FAULTS
              if any(f"{f}_1.0" in rob[m] for m, _ in panels)]

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    im = None
    for ax, (model, label) in zip(axes, panels):
        block = rob[model]
        mat = np.full((len(faults), len(intens)), np.nan)
        for i, f in enumerate(faults):
            for j, it in enumerate(intens):
                s = block.get(f"{f}_{it:.1f}")
                if s:
                    mat[i, j] = s["picp"]
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap,
                       vmin=0.0, vmax=0.95)
        ax.set_xticks(range(len(intens)))
        ax.set_xticklabels([f"{i:g}×" for i in intens])
        ax.set_yticks(range(len(faults)))
        ax.set_yticklabels([_fault_label(f) for f in faults])
        for i in range(len(faults)):
            for j in range(len(intens)):
                if np.isnan(mat[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=7, color="#555")
                else:
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=7)
        for y in FAULT_GROUP_SEPS:
            if y < len(faults) - 0.5:
                ax.axhline(y, color="k", lw=1.0)
        ax.set_title(label, fontsize=9)
    fig.suptitle("Fault catalog × intensity — raw PICP")
    fig.colorbar(im, ax=list(axes), fraction=0.046 / len(panels),
                 label="PICP (target 0.90)")
    _save(fig, "fault_catalog_heatmap.png")


def fig_robust_generalize() -> None:
    p = RES / "base" / "robust_generalize.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    settings = d["settings"]
    models = list(d["models"])
    x = np.arange(len(settings))
    w = 0.8 / (2 * len(models))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    colors = {"qlstm": "#1f4e79", "qtransformer": "#e67e22", "deepar": "#c0392b"}
    for k, m in enumerate(models):
        base = colors.get(m, "#555")
        normal = [d["models"][m]["normal"][s]["picp"] for s in settings]
        robust = [d["models"][m]["robust"][s]["picp"] for s in settings]
        off = (k - len(models) / 2) * 2 * w + w
        ax.bar(x + off - w / 2, normal, w, color=base, alpha=0.4,
               label=f"{MODEL_LABEL.get(m, m)} normal" if k == 0 else None)
        ax.bar(x + off + w / 2, robust, w, color=base,
               label=f"{MODEL_LABEL.get(m, m)} robust" if k == 0 else None)
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in settings], fontsize=7)
    ax.set_ylabel("raw PICP")
    ax.set_title("Robust training generalises across 3 architectures (light=normal, dark=robust)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "robust_generalize.png")


def fig_resolution() -> None:
    p = RES / "base" / "ablation_10min.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    a, h = d["10min_hourly_equiv"], d["hourly_reference_qlstm"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    labels = ["hourly\nqLSTM", "10-min\n(hourly-equivalent)"]
    rmses = [h["rmse"], a["rmse"]]
    # multiseed qLSTM std for the reference band
    ms = RES / "base" / "multiseed.json"
    std = json.loads(ms.read_text())["models"]["qlstm"]["rmse_std"] if ms.exists() else 0.013
    ax.bar(labels, rmses, color=["#1f4e79", "#c0392b"])
    ax.errorbar([0], [h["rmse"]], yerr=[std], fmt="none", ecolor="k", capsize=5)
    for i, v in enumerate(rmses):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("test RMSE °C")
    ax.set_ylim(2.3, 2.45)
    ax.set_title(f"10-min resolution does not help\n(diff +{a['rmse']-h['rmse']:.3f}, within seed std ±{std:.3f})")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "resolution_ablation.png")


def fig_robust_plus_cal() -> None:
    p = RES / "base" / "robust_plus_cal.json"
    if not p.exists():
        return
    rc = json.loads(p.read_text())
    settings = [s for s in ("clean", "level_shift_1.0", "level_shift_2.0",
                            "level_shift_4.0", "fgsm_4.0") if s in rc["settings"]]
    corners = [("normal_raw", "#7f8c8d", "normal raw"),
               ("normal_aci", "#1f4e79", "normal+ACI"),
               ("robust_raw", "#e67e22", "robust raw"),
               ("robust_aci", "#27ae60", "robust+ACI")]
    fams = rc.get("families", {})
    # stacked layout: the report places this next to another figure, so the
    # two sub-panels go on top of each other instead of side by side.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.2, 5.6))

    # top: qlstm intensity sweep (the canonical four corners)
    x = np.arange(len(settings))
    w = 0.2
    for k, (key, color, label) in enumerate(corners):
        ax.bar(x + (k - 1.5) * w, [rc["settings"][s][key]["picp"] for s in settings],
               w, color=color, label=label)
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("fgsm", "FGSM").replace("_", "\n")
                        for s in settings], fontsize=7)
    ax.set_ylabel("PICP")
    ax.set_ylim(0, 1.18)
    ax.set_title("qLSTM: four corners across intensity", fontsize=9)
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # bottom: robust+ACI at level shift 4x across the whole roster
    if fams:
        order = [f for f in ("qlstm", "qtransformer", "qdlinear", "deepar",
                             "lgbm", "qrf") if f in fams]
        ls4 = {f: fams[f].get("level_shift_4.0") for f in order}
        order = [f for f in order if ls4[f]]
        xr = np.arange(len(order))
        for k, (key, color, label) in enumerate(corners):
            ax2.bar(xr + (k - 1.5) * w,
                    [ls4[f][key]["picp"] for f in order], w, color=color)
        ax2.axhline(0.9, color="k", lw=0.7, ls=":")
        ax2.set_xticks(xr)
        ax2.set_xticklabels([MODEL_LABEL.get(f, f) for f in order],
                            rotation=15, fontsize=7)
        ax2.set_ylabel("PICP")
        ax2.set_title("Level shift 4× across the roster", fontsize=9)
        ax2.grid(axis="y", alpha=0.25)
    fig.suptitle("Model-side and interval-side repairs compose", y=1.0)
    fig.tight_layout()
    _save(fig, "robust_plus_cal.png")


def fig_leaderboard() -> None:
    p = RES / "base" / "report_tables.json"
    if not p.exists():
        return
    lb = json.loads(p.read_text())["clean_leaderboard"]
    rows = sorted(lb.items(), key=lambda kv: kv[1]["rmse"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    names = [MODEL_LABEL.get(r[0], r[0]) for r in rows]
    ax.barh(names, [r[1]["rmse"] for r in rows], color="#1f4e79")
    for i, (_, r) in enumerate(rows):
        if "picp" in r:
            ax.text(r["rmse"] + 0.02, i, f"PICP {r['picp']:.2f}", va="center", fontsize=6.5)
    ax.set_xlabel("clean-test RMSE °C (point / median)")
    ax.set_title("Full-roster clean-test ranking")
    ax.invert_yaxis()
    _save(fig, "final_leaderboard.png")


# ----------------------------------------------------- detection story
def fig_detect_adapt() -> None:
    """Detect-then-adapt: 4-regime repair comparison + detection ladder."""
    det_path = RES / "base" / "detect_adapt_detection.json"
    if not det_path.exists():
        return
    det = json.loads(det_path.read_text())["models"].get("qlstm")
    regimes = ("static", "aci", "input_tau", "detect_adapt")
    labels = ("static τ", "ACI (online)", "input-τ", "detect-then-adapt")
    cals = {r: _load_cal(r, "qlstm") for r in regimes}
    if det is None or any(v is None for v in cals.values()):
        return

    faults = ("level_shift_4.0", "drift_4.0", "fgsm_4.0", "flatline_4.0",
              "clock_skew_4.0")
    short = [("FGSM" if f.startswith("fgsm") else
              f.rsplit("_", 1)[0].replace("_", "\n")) for f in faults]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))

    width = 0.16
    xs = np.arange(len(faults))
    colors = ("#9aa5b1", "#1f4e79", "#7aa6c2", "#c0504d")
    raw = [cals["static"]["settings"][f]["before"]["picp"] for f in faults]
    ax1.bar(xs - 2 * width, raw, width, label="raw", color="#d9d9d9")
    for i, (r, lab, col) in enumerate(zip(regimes, labels, colors)):
        vals = [cals[r]["settings"][f]["after"]["picp"] for f in faults]
        ax1.bar(xs + (i - 1) * width, vals, width, label=lab, color=col)
    ax1.axhline(0.9, color="k", lw=0.8, ls="--")
    ax1.set_xticks(xs, short, fontsize=7)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("PICP (target 0.90)")
    ax1.set_title("Repair per regime (qLSTM, 4× intensity)", fontsize=9)
    ax1.legend(fontsize=6.5, ncols=3, loc="upper center", frameon=False)
    clean_mis = {r: cals[r]["settings"]["clean"]["after"]["mis"] for r in regimes}
    raw_mis = cals["static"]["settings"]["clean"]["before"]["mis"]
    ax1.text(0.5, -0.24,
             f"clean-data width cost (MIS): raw {raw_mis:.1f} | " +
             " | ".join(f"{l.split()[0]} {clean_mis[r]:.1f}"
                        for r, l in zip(regimes, labels)),
             transform=ax1.transAxes, ha="center", fontsize=6.5)

    ps = det["per_setting"]
    kinds = sorted({k.rsplit("_", 1)[0] for k in ps})
    intens = ("1.0", "2.0", "4.0")
    for kind in kinds:
        aucs = [ps.get(f"{kind}_{i}", {}).get("auc") for i in intens]
        if any(a is None for a in aucs):
            continue
        ax2.plot([1, 2, 4], aucs, marker="o", ms=3, lw=1.2,
                 label=_fault_label(kind))
    ax2.axhline(0.5, color="k", lw=0.8, ls=":")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 2, 4], ["1×", "2×", "4×"])
    ax2.set_xlabel("fault intensity")
    ax2.set_ylabel("detection AUC vs clean test")
    ax2.set_title("Detectability scales with severity", fontsize=9)
    ax2.legend(fontsize=6, ncols=2)
    fig.suptitle("Detect-then-adapt: gated repair + detection quality",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "detect_then_adapt.png")


FIGURES = (
    # calibration story
    fig_picp_vs_intensity, fig_mis_ls4, fig_significance,
    # roster story
    fig_roster, fig_paired_families, fig_raw_robustness,
    fig_permutation_importance, fig_covariate_full, fig_covariate_independent,
    fig_input_value, fig_error_breakdowns, fig_natural_extremes,
    # optimization story
    fig_hpo, fig_multiseed, fig_cv, fig_robust_training,
    fig_horizon, fig_extreme_quantiles, fig_ensemble_intervals,
    # anomaly story
    fig_fault_heatmap, fig_resolution, fig_robust_generalize,
    fig_robust_plus_cal, fig_leaderboard,
    # detection story
    fig_detect_adapt,
)


def main() -> None:
    global PAPER_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, metavar="SUBSTR",
                    help="regenerate only figures whose function name contains SUBSTR")
    ap.add_argument("--paper", default=None, metavar="DIR",
                    help="also write suptitle-less copies of MAIN figures into DIR")
    args = ap.parse_args()
    if args.paper:
        PAPER_DIR = Path(args.paper)
    for fn in FIGURES:
        if args.only and args.only not in fn.__name__:
            continue
        fn()


if __name__ == "__main__":
    main()
