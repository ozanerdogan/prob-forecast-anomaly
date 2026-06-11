"""Phase-report figures, generated from the result JSONs only (no model runs).

  python scripts/make_phase_figures.py --phase 1     # calibration story
  python scripts/make_phase_figures.py --phase 2     # roster story
  python scripts/make_phase_figures.py --phase 3     # optimization story
  python scripts/make_phase_figures.py --phase all

Output: results/figures/phase<N>/*.png. Older progress-report figures live in
results/figures/old/. Figures skip gracefully when their inputs are missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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
METHOD_LABEL = {"static": "statik τ", "cqr": "CQR", "aci": "ACI (online)",
                "input_tau": "girdi-koşullu τ (offline)"}
METHOD_COLOR = {"raw": "#7f8c8d", "static": "#c0392b", "cqr": "#e67e22",
                "aci": "#1f4e79", "input_tau": "#27ae60"}


def _load_cal(method: str, model: str) -> dict | None:
    p = CAL / method / f"{model}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, phase: int, name: str) -> None:
    out_dir = RES / "figures" / f"phase{phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ phase 1
def fig_p1_picp_vs_intensity() -> None:
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
            ax.set_title(f"{model} — {kind.replace('_', ' ')}")
            ax.set_ylim(0, 1.0)
            if i == len(kinds) - 1:
                ax.set_xlabel("intensity (× local σ)")
            if j == 0:
                ax.set_ylabel("PICP (target 0.90)")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Phase 1 — Coverage vs corruption intensity: static repair collapses, adaptive holds", y=1.01)
    _save(fig, 1, "calibration_picp_vs_intensity.png")


def fig_p1_mis_ls4() -> None:
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
    ax.set_title("Phase 1 — Interval score: adaptive repair halves MIS")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 1, "calibration_mis_ls4.png")


def fig_p1_significance() -> None:
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
    ax.set_xlabel("ΔRMSE (negatif = soldaki daha iyi), %95 bootstrap CI")
    ax.set_title("Phase 1 — Significance: LSTM≈QT tie; ensemble gain is real")
    _save(fig, 1, "significance_headline.png")


# ------------------------------------------------------------------ phase 2
FAMILY = {  # model -> (family, deterministic?)
    "naive_seasonal": ("klasik", True), "arima": ("klasik", True),
    "sarima": ("klasik", True), "lstm": ("recurrent", True),
    "gru": ("recurrent", True), "dlinear": ("linear", True),
    "lgbm": ("tree", False), "qrf": ("tree", False),
    "qlstm": ("recurrent", False), "qdlinear": ("linear", False),
    "deepar": ("AR-likelihood", False), "qtransformer": ("transformer", False),
    "qtransformer_multi": ("transformer", False),
}
FAM_COLOR = {"klasik": "#7f8c8d", "recurrent": "#1f4e79", "linear": "#8e44ad",
             "tree": "#27ae60", "AR-likelihood": "#c0392b", "transformer": "#e67e22"}


def _base_json(model: str) -> dict | None:
    p = RES / "base" / f"{model}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_p2_roster() -> None:
    rows = []
    for m in FAMILY:
        d = _base_json(m)
        if d is None:
            continue
        rmse = d.get("rmse")
        rows.append((m, rmse, d.get("crps"), FAMILY[m][0]))
    rows.sort(key=lambda r: r[1])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    names = [r[0] for r in rows]
    axes[0].barh(names, [r[1] for r in rows],
                 color=[FAM_COLOR[r[3]] for r in rows])
    axes[0].set_xlabel("RMSE °C (point / median)")
    axes[0].set_title("Roster — point accuracy")
    axes[0].invert_yaxis()
    prob = [r for r in rows if r[2] is not None]
    prob.sort(key=lambda r: r[2])
    axes[1].barh([r[0] for r in prob], [r[2] for r in prob],
                 color=[FAM_COLOR[r[3]] for r in prob])
    axes[1].set_xlabel("CRPS")
    axes[1].set_title("Probabilistik — CRPS")
    axes[1].invert_yaxis()
    fig.suptitle("Phase 2 — 13-model roster (colour = family)", y=1.02)
    _save(fig, 2, "roster_overview.png")


def fig_p2_paired_families() -> None:
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
               label="deterministik" if i == 0 else None)
        ax.bar(i + width / 2, prob_rmse, width, color="#1f4e79",
               label="probabilistik (medyan)" if i == 0 else None)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p[2] for p in pairs])
    ax.set_ylabel("RMSE °C")
    ax.set_ylim(2.0, None)
    ax.set_title("Phase 2 — Paired families: the pinball head costs no point accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 2, "paired_families.png")


def fig_p2_raw_robustness() -> None:
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
    ax.bar([names[i] for i in order], [vals[i] for i in order],
           color=[colors[i] for i in order])
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_ylabel("uncalibrated PICP (level shift 4×)")
    ax.set_title("Phase 2 — Raw robustness: trees do not 'follow' the shift")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 2, "raw_robustness_ls4.png")


def fig_p2_permutation_importance() -> None:
    p = RES / "base" / "feature_importance.json"
    if not p.exists():
        return
    fi = json.loads(p.read_text())
    rows = sorted(fi["splits"]["test"]["permuted"].items(),
                  key=lambda kv: kv[1]["delta_crps"])
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh([k for k, _ in rows], [v["delta_crps"] for _, v in rows], color="#1f4e79")
    ax.set_xlabel("ΔCRPS (channel shuffled across windows)")
    ax.set_title("Phase 2 — Permutation importance (QT-multi, test)")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, 2, "permutation_importance.png")


def fig_p2_covariate_full() -> None:
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
    ax.set_title("Phase 2+ — Full 13-covariate importance (QT, test)\n"
                 "red = temperature-derived (partial leakage) · green = independent sensor · grey = calendar")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, 2, "covariate_importance_full.png")


def fig_p2_covariate_independent() -> None:
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
    ax.set_title("Phase 2+ — Independent covariate importance (LEAKAGE-FREE)\n"
                 "red = target T (dominant) · green = independent sensor · grey = calendar")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, 2, "covariate_importance_independent.png")


def fig_p2_input_value() -> None:
    p = RES / "base" / "exogenous_only.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    e, r = d["exogenous_only"], d["references"]
    pi = RES / "base" / "exogenous_only_independent.json"
    indep = json.loads(pi.read_text())["exogenous_only"]["rmse"] if pi.exists() else None
    bars = [
        ("naive\n(referans)", r.get("naive_seasonal", {}).get("rmse"), "#7f8c8d"),
        ("bağımsız-only\n(p/rh/wv, T izi YOK)", indep, "#c0392b"),
        ("exo+proxy\n(VPmax→T sızıntı)", e["rmse"], "#e67e22"),
        ("sadece T\n(target-only)", r.get("target_only", {}).get("rmse"), "#1f4e79"),
        ("T + 13 cov\n(full)", r.get("full_T_plus_cov", {}).get("rmse"), "#27ae60"),
    ]
    bars = [b for b in bars if b[1] is not None]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.03, f"{b[1]:.2f}", ha="center", fontsize=8)
    ax.axhline(r.get("naive_seasonal", {}).get("rmse", 3.21), color="k", lw=0.7, ls=":")
    ax.set_ylabel("test RMSE °C")
    ax.set_ylim(2.0, None)
    ax.set_title("Phase 2+ — Temperature information is indispensable\n"
                 "without T-proxies (red) it falls BELOW naive; the VPmax leak brings T back")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 2, "input_value.png")


def fig_p2_error_breakdowns() -> None:
    p = RES / "base" / "error_tables_full.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    # show the spread across the roster as 3 panels: per-horizon curves,
    # by-temperature heatmap, by-season heatmap.
    models = sorted(d["per_horizon"], key=lambda m: np.mean(d["per_horizon"][m]))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # panel 1: per-horizon curves (highlight best + worst, grey the rest)
    ax = axes[0]
    for m in d["per_horizon"]:
        ph = d["per_horizon"][m]
        hl = m in (models[0], models[-1])
        ax.plot(range(1, len(ph) + 1), ph, lw=1.6 if hl else 0.7,
                color={"qtransformer_multi": "#27ae60"}.get(m, "#c0392b" if m == models[-1] else "#bbb"),
                label=m if hl else None, zorder=3 if hl else 1)
    ax.set_xlabel("forecast step (hours)")
    ax.set_ylabel("RMSE °C")
    ax.set_title("Per-horizon (full roster)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)

    # panel 2: by-temperature heatmap
    temps = d["temp_order"]
    mat = np.array([[d["by_temperature"][m].get(t, np.nan) for t in temps] for m in models])
    im = axes[1].imshow(mat, aspect="auto", cmap="YlOrRd")
    axes[1].set_xticks(range(len(temps))); axes[1].set_xticklabels(temps, fontsize=7)
    axes[1].set_yticks(range(len(models))); axes[1].set_yticklabels(models, fontsize=6)
    axes[1].set_title("RMSE by temperature range")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    # panel 3: by-season heatmap
    seas = d["season_order"]
    mat2 = np.array([[d["by_season"][m].get(s, np.nan) for s in seas] for m in models])
    im2 = axes[2].imshow(mat2, aspect="auto", cmap="YlGnBu")
    axes[2].set_xticks(range(len(seas))); axes[2].set_xticklabels(seas, fontsize=7)
    axes[2].set_yticks(range(len(models))); axes[2].set_yticklabels(models, fontsize=6)
    axes[2].set_title("RMSE by season")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("Phase 2 — Full-roster error breakdown (per-horizon · temperature · season)", y=1.02)
    _save(fig, 2, "error_breakdowns_full.png")


def fig_p2_natural_extremes() -> None:
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
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("MPIW °C (input-conditional τ)")
    ax.set_title("Phase 2 — Low false-alarm cost on natural extremes (label = slice PICP)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 2, "natural_extremes_falsealarm.png")


# ------------------------------------------------------------------ phase 3
def fig_p3_hpo() -> None:
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
    ax.set_title("Phase 3 — HPO: default vs optimised (selection on validation only)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 3, "hpo_default_vs_best.png")


def fig_p3_multiseed() -> None:
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
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("RMSE °C (3 seed)")
    ax.set_title("Phase 3 — Seed sensitivity (mean ± std; dots = individual seeds)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 3, "multiseed_rmse.png")


def fig_p3_cv() -> None:
    p = RES / "base" / "cv_forward_chaining.json"
    if not p.exists():
        return
    cv = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for m, block in cv["models"].items():
        years = sorted(block["folds"])
        ax.plot([int(y) for y in years], [block["folds"][y]["rmse"] for y in years],
                "o-", label=m)
    ax.set_xlabel("test year (expanding train)")
    ax.set_ylabel("RMSE °C")
    ax.set_title("Phase 3 — Year-based forward-chaining CV (fold variance)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, 3, "cv_fold_variance.png")


def fig_p3_robust_training() -> None:
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
    ax.set_title("Phase 3 — Robust (anomaly-augmented) training: normal vs optimised")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 3, "robust_training_picp.png")


def fig_p3_horizon() -> None:
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
    ax.set_ylabel("hata")
    ax.set_title("Phase 3 — Horizon ablation: error grows with horizon (24h choice)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, 3, "horizon_ablation.png")


def fig_p3_extreme_quantiles() -> None:
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
    ax.set_xticklabels(["%90", "%95", "%98"])
    ax.set_ylabel("PICP")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Phase 3 — Extreme quantiles: 90/95/98% intervals (full + extreme deciles)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 3, "extreme_quantiles.png")


def fig_p3_ensemble_intervals() -> None:
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
    ax.set_title("Phase 3 — Interval combination: ensemble cannot beat the strongest member")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, 3, "ensemble_intervals.png")


# ------------------------------------------------------------------ phase 4
V2_FAULTS = ("point_spike", "contextual_outlier", "level_shift", "flatline",
             "drift", "noise_burst", "gap_imputation", "clock_skew", "fgsm")


def fig_p4_fault_heatmap() -> None:
    """v2 fault catalog x intensity -> raw PICP, for a headline prob model."""
    p = RES / "base" / "report_tables.json"
    if not p.exists():
        return
    rob = json.loads(p.read_text())["robustness_matrix"]
    models = [m for m in ("qtransformer_multi", "qlstm", "deepar", "qtransformer")
              if m in rob]
    if not models:
        return
    model = models[0]
    block = rob[model]
    faults = [f for f in V2_FAULTS if f"{f}_1.0" in block]
    intens = (1.0, 2.0, 4.0)
    mat = np.full((len(faults), len(intens)), np.nan)
    for i, f in enumerate(faults):
        for j, it in enumerate(intens):
            s = block.get(f"{f}_{it:.1f}")
            if s:
                mat[i, j] = s["picp"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=0.95)
    ax.set_xticks(range(len(intens)))
    ax.set_xticklabels([f"{i:g}×" for i in intens])
    ax.set_yticks(range(len(faults)))
    ax.set_yticklabels(faults)
    for i in range(len(faults)):
        for j in range(len(intens)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.axhline(2.5, color="k", lw=1.2)  # separates v1 (top 3) from v2 faults
    ax.set_title(f"Phase 4 — v2 fault catalog, raw PICP ({model})\n(top 3 = v1, bottom 5 = new)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="PICP (target 0.90)")
    _save(fig, 4, "fault_catalog_heatmap.png")


def fig_p4_robust_generalize() -> None:
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
               label=f"{m} normal" if k == 0 else None)
        ax.bar(x + off + w / 2, robust, w, color=base,
               label=f"{m} robust" if k == 0 else None)
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in settings], fontsize=7)
    ax.set_ylabel("raw PICP")
    ax.set_title("Phase 4 — Robust training generalises across 3 architectures (light=normal, dark=robust)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 4, "robust_generalize.png")


def fig_p4_resolution() -> None:
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
    ax.set_title(f"Phase 4 — 10-min resolution does not help\n(diff +{a['rmse']-h['rmse']:.3f}, within seed std ±{std:.3f})")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 4, "resolution_ablation.png")


def fig_p4_robust_plus_cal() -> None:
    p = RES / "base" / "robust_plus_cal.json"
    if not p.exists():
        return
    rc = json.loads(p.read_text())
    settings = [s for s in ("clean", "level_shift_1.0", "level_shift_2.0",
                            "level_shift_4.0", "fgsm_4.0") if s in rc["settings"]]
    corners = [("normal_raw", "#7f8c8d", "normal ham"),
               ("normal_aci", "#1f4e79", "normal+ACI"),
               ("robust_raw", "#e67e22", "robust ham"),
               ("robust_aci", "#27ae60", "robust+ACI")]
    x = np.arange(len(settings))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for k, (key, color, label) in enumerate(corners):
        ax.bar(x + (k - 1.5) * w, [rc["settings"][s][key]["picp"] for s in settings],
               w, color=color, label=label)
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in settings], fontsize=7)
    ax.set_ylabel("PICP")
    ax.set_title("Phase 4 — Model-side (robust) × interval-side (ACI): 4 corners")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, 4, "robust_plus_cal.png")


def fig_p4_leaderboard() -> None:
    p = RES / "base" / "report_tables.json"
    if not p.exists():
        return
    lb = json.loads(p.read_text())["clean_leaderboard"]
    rows = sorted(lb.items(), key=lambda kv: kv[1]["rmse"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    names = [r[0] for r in rows]
    ax.barh(names, [r[1]["rmse"] for r in rows], color="#1f4e79")
    for i, (_, r) in enumerate(rows):
        if "picp" in r:
            ax.text(r["rmse"] + 0.02, i, f"PICP {r['picp']:.2f}", va="center", fontsize=6.5)
    ax.set_xlabel("clean-test RMSE °C (point / median)")
    ax.set_title("Phase 4 — Full-roster clean-test ranking")
    ax.invert_yaxis()
    _save(fig, 4, "final_leaderboard.png")


PHASES = {
    1: (fig_p1_picp_vs_intensity, fig_p1_mis_ls4, fig_p1_significance),
    2: (fig_p2_roster, fig_p2_paired_families, fig_p2_raw_robustness,
        fig_p2_permutation_importance, fig_p2_covariate_full, fig_p2_covariate_independent, fig_p2_input_value, fig_p2_error_breakdowns, fig_p2_natural_extremes),
    3: (fig_p3_hpo, fig_p3_multiseed, fig_p3_cv, fig_p3_robust_training,
        fig_p3_ensemble_intervals),
    4: (fig_p4_fault_heatmap, fig_p4_resolution, fig_p4_robust_generalize, fig_p4_robust_plus_cal, fig_p4_leaderboard),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=("1", "2", "3", "4", "all"))
    args = ap.parse_args()
    phases = (1, 2, 3, 4) if args.phase == "all" else (int(args.phase),)
    for ph in phases:
        print(f"Phase {ph} figures:")
        for fn in PHASES[ph]:
            fn()


if __name__ == "__main__":
    main()
