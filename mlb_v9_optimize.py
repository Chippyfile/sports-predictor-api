#!/usr/bin/env python3
"""
mlb_v9_optimize.py — Full MLB v9 optimization pipeline
=======================================================
Same approach as NBA v28:
  1. Per-model forward selection (Lasso, ElasticNet, CatBoost)
  2. Weight sweep across all combos
  3. Agreement gating analysis
  4. Fold count calibration
  5. Sigma calibration for win probability

Usage:
  python mlb_v9_optimize.py                    # full pipeline
  python mlb_v9_optimize.py --select-only      # just feature selection
  python mlb_v9_optimize.py --sweep-only       # just weight sweep (needs feature files)
  python mlb_v9_optimize.py --upload           # train production + upload
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)

# ── V8 base features ──
V8_FEATURES = [
    "woba_diff", "fip_diff", "k_bb_diff", "bullpen_era_diff",
    "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff", "sp_fip_spread",
    "sp_relative_fip_diff",
    "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold",
    "temp_x_park", "rest_diff", "market_spread", "woba_x_park", "platoon_diff",
    "pyth_residual_diff", "scoring_entropy_diff",
    "first_inn_rate_diff", "clutch_divergence_diff", "opp_adj_form_diff",
    "ump_run_env", "series_game_num",
    "scoring_entropy_combined", "first_inn_rate_combined",
]

# ── Lineup features (raw + advanced) ──
LINEUP_RAW = ["lineup_woba_diff", "lineup_ops_diff", "lineup_iso_diff", "top3_woba_diff"]
LINEUP_ADVANCED = [
    "lineup_delta_diff", "lineup_delta_sum",
    "home_woba_vs_rolling", "away_woba_vs_rolling",
    "lineup_bot3_diff", "lineup_top_heavy_diff", "lineup_consistency_diff",
    "lineup_trend_diff", "lineup_trend_sum",
    "lineup_total_woba", "lineup_total_iso", "lineup_total_top3",
]

# ── Core features (intersection of strong performers) ──
CORE_FEATURES = [
    "k_bb_diff", "woba_x_park", "woba_diff", "market_spread",
    "sp_relative_fip_diff", "sp_ip_diff", "lineup_delta_sum",
    "home_woba_vs_rolling", "away_woba_vs_rolling",
]


def load_data():
    from mlb_retrain import load_data as _load, build_features

    df = _load()
    X = build_features(df)

    # Merge raw lineup
    lineup = pd.read_parquet("mlb_lineup_backfill.parquet")
    lineup["_key"] = lineup["game_date"] + "|" + lineup["home_abbr"]
    df["_key"] = df["game_date"].astype(str) + "|" + df["home_team"].astype(str)
    for src, cols in [("mlb_lineup_backfill.parquet", LINEUP_RAW),
                      ("mlb_lineup_features_advanced.parquet", LINEUP_ADVANCED)]:
        if not os.path.exists(src):
            for c in cols: X[c] = 0
            continue
        tmp = pd.read_parquet(src)
        tmp["_key"] = tmp["game_date"] + "|" + tmp["home_abbr"]
        avail = [c for c in cols if c in tmp.columns]
        if avail:
            sub = tmp[["_key"] + avail].drop_duplicates(subset="_key", keep="first")
            merged = df[["_key"]].merge(sub, on="_key", how="left")
            for c in avail:
                X[c] = merged[c].fillna(0).values

    y = (df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)).values
    sp = X["market_spread"].values if "market_spread" in X.columns else np.zeros(len(X))
    has_sp = np.abs(sp) > 0.1
    seasons = pd.to_numeric(df.get("season", 2026), errors="coerce").fillna(2026).astype(int).values
    w = np.array([{0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8}.get(2026 - s, 0.7) for s in seasons])

    # All candidate features
    exclude = {"market_total", "spread_vs_market", "run_diff_pred", "has_heuristic", "has_market"}
    candidates = [f for f in X.columns if f not in exclude and f in X.columns]
    X = X[candidates].fillna(0)

    print(f"  Data: {len(X)} games, {len(candidates)} candidate features")
    print(f"  Lineup coverage: {(X.get('lineup_delta_sum', pd.Series([0])).abs() > 0.001).sum()}/{len(X)}")
    return X, candidates, y, sp, has_sp, w, seasons


def make_model(name):
    if name == "Lasso":
        return Lasso(alpha=0.01, max_iter=5000, random_state=SEED)
    elif name == "ElasticNet":
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=SEED)
    elif name == "CatBoost":
        return CatBoostRegressor(depth=6, iterations=200, learning_rate=0.03,
                                 subsample=0.8, min_data_in_leaf=20,
                                 random_seed=SEED, verbose=0)


def wf_oof(model_name, X_vals, y, w, n_folds=20):
    """Walk-forward OOF predictions."""
    n = len(X_vals); fs = n // (n_folds + 1); mt = max(fs * 3, 1000)
    oof = np.full(n, np.nan)
    for fold in range(n_folds):
        ts = mt + fold * fs; te = min(ts + fs, n)
        if ts >= n: break
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_vals[:ts]); Xte = sc.transform(X_vals[ts:te])
        m = make_model(model_name)
        m.fit(Xtr, y[:ts], sample_weight=w[:ts])
        oof[ts:te] = m.predict(Xte)
    return oof


def ats_score(oof, y, sp, has_sp):
    """Composite ATS score for MLB: weighted accuracy at 1+, 1.5+, 2+ tiers."""
    ats = y + sp; push = ats == 0
    valid = ~np.isnan(oof) & ~push & has_sp
    edge = np.abs(oof[valid] - (-sp[valid]))
    p = oof[valid]; yv = y[valid]; sv = sp[valid]; atsv = ats[valid]
    results = {}
    for t in [0.5, 1.0, 1.5, 2.0]:
        mask = (np.abs(p - (-sv)) >= t) & (atsv != 0)
        if mask.sum() >= 30:
            correct = ((p[mask] > (-sv[mask])) == (atsv[mask] > 0)).mean()
            results[t] = correct
    # Composite: emphasize 1.0 and 1.5 tiers
    c1 = results.get(1.0, 0.5)
    c15 = results.get(1.5, 0.5)
    c2 = results.get(2.0, 0.5)
    return c1 * 0.4 + c15 * 0.35 + c2 * 0.25, results


def ats_detail(oof, y, sp, has_sp, label=""):
    """Full ATS breakdown."""
    ats = y + sp; push = ats == 0
    valid = ~np.isnan(oof) & ~push & has_sp
    p = oof[valid]; yv = y[valid]; sv = sp[valid]; atsv = ats[valid]
    print(f"\n  {label}")
    print(f"  {'Thresh':>7} {'Games':>6} {'ATS%':>6} {'ROI%':>7} {'ML%':>6}")
    print(f"  {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*6}")
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        mask = (np.abs(p - (-sv)) >= t) & (atsv != 0)
        if mask.sum() < 25: continue
        ats_c = ((p[mask] > (-sv[mask])) == (atsv[mask] > 0)).mean()
        ml_c = ((p[mask] > 0) == (yv[mask] > 0)).mean()
        roi = (ats_c * 1.909 - 1) * 100
        print(f"  {t:>6}+ {mask.sum():>6} {ats_c:>5.1%} {roi:>+6.1f}% {ml_c:>5.1%}")


# ═══════════════════════════════════════════════════════════
# FORWARD SELECTION
# ═══════════════════════════════════════════════════════════

def forward_select(model_name, X, candidates, y, sp, has_sp, w, n_folds=20):
    core = [f for f in CORE_FEATURES if f in candidates]
    remaining = [f for f in candidates if f not in core]

    print(f"\n  ── {model_name} FORWARD SELECTION ──")
    core_oof = wf_oof(model_name, X[core].values, y, w, n_folds)
    core_score, cd = ats_score(core_oof, y, sp, has_sp)
    print(f"  Core ({len(core)}): {core_score:.5f} (1+={cd.get(1,0):.1%} 1.5+={cd.get(1.5,0):.1%})")

    selected = list(core)
    best_score = core_score
    added = []

    t0 = time.time()
    for i, feat in enumerate(remaining):
        trial = selected + [feat]
        oof = wf_oof(model_name, X[trial].values, y, w, n_folds)
        score, detail = ats_score(oof, y, sp, has_sp)
        delta = score - best_score
        if delta > 0.00005:
            selected.append(feat)
            best_score = score
            added.append((feat, delta, detail))
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 1
        rem = (len(remaining) - i - 1) / rate / 60
        if (i + 1) % 10 == 0:
            print(f"    [{i+1:3d}/{len(remaining)}] {len(selected)} feats, score={best_score:.5f} ~{rem:.1f}m")

    print(f"\n  {model_name}: {len(selected)} features, score={best_score:.5f} ({best_score-core_score:+.5f})")
    print(f"  Added {len(added)} features:")
    for f, d, det in added:
        marker = " ← LINEUP" if f in LINEUP_RAW + LINEUP_ADVANCED else ""
        print(f"    {f:35s} d={d:+.6f} (1+={det.get(1,0):.1%} 1.5+={det.get(1.5,0):.1%}){marker}")

    return selected


# ═══════════════════════════════════════════════════════════
# WEIGHT SWEEP + AGREEMENT
# ═══════════════════════════════════════════════════════════

def sweep_and_agree(X, y, sp, has_sp, w, lf, ef, cf, n_folds=25):
    """Generate OOF for all 3 models, sweep weights, test agreement."""
    print(f"\n{'='*70}")
    print(f"  GENERATING OOF ({n_folds}-fold)")
    print(f"  Lasso: {len(lf)} feats | EN: {len(ef)} feats | CatBoost: {len(cf)} feats")
    print(f"{'='*70}")

    oof_l = wf_oof("Lasso", X[lf].values, y, w, n_folds); print(f"  Lasso done")
    oof_e = wf_oof("ElasticNet", X[ef].values, y, w, n_folds); print(f"  ElasticNet done")
    oof_c = wf_oof("CatBoost", X[cf].values, y, w, n_folds); print(f"  CatBoost done")

    ats = y + sp; push = ats == 0

    # ── Solo baselines ──
    print(f"\n  SOLO BASELINES:")
    for label, oof in [("Lasso", oof_l), ("ElasticNet", oof_e), ("CatBoost", oof_c)]:
        ats_detail(oof, y, sp, has_sp, label)

    # ── Weight sweep ──
    print(f"\n{'='*70}")
    print(f"  WEIGHT SWEEP")
    print(f"{'='*70}")

    combos = []
    # Two-model
    for w_a in np.arange(0.1, 1.0, 0.1):
        w_b = round(1 - w_a, 2)
        combos.append(("CB/Lasso", {"cb": w_a, "lasso": w_b, "en": 0}))
        combos.append(("CB/EN", {"cb": w_a, "en": w_b, "lasso": 0}))
        combos.append(("Lasso/EN", {"lasso": w_a, "en": w_b, "cb": 0}))
    # Three-model
    for wc in np.arange(0.1, 0.9, 0.1):
        for wl in np.arange(0.1, round(1 - wc, 2) + 0.01, 0.1):
            we = round(1 - wc - wl, 2)
            if we < 0.05: continue
            combos.append(("All3", {"cb": round(wc, 2), "lasso": round(wl, 2), "en": round(we, 2)}))
    # Equal
    combos.append(("Equal", {"cb": 1/3, "lasso": 1/3, "en": 1/3}))

    best_by_comp = None; best_comp = 0
    all_results = []
    for name, wts in combos:
        blend = wts["cb"] * oof_c + wts["lasso"] * oof_l + wts["en"] * oof_e
        score, detail = ats_score(blend, y, sp, has_sp)
        w_str = f"CB={wts['cb']:.1f}/L={wts['lasso']:.1f}/E={wts['en']:.1f}"
        row = {"name": name, "weights": w_str, "composite": score}
        for t in [0.5, 1.0, 1.5, 2.0]:
            row[f"ats_{t}"] = detail.get(t, 0.5)
        all_results.append(row)
        if score > best_comp:
            best_comp = score; best_by_comp = (name, wts, w_str)

    all_results.sort(key=lambda x: -x["composite"])
    print(f"\n  TOP 15 COMBOS:")
    print(f"  {'#':>3} {'Combo':>10} {'Weights':>22} {'Comp':>7} {'@1.0':>7} {'@1.5':>7} {'@2.0':>7}")
    for i, row in enumerate(all_results[:15]):
        print(f"  {i+1:>3} {row['name']:>10} {row['weights']:>22} "
              f"{row['composite']:.4f} {row.get('ats_1.0',0):.1%} {row.get('ats_1.5',0):.1%} {row.get('ats_2.0',0):.1%}")

    # ── Agreement analysis with best weights ──
    bw = best_by_comp[1]
    blend = bw["cb"] * oof_c + bw["lasso"] * oof_l + bw["en"] * oof_e
    all_agree = ((oof_l > 0) & (oof_e > 0) & (oof_c > 0)) | \
                ((oof_l < 0) & (oof_e < 0) & (oof_c < 0))
    valid = ~np.isnan(blend) & ~push & has_sp
    edge = np.abs(blend - (-sp))

    print(f"\n{'='*70}")
    print(f"  AGREEMENT ANALYSIS (best: {best_by_comp[2]})")
    print(f"{'='*70}")
    print(f"  All 3 agree: {all_agree[valid].mean():.1%}")

    print(f"\n  {'Filter':>35} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
    print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*7}")
    for label, mask_extra in [
        ("All (edge≥1.0)", edge >= 1.0),
        ("Agree + edge≥1.0", all_agree & (edge >= 1.0)),
        ("Disagree + edge≥1.0", ~all_agree & (edge >= 1.0)),
        ("All (edge≥1.5)", edge >= 1.5),
        ("Agree + edge≥1.5", all_agree & (edge >= 1.5)),
        ("Disagree + edge≥1.5", ~all_agree & (edge >= 1.5)),
        ("All (edge≥2.0)", edge >= 2.0),
        ("Agree + edge≥2.0", all_agree & (edge >= 2.0)),
    ]:
        m = valid & mask_extra
        if m.sum() < 25: continue
        model_home = blend[m] > (-sp[m])
        actual_home = ats[m] > 0
        correct = (model_home == actual_home).mean()
        roi = (correct * 1.909 - 1) * 100
        print(f"  {label:>35} {m.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")

    # ── Flat vs Tiered profit ──
    print(f"\n  PROFIT COMPARISON (edge≥1.0):")
    for label, gate in [("Flat 1u", None), ("All agree", all_agree)]:
        m = valid & (edge >= 1.0)
        if gate is not None: m = m & gate
        if m.sum() < 25: continue
        model_home = blend[m] > (-sp[m])
        actual_home = ats[m] > 0
        correct = model_home == actual_home
        profit = np.where(correct, 0.909, -1.0).sum()
        roi = profit / m.sum() * 100
        print(f"    {label:>15}: {m.sum()} bets, {correct.mean():.1%} ATS, {profit:+.1f}u, {roi:+.1f}% ROI")

    # ── Fold calibration ──
    print(f"\n{'='*70}")
    print(f"  FOLD CALIBRATION (CatBoost solo, {len(cf)} features)")
    print(f"{'='*70}")
    for nf in [15, 20, 25, 30]:
        oof_test = wf_oof("CatBoost", X[cf].values, y, w, nf)
        score, detail = ats_score(oof_test, y, sp, has_sp)
        print(f"  {nf:>2} folds: composite={score:.5f} (1+={detail.get(1,0):.1%} 1.5+={detail.get(1.5,0):.1%} 2+={detail.get(2,0):.1%})")

    # ── Full detail on best blend ──
    ats_detail(blend, y, sp, has_sp, f"BEST BLEND: {best_by_comp[2]}")

    return oof_l, oof_e, oof_c, best_by_comp


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  MLB v9 FULL OPTIMIZATION")
    print("  Per-model selection + weight sweep + agreement + calibration")
    print("=" * 70)

    X, candidates, y, sp, has_sp, w, seasons = load_data()

    if not args.sweep_only:
        # ── Per-model feature selection ──
        lf = forward_select("Lasso", X, candidates, y, sp, has_sp, w, n_folds=15)
        ef = forward_select("ElasticNet", X, candidates, y, sp, has_sp, w, n_folds=15)
        cf = forward_select("CatBoost", X, candidates, y, sp, has_sp, w, n_folds=15)

        # Save feature lists
        for name, feats in [("lasso", lf), ("elasticnet", ef), ("catboost", cf)]:
            with open(f"mlb_v9_{name}_features.txt", "w") as f:
                for feat in feats:
                    f.write(feat + "\n")
            print(f"  Saved {len(feats)} features to mlb_v9_{name}_features.txt")

        if args.select_only:
            print("\n  Done (--select-only).")
            return
    else:
        # Load saved features
        def load_feats(path):
            with open(path) as f: return [l.strip() for l in f if l.strip()]
        lf = load_feats("mlb_v9_lasso_features.txt")
        ef = load_feats("mlb_v9_elasticnet_features.txt")
        cf = load_feats("mlb_v9_catboost_features.txt")

    # ── Weight sweep + agreement + fold calibration ──
    oof_l, oof_e, oof_c, best = sweep_and_agree(X, y, sp, has_sp, w, lf, ef, cf, n_folds=25)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Lasso: {len(lf)} features")
    print(f"  ElasticNet: {len(ef)} features")
    print(f"  CatBoost: {len(cf)} features")
    print(f"  Best weights: {best[2]}")
    print(f"\n  Done. Use --upload to deploy.")


if __name__ == "__main__":
    main()
