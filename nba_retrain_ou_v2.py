#!/usr/bin/env python3
"""
nba_retrain_ou_v2.py — NBA O/U Triple Agreement System
========================================================
Ported from NCAA O/U v4 (63-80% holdout accuracy).

Architecture:
  3 residual models (predict market error) — Lasso + LGBM + CatBoost
  2 classifier models (predict P(under)) — LogReg + LGBM
  ATS score models (predict home + away scores) — from existing ATS model

Target: residual = actual_total - market_total (NOT total directly)
Insight: predicting WHERE VEGAS IS WRONG dramatically outperforms direct total prediction.

UNDER is structurally stronger than OVER (~5-10% accuracy advantage, same as NCAA).

Usage:
    python3 nba_retrain_ou_v2.py              # Train + evaluate
    python3 nba_retrain_ou_v2.py --upload     # Train + upload to Supabase as 'nba_ou'
"""
import sys, os, time, warnings, argparse
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

from nba_build_features_v27 import load_training_data, build_features

SEED = 42
N_FOLDS = 30

# ── O/U Combined features (sums/averages, not diffs) ──
# Totals care about scoring ENVIRONMENT, not which team scores more
def build_ou_combined(df):
    """Build O/U-specific combined features alongside V27 diffs."""
    f = pd.DataFrame(index=df.index)

    sc = lambda col, d=0: pd.to_numeric(df.get(col, d), errors="coerce").fillna(d)

    # Market
    f["market_total"] = sc("market_ou_total", 220)
    mkt_t2 = sc("ou_total", 0)
    mkt_t3 = sc("dk_ou", 0)
    f["market_total"] = np.where(f["market_total"] == 0, mkt_t2, f["market_total"])
    f["market_total"] = np.where(f["market_total"] == 0, mkt_t3, f["market_total"])

    # Scoring environment (combined)
    h_ppg = sc("home_ppg", 112); a_ppg = sc("away_ppg", 112)
    f["ppg_combined"] = h_ppg + a_ppg
    f["opp_ppg_combined"] = sc("home_opp_ppg", 112) + sc("away_opp_ppg", 112)
    f["ou_gap"] = f["ppg_combined"] - f["market_total"]

    # Pace
    h_tempo = sc("home_tempo", 100); a_tempo = sc("away_tempo", 100)
    f["tempo_combined"] = h_tempo + a_tempo
    f["pace_min"] = np.minimum(h_tempo, a_tempo)

    # Shooting (averages)
    h_fg = sc("home_fgpct", 0.46); a_fg = sc("away_fgpct", 0.46)
    h_3p = sc("home_threepct", 0.36); a_3p = sc("away_threepct", 0.36)
    h_ft = sc("home_ftpct", 0.77); a_ft = sc("away_ftpct", 0.77)
    f["fgpct_avg"] = (h_fg + a_fg) / 2
    f["threepct_avg"] = (h_3p + a_3p) / 2
    f["ftpct_avg"] = (h_ft + a_ft) / 2
    f["efg_avg"] = ((h_fg + 0.2 * h_3p) + (a_fg + 0.2 * a_3p)) / 2

    # Turnovers/steals combined
    f["turnovers_combined"] = sc("home_turnovers", 14) + sc("away_turnovers", 14)
    f["steals_combined"] = sc("home_steals", 7.5) + sc("away_steals", 7.5)

    # Offensive rebounding (more OREBs → more possessions → higher scoring)
    f["oreb_combined"] = sc("home_orb_pct", 0.25) + sc("away_orb_pct", 0.25)

    # Win quality (blowout risk = lower total due to garbage time)
    hw = sc("home_wins", 20); hl = sc("home_losses", 20)
    aw = sc("away_wins", 20); al = sc("away_losses", 20)
    f["blowout_risk"] = abs(hw / np.maximum(hw + hl, 1) - aw / np.maximum(aw + al, 1))

    # Rest combined (more rest = higher scoring due to fresher legs)
    h_rest = sc("home_days_rest", 2); a_rest = sc("away_days_rest", 2)
    f["rest_combined"] = h_rest + a_rest
    f["b2b_either"] = ((h_rest == 0) | (a_rest == 0)).astype(int)

    # Rolling PBP combined
    f["roll_paint_combined"] = sc("home_roll_paint_pts", 0) + sc("away_roll_paint_pts", 0)
    f["roll_fast_break_combined"] = sc("home_roll_fast_break_pts", 0) + sc("away_roll_fast_break_pts", 0)
    f["roll_bench_combined"] = sc("home_roll_bench_pts", 0) + sc("away_roll_bench_pts", 0)

    # Enrichment combined
    f["ceiling_combined"] = sc("home_ceiling", 0) + sc("away_ceiling", 0)
    f["floor_combined"] = sc("home_floor", 0) + sc("away_floor", 0)
    f["scoring_var_combined"] = sc("home_scoring_var", 0) + sc("away_scoring_var", 0)

    # Altitude (Denver = higher scoring historically)
    f["altitude_factor"] = sc("altitude_factor", 0)

    # Ref O/U bias (refs who call more fouls → more FTs → higher scoring)
    f["ref_ou_bias"] = sc("ref_ou_bias", 0)
    f["ref_pace_impact"] = sc("ref_pace_impact", 0)

    # Market overround
    f["overround"] = sc("overround", 0.04)

    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA O/U v2 — Triple Agreement (Residual) System")
    print("  Ported from NCAA O/U v4 (63-80% holdout)")
    print("=" * 70)
    t0 = time.time()

    # ── Load data ──
    df = load_training_data("nba_training_data.parquet")

    # Targets
    df["actual_total"] = pd.to_numeric(df["actual_home_score"], errors="coerce") + \
                         pd.to_numeric(df["actual_away_score"], errors="coerce")

    # Market total (cascade backfill)
    mkt = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)
    for fallback in ["ou_total", "dk_ou"]:
        if fallback in df.columns:
            fb = pd.to_numeric(df[fallback], errors="coerce").fillna(0)
            mkt = np.where(mkt == 0, fb, mkt)
    df["market_total_clean"] = mkt

    # Filter: need real totals + market total
    valid_mask = (df["actual_total"].notna()) & (df["market_total_clean"] > 100)
    df = df[valid_mask].copy()
    print(f"\n  Games with scores + market total: {len(df)}")

    # Residual target (where Vegas is wrong)
    df["residual"] = df["actual_total"] - df["market_total_clean"]
    df["went_under"] = (df["actual_total"] < df["market_total_clean"]).astype(int)

    print(f"  Residual: mean={df['residual'].mean():.2f}, std={df['residual'].std():.2f}")
    print(f"  Under rate: {df['went_under'].mean():.1%}")

    # ── Build features ──
    # V27 diff features (same as ATS model)
    X_v27, v27_names = build_features(df)

    # O/U combined features
    X_ou = build_ou_combined(df)
    ou_names = list(X_ou.columns)

    # Merge
    X_all = pd.concat([X_v27, X_ou], axis=1)
    # Remove duplicates (ou_gap, overround, altitude_factor may be in both)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    all_feats = list(X_all.columns)
    print(f"  Total candidate features: {len(all_feats)} ({len(v27_names)} diff + {len(ou_names)} combined)")

    # Sort by date
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort().values
    X_all = X_all.iloc[sort_idx].reset_index(drop=True)
    df = df.iloc[sort_idx].reset_index(drop=True)

    y_res = df["residual"].values
    y_under = df["went_under"].values
    y_home = pd.to_numeric(df["actual_home_score"], errors="coerce").fillna(110).values
    y_away = pd.to_numeric(df["actual_away_score"], errors="coerce").fillna(110).values
    mkt_total = df["market_total_clean"].values

    # ── Feature selection via Lasso/LogReg sparsity ──
    print(f"\n  Feature selection...")
    sc_sel = StandardScaler()
    X_sel = sc_sel.fit_transform(X_all)

    # Residual features — separate per model type
    # Lasso: ultra-tight (α=0.5, ~6 features)
    lasso_tight = Lasso(alpha=0.5, max_iter=5000)
    lasso_tight.fit(X_sel, y_res)
    res_feats_lasso = [f for f, c in zip(all_feats, lasso_tight.coef_) if abs(c) > 0.001]
    print(f"  Residual features (Lasso α=0.5): {len(res_feats_lasso)}")

    # Trees: looser (α=0.15, ~45 features)
    lasso_loose = Lasso(alpha=0.15, max_iter=5000)
    lasso_loose.fit(X_sel, y_res)
    res_feats_trees = [f for f, c in zip(all_feats, lasso_loose.coef_) if abs(c) > 0.001]
    print(f"  Residual features (Trees α=0.15): {len(res_feats_trees)}")

    # Superset for bundle storage
    res_feats = list(dict.fromkeys(res_feats_trees + res_feats_lasso))

    # Classifier features (LogReg C=0.01)
    lr_sel = LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=5000, random_state=SEED)
    lr_sel.fit(X_sel, y_under)
    cls_feats = [f for f, c in zip(all_feats, lr_sel.coef_[0]) if abs(c) > 0.001]
    print(f"  Classifier features: {len(cls_feats)} (LogReg C=0.01)")

    # ATS score features — reuse V27 feature set for home/away score prediction
    ats_feats = [f for f in all_feats if f in X_all.columns][:60]  # use all available

    # ── Walk-forward validation ──
    print(f"\n  Walk-forward validation ({N_FOLDS} folds)...")
    fold_size = len(X_all) // (N_FOLDS + 2)
    min_train = fold_size * 2

    oof_res = np.full(len(X_all), np.nan)
    oof_cls = np.full(len(X_all), np.nan)
    oof_ats_total = np.full(len(X_all), np.nan)

    for fold in range(N_FOLDS):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, len(X_all))
        if ts >= len(X_all):
            break

        # ── Residual models (dual feature sets) ──
        # Lasso: tight features
        X_res_l = X_all[res_feats_lasso]
        sc_rl = StandardScaler()
        Xrl_tr = sc_rl.fit_transform(X_res_l.iloc[:ts])
        Xrl_te = sc_rl.transform(X_res_l.iloc[ts:te])

        # Trees: looser features
        X_res_t = X_all[res_feats_trees]
        sc_rt = StandardScaler()
        Xrt_tr = sc_rt.fit_transform(X_res_t.iloc[:ts])
        Xrt_te = sc_rt.transform(X_res_t.iloc[ts:te])

        # Model 1: Lasso (tight features)
        m_lasso = Lasso(alpha=0.5, max_iter=5000)
        m_lasso.fit(Xrl_tr, y_res[:ts])
        p1 = m_lasso.predict(Xrl_te)

        # Model 2: LGBM (loose features)
        if HAS_LGBM:
            m_lgbm = LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    num_leaves=5, min_child_samples=50, verbose=-1, random_state=SEED)
            m_lgbm.fit(Xrt_tr, y_res[:ts])
            p2 = m_lgbm.predict(Xrt_te)
        else:
            p2 = p1  # fallback

        # Model 3: CatBoost (loose features)
        if HAS_CAT:
            m_cat = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                                       l2_leaf_reg=3, verbose=0, random_seed=SEED)
            m_cat.fit(Xrt_tr, y_res[:ts])
            p3 = m_cat.predict(Xrt_te)
        else:
            p3 = p1  # fallback

        oof_res[ts:te] = (p1 + p2 + p3) / 3

        # ── Classifier models ──
        X_cls = X_all[cls_feats]
        sc_c = StandardScaler()
        Xc_tr = sc_c.fit_transform(X_cls.iloc[:ts])
        Xc_te = sc_c.transform(X_cls.iloc[ts:te])

        # Model 1: LogReg
        m_lr = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)
        m_lr.fit(Xc_tr, y_under[:ts])
        cp1 = m_lr.predict_proba(Xc_te)[:, 1]

        # Model 2: LGBM classifier
        if HAS_LGBM:
            m_lgbm_c = LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                       num_leaves=5, min_child_samples=50, verbose=-1, random_state=SEED)
            m_lgbm_c.fit(Xc_tr, y_under[:ts])
            cp2 = m_lgbm_c.predict_proba(Xc_te)[:, 1]
        else:
            cp2 = cp1

        oof_cls[ts:te] = (cp1 + cp2) / 2

        # ── ATS score models (predict home + away scores) ──
        X_ats = X_all[ats_feats]
        sc_a = StandardScaler()
        Xa_tr = sc_a.fit_transform(X_ats.iloc[:ts])
        Xa_te = sc_a.transform(X_ats.iloc[ts:te])

        m_home = Lasso(alpha=0.1, max_iter=5000)
        m_home.fit(Xa_tr, y_home[:ts])
        m_away = Lasso(alpha=0.1, max_iter=5000)
        m_away.fit(Xa_tr, y_away[:ts])

        oof_ats_total[ts:te] = m_home.predict(Xa_te) + m_away.predict(Xa_te)

    # ── Evaluate ──
    valid = ~np.isnan(oof_res) & (mkt_total > 100)
    n_valid = valid.sum()
    ats_edge = oof_ats_total[valid] - mkt_total[valid]
    res = oof_res[valid]
    cls = oof_cls[valid]
    yu = y_under[valid]
    yo = 1 - yu

    res_mae = np.mean(np.abs(res))
    print(f"\n  Residual MAE: {res_mae:.3f}")
    print(f"  Classifier AUC-proxy: {np.mean((cls > 0.5) == yu.astype(bool)):.1%}")

    # ── Threshold sweep ──
    print(f"\n  {'Dir':>6s} {'Res':>8s} {'Cls':>8s} {'ATS':>8s} {'Picks':>7s} {'Acc':>6s} {'ROI':>7s}")
    print(f"  {'-'*55}")

    best_under_tiers = {}
    best_over_tiers = {}

    # UNDER sweeps
    for res_t in [-0.5, -1.0, -1.5, -2.0, -2.5]:
        for cls_t in [0.52, 0.54, 0.56, 0.58]:
            for ats_t in [0, -1, -2, -3, -4, -5]:
                mask = (res <= res_t) & (cls >= cls_t) & (ats_edge <= ats_t)
                n = mask.sum()
                if n < 30:
                    continue
                acc = yu[mask].mean()
                roi = (acc * 1.909 - 1) * 100
                if acc >= 0.56 and roi > 0:
                    key = f"U|r{res_t}|c{cls_t}|a{ats_t}"
                    # Keep best per approximate tier
                    tier = 1 if n > 200 else (2 if n > 80 else 3)
                    if tier not in best_under_tiers or acc > best_under_tiers[tier]["acc"]:
                        best_under_tiers[tier] = {
                            "res_avg": res_t, "cls_avg": cls_t, "ats_edge": ats_t,
                            "n": n, "acc": acc, "roi": roi
                        }

    # OVER sweeps
    for res_t in [0.5, 1.0, 1.5, 2.0, 2.5]:
        for cls_t in [0.48, 0.46, 0.44, 0.42, 0.40]:
            for ats_t in [0, 1, 2, 3, 4, 5]:
                mask = (res >= res_t) & (cls <= cls_t) & (ats_edge >= ats_t)
                n = mask.sum()
                if n < 20:
                    continue
                acc = yo[mask].mean()
                roi = (acc * 1.909 - 1) * 100
                if acc >= 0.55 and roi > 0:
                    tier = 1 if n > 150 else (2 if n > 50 else 3)
                    if tier not in best_over_tiers or acc > best_over_tiers[tier]["acc"]:
                        best_over_tiers[tier] = {
                            "res_avg": res_t, "cls_avg": cls_t, "ats_edge": ats_t,
                            "n": n, "acc": acc, "roi": roi
                        }

    print(f"\n  UNDER tiers (validated):")
    for tier in sorted(best_under_tiers.keys()):
        t = best_under_tiers[tier]
        print(f"    {tier}u: res≤{t['res_avg']:+.1f}, cls≥{t['cls_avg']:.0%}, ats≤{t['ats_edge']:+d}"
              f"  → {t['acc']:.1%} ({t['n']} picks, {t['roi']:+.1f}% ROI)")

    print(f"\n  OVER tiers (validated):")
    for tier in sorted(best_over_tiers.keys()):
        t = best_over_tiers[tier]
        print(f"    {tier}u: res≥{t['res_avg']:+.1f}, cls≤{t['cls_avg']:.0%}, ats≥{t['ats_edge']:+d}"
              f"  → {t['acc']:.1%} ({t['n']} picks, {t['roi']:+.1f}% ROI)")

    # ── Direct total comparison (baseline) ──
    oof_direct = oof_res[valid] + mkt_total[valid]  # residual + market = predicted total
    direct_mae = np.mean(np.abs(oof_direct - (y_home[valid] + y_away[valid])))
    print(f"\n  Predicted total MAE (residual + market): {direct_mae:.3f}")
    print(f"  Market-only MAE: {np.mean(np.abs(mkt_total[valid] - (y_home[valid] + y_away[valid]))):.3f}")

    # ── Simple edge analysis (for comparison with current O/U v1) ──
    print(f"\n  Simple edge analysis (residual model only):")
    for edge in [1, 2, 3, 4, 5]:
        over_mask = res >= edge
        under_mask = res <= -edge
        has_pick = over_mask | under_mask
        if has_pick.sum() < 20:
            continue
        correct = (over_mask & (yo == 1)) | (under_mask & (yu == 1))
        decided = has_pick
        acc = correct[decided].mean()
        n_o = over_mask.sum()
        n_u = under_mask.sum()
        # Split accuracy
        o_acc = yo[over_mask].mean() if n_o > 0 else 0
        u_acc = yu[under_mask].mean() if n_u > 0 else 0
        roi = (acc * 1.909 - 1) * 100
        print(f"    edge≥{edge}: {acc:.1%} ({decided.sum()}g, {roi:+.1f}% ROI)"
              f" | OVER {o_acc:.0%} ({n_o}g) UNDER {u_acc:.0%} ({n_u}g)")

    # ── Production training (full data) ──
    print(f"\n  Training production models on full data...")

    # Residual — dual feature sets
    sc_res_lasso = StandardScaler()
    Xrl_full = sc_res_lasso.fit_transform(X_all[res_feats_lasso])
    sc_res_trees = StandardScaler()
    Xrt_full = sc_res_trees.fit_transform(X_all[res_feats_trees])

    res_models = []
    m1 = Lasso(alpha=0.5, max_iter=5000); m1.fit(Xrl_full, y_res); res_models.append(m1)
    if HAS_LGBM:
        m2 = LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                            num_leaves=5, min_child_samples=50, verbose=-1, random_state=SEED)
        m2.fit(Xrt_full, y_res); res_models.append(m2)
    if HAS_CAT:
        m3 = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                                l2_leaf_reg=3, verbose=0, random_seed=SEED)
        m3.fit(Xrt_full, y_res); res_models.append(m3)
    print(f"    Residual: {len(res_models)} models (Lasso={len(res_feats_lasso)}f, Trees={len(res_feats_trees)}f)")

    # Classifier
    sc_cls = StandardScaler()
    Xc_full = sc_cls.fit_transform(X_all[cls_feats])
    cls_models = []
    mc1 = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED); mc1.fit(Xc_full, y_under); cls_models.append(mc1)
    if HAS_LGBM:
        mc2 = LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              num_leaves=5, min_child_samples=50, verbose=-1, random_state=SEED)
        mc2.fit(Xc_full, y_under); cls_models.append(mc2)
    print(f"    Classifier: {len(cls_models)} models, {len(cls_feats)} features")

    # ATS score models
    sc_ats = StandardScaler()
    Xa_full = sc_ats.fit_transform(X_all[ats_feats])
    ats_home_models = []
    ats_away_models = []
    mh = Lasso(alpha=0.1, max_iter=5000); mh.fit(Xa_full, y_home); ats_home_models.append(mh)
    ma = Lasso(alpha=0.1, max_iter=5000); ma.fit(Xa_full, y_away); ats_away_models.append(ma)
    if HAS_LGBM:
        mh2 = LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, verbose=-1, random_state=SEED)
        mh2.fit(Xa_full, y_home); ats_home_models.append(mh2)
        ma2 = LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, verbose=-1, random_state=SEED)
        ma2.fit(Xa_full, y_away); ats_away_models.append(ma2)
    print(f"    ATS scores: {len(ats_home_models)} home + {len(ats_away_models)} away models, {len(ats_feats)} features")

    # ── Bundle ──
    bundle = {
        "res_models": res_models,
        "cls_models": cls_models,
        "ats_home_models": ats_home_models,
        "ats_away_models": ats_away_models,
        "res_scaler_lasso": sc_res_lasso,
        "res_scaler_trees": sc_res_trees,
        "res_scaler": sc_res_trees,  # backward compat
        "cls_scaler": sc_cls,
        "ats_scaler": sc_ats,
        "res_feature_cols_lasso": res_feats_lasso,
        "res_feature_cols_trees": res_feats_trees,
        "res_feature_cols": res_feats,  # superset for backward compat
        "cls_feature_cols": cls_feats,
        "ats_feature_cols": ats_feats,
        # Keep these for backward compat with nba_full_predict.py v1 O/U path
        "reg": None,
        "scaler": None,
        "ou_feature_cols": None,
        "under_tiers": best_under_tiers,
        "over_tiers": best_over_tiers,
        "model_type": "nba_ou_v2_triple",
        "architecture": f"3res({len(res_models)}) + 2cls({len(cls_models)}) + ATS({len(ats_home_models)}+{len(ats_away_models)})",
        "n_games": len(df),
        "res_mae": round(res_mae, 3),
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    # Save locally
    import joblib
    local_path = "nba_ou_v2.pkl"
    joblib.dump(bundle, local_path)
    pkl_size = os.path.getsize(local_path) / 1024
    print(f"\n  Saved: {local_path} ({pkl_size:.0f} KB)")

    if args.upload:
        print(f"\n  Uploading to Supabase model_store as 'nba_ou'...")
        try:
            from db import save_model
            save_model("nba_ou", bundle)
            print(f"  ✅ Uploaded successfully")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  NBA O/U v2 COMPLETE")
    print(f"  Architecture: {len(res_models)}res + {len(cls_models)}cls + ATS({len(ats_home_models)}+{len(ats_away_models)})")
    print(f"  Features: res={len(res_feats)}, cls={len(cls_feats)}, ats={len(ats_feats)}")
    print(f"  Games: {len(df)}")
    print(f"  Residual MAE: {res_mae:.3f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
