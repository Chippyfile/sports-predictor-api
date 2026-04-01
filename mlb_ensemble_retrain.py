#!/usr/bin/env python3
"""
mlb_ensemble_retrain.py — Production MLB v8 ensemble
═══════════════════════════════════════════════════════
Architecture: Lasso_0.01 + ElasticNet_a0.1_r0.5 + CatBoost_d6_i200_lr0.03
Features: 32 (trimmed from 41 — dropped 9 constants/noise)
Includes: ATS + O/U threshold analysis at various edge levels

Usage:
  python3 mlb_ensemble_retrain.py               # Train + evaluate
  python3 mlb_ensemble_retrain.py --upload      # Train + upload to Supabase
"""
import sys, os, time, warnings, pickle, io, base64
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from catboost import CatBoostRegressor

# ── Feature set (32 — trimmed from 41) ──
FEATURE_COLS = [
    # Core offensive/pitching
    "woba_diff", "fip_diff", "k_bb_diff", "bullpen_era_diff",
    "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff", "sp_fip_spread",
    "sp_relative_fip_diff",
    # Park & environment
    "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold",
    "temp_x_park",
    # Context
    "rest_diff", "run_diff_pred",
    # Market
    "market_spread", "spread_vs_market",
    # Interactions
    "woba_x_park",
    # Platoon
    "platoon_diff",
    # Advanced rolling (v7)
    "pyth_residual_diff", "babip_luck_diff", "scoring_entropy_diff",
    "first_inn_rate_diff", "clutch_divergence_diff", "opp_adj_form_diff",
    "ump_run_env", "series_game_num",
    "scoring_entropy_combined", "first_inn_rate_combined",
]

SEED = 42
N_FOLDS = 20

# Import the feature builder from mlb_retrain (uses the full 41 — we subset)
from mlb_retrain import build_features as build_41_features, load_data


def build_features_32(df):
    """Build 41 features then subset to 32."""
    full = build_41_features(df)
    return full[FEATURE_COLS]


def walk_forward_ensemble(X, y, weights, n_folds=N_FOLDS):
    """Walk-forward with the 3-model ensemble. Returns predictions array + metrics."""
    fold_size = len(X) // (n_folds + 3)
    min_train = fold_size * 3
    
    preds = np.full(len(X), np.nan)
    
    for fold in range(n_folds):
        te_s = min_train + fold * fold_size
        te_e = min(te_s + fold_size, len(X))
        if te_s >= len(X):
            break
        
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:te_s])
        Xte = sc.transform(X[te_s:te_e])
        wt = weights[:te_s] if weights is not None else None
        
        m1 = Lasso(alpha=0.01, max_iter=5000)
        m1.fit(Xtr, y[:te_s])
        p1 = m1.predict(Xte)
        
        m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        m2.fit(Xtr, y[:te_s])
        p2 = m2.predict(Xte)
        
        m3 = CatBoostRegressor(
            depth=6, iterations=200, learning_rate=0.03,
            subsample=0.8, min_data_in_leaf=20,
            random_seed=SEED, verbose=0,
        )
        m3.fit(Xtr, y[:te_s], sample_weight=wt)
        p3 = m3.predict(Xte)
        
        preds[te_s:te_e] = (p1 + p2 + p3) / 3.0
        
        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{n_folds}")
    
    return preds


def threshold_analysis(preds, y, market_spread, market_total, actual_total):
    """Compute accuracy at various edge thresholds for ATS and O/U."""
    valid = ~np.isnan(preds)
    pv, tv = preds[valid], y[valid]
    ms = market_spread[valid]
    mt = market_total[valid]
    at = actual_total[valid]
    
    # ══════════════════════════════════════
    # RUN LINE (ATS) ANALYSIS
    # ══════════════════════════════════════
    # Model margin vs market spread: edge = model_margin - (-market_spread)
    # Positive edge = model thinks home covers more than market expects
    has_rl = ms != 0
    model_margin = pv[has_rl]
    mkt_implied = -ms[has_rl]  # market_spread_home is negative for favorites
    actual_margin = tv[has_rl]
    edge = np.abs(model_margin - mkt_implied)
    
    # ATS correct: model side vs actual side relative to spread
    model_side = model_margin - mkt_implied  # positive = bet home, negative = bet away
    actual_side = actual_margin - mkt_implied  # positive = home covered, negative = away covered
    ats_correct = (model_side > 0) == (actual_side > 0)
    # Exclude pushes
    not_push = actual_side != 0
    
    print(f"\n  {'='*65}")
    print(f"  RUN LINE (ATS) ACCURACY BY EDGE THRESHOLD")
    print(f"  {'='*65}")
    print(f"  {'Edge':>6} {'Games':>7} {'Correct':>8} {'Acc%':>7} {'ROI%':>7} {'Verdict':<10}")
    print(f"  {'─'*6} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*10}")
    
    rl_thresholds = {}
    for thresh in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        mask = (edge >= thresh) & not_push
        if mask.sum() < 20:
            continue
        correct = ats_correct[mask].sum()
        total = mask.sum()
        acc = correct / total * 100
        # ROI at -110: profit = wins * 100 - losses * 110
        roi = (correct * 100 - (total - correct) * 110) / (total * 110) * 100
        verdict = "✅ PROFIT" if roi > 0 else "❌ LOSS"
        if acc >= 55:
            verdict = "🔥 STRONG"
        print(f"  {thresh:>5.1f}+ {total:>7d} {correct:>8d} {acc:>6.1f}% {roi:>+6.1f}% {verdict}")
        rl_thresholds[f"rl_{thresh}+"] = {"games": int(total), "acc": round(acc, 2), "roi": round(roi, 2)}
    
    # ══════════════════════════════════════
    # O/U ANALYSIS
    # ══════════════════════════════════════
    # Model predicts margin (home - away). Total = sum of scores.
    # For O/U: we need predicted total. Estimate: predicted_total ≈ league_avg + adjustments
    # Since we don't have explicit total prediction, use: actual_margin + 2*avg_score proxy
    # Better: use the model's predicted runs if available
    # For now: O/U edge = |predicted_total_proxy - market_total|
    # Predicted total proxy: from training data, use avg total + margin signal
    
    has_ou = mt > 0
    if has_ou.sum() > 100:
        # Simple O/U proxy: use home_runs + away_runs predictions from heuristic
        # Or: market total + model's deviation signal
        # The margin model tells us WHO wins, not total. But some features are O/U-relevant.
        # Use market total as base, adjust by scoring entropy and first-inning signals
        
        # For now: check if model margin direction helps O/U at all
        # High-scoring games correlate with larger absolute margins? Let's test differently.
        
        # Direct O/U: actual total vs market total
        actual_t = at[has_ou]
        market_t = mt[has_ou]
        
        print(f"\n  {'='*65}")
        print(f"  OVER/UNDER ANALYSIS (market total available)")
        print(f"  {'='*65}")
        print(f"  Games with market O/U: {has_ou.sum()}")
        print(f"  Market total mean: {market_t.mean():.1f}")
        print(f"  Actual total mean: {actual_t.mean():.1f}")
        
        # Check if any features predict O/U direction
        # Simple test: does scoring_entropy_combined predict over/under?
        went_over = actual_t > market_t
        went_under = actual_t < market_t
        base_over_rate = went_over.sum() / (went_over.sum() + went_under.sum()) * 100
        print(f"  Base over rate: {base_over_rate:.1f}%")
        
        print(f"\n  Note: Current model predicts MARGIN (ATS), not TOTAL (O/U).")
        print(f"  For O/U, a separate model targeting total runs is recommended.")
        print(f"  O/U features available: scoring_entropy_combined, first_inn_rate_combined,")
        print(f"  ump_run_env, temp_f, wind_mph, park_factor, temp_x_park")
    
    # ══════════════════════════════════════
    # MONEYLINE (WIN) ACCURACY
    # ══════════════════════════════════════
    print(f"\n  {'='*65}")
    print(f"  MONEYLINE (WIN) ACCURACY BY PROBABILITY THRESHOLD")
    print(f"  {'='*65}")
    
    # Convert margin to probability: P(home) = 1/(1+10^(-margin/σ))
    from scipy.stats import norm as _norm
    SIGMA = 4.0
    win_prob = _norm.cdf(pv / SIGMA)
    home_won = tv > 0
    
    print(f"  {'Prob':>7} {'Side':>6} {'Games':>7} {'Correct':>8} {'Acc%':>7}")
    print(f"  {'─'*7} {'─'*6} {'─'*7} {'─'*8} {'─'*7}")
    
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        # Home favorites above threshold
        mask_h = win_prob >= thresh
        if mask_h.sum() >= 20:
            correct_h = home_won[mask_h].sum()
            acc_h = correct_h / mask_h.sum() * 100
            print(f"  ≥{thresh:.0%} {'HOME':>6} {mask_h.sum():>7d} {correct_h:>8.0f} {acc_h:>6.1f}%")
        
        # Away favorites (1-prob >= threshold)
        mask_a = (1 - win_prob) >= thresh
        if mask_a.sum() >= 20:
            correct_a = (~home_won[mask_a]).sum()
            acc_a = correct_a / mask_a.sum() * 100
            print(f"  ≥{thresh:.0%} {'AWAY':>6} {mask_a.sum():>7d} {correct_a:>8.0f} {acc_a:>6.1f}%")
    
    return rl_thresholds


# No custom class needed — store models as list, average at predict time


def main():
    upload = "--upload" in sys.argv
    
    print("=" * 70)
    print("  MLB v8 ENSEMBLE RETRAIN")
    print("  Lasso_0.01 + ElasticNet_a0.1_r0.5 + CatBoost_d6_i200_lr0.03")
    print("  32 features (trimmed from 41)")
    print("=" * 70)
    
    # ── Load data ──
    df = load_data()
    y = df["target_margin"].values
    weights = df["season_weight"].values if "season_weight" in df.columns else None
    
    X_df = build_features_32(df)
    X = X_df.values
    feature_names = list(X_df.columns)
    print(f"\n  Features: {len(feature_names)}")
    print(f"  Games: {len(X)}")
    
    # ── Walk-forward validation ──
    print(f"\n  Walk-forward validation ({N_FOLDS}-fold ensemble)...")
    t0 = time.time()
    wf_preds = walk_forward_ensemble(X, y, weights)
    print(f"  Complete in {time.time()-t0:.0f}s")
    
    valid = ~np.isnan(wf_preds)
    pv, tv = wf_preds[valid], y[valid]
    mae = float(np.mean(np.abs(pv - tv)))
    bias = float(np.mean(tv - pv))
    win_correct = ((pv > 0) == (tv > 0)).sum()
    win_total = (tv != 0).sum()
    win_acc = win_correct / win_total * 100
    
    print(f"\n  Walk-forward MAE: {mae:.4f}")
    print(f"  Win accuracy: {win_acc:.2f}%")
    print(f"  Bias: {bias:+.4f}")
    
    # ── Threshold analysis ──
    # Get market spread and O/U data aligned with walk-forward predictions
    market_spread = np.zeros(len(y))
    market_total = np.zeros(len(y))
    actual_total = np.zeros(len(y))
    
    full_df = build_41_features(df)
    if "market_spread" in full_df.columns:
        market_spread = full_df["market_spread"].values
    
    # Market total from raw data
    mt_raw = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0).values
    market_total = mt_raw
    
    # Actual total
    actual_total = (
        pd.to_numeric(df["actual_home_runs"], errors="coerce").fillna(0) +
        pd.to_numeric(df["actual_away_runs"], errors="coerce").fillna(0)
    ).values
    
    rl_thresholds = threshold_analysis(wf_preds, y, market_spread, market_total, actual_total)
    
    # ── Train production models ──
    print(f"\n  Training production ensemble on {len(X)} games...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    m1 = Lasso(alpha=0.01, max_iter=5000)
    m1.fit(X_s, y)
    
    m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
    m2.fit(X_s, y)
    
    m3 = CatBoostRegressor(
        depth=6, iterations=200, learning_rate=0.03,
        subsample=0.8, min_data_in_leaf=20,
        random_seed=SEED, verbose=0,
    )
    m3.fit(X_s, y, sample_weight=weights)
    
    # Average predictions from all 3 models
    in_preds = (m1.predict(X_s) + m2.predict(X_s) + m3.predict(X_s)) / 3.0
    in_mae = float(np.mean(np.abs(in_preds - y)))
    print(f"  In-sample MAE: {in_mae:.4f}")
    
    # Lasso feature selection info
    lasso_kept = sum(1 for c in m1.coef_ if c != 0)
    print(f"  Lasso kept: {lasso_kept}/{len(feature_names)} features")
    
    # SHAP from CatBoost
    print("  Building SHAP explainer (CatBoost)...")
    try:
        import shap
        explainer = shap.TreeExplainer(m3)
        print("  ✅ TreeExplainer built")
    except Exception as e:
        explainer = None
        print(f"  ⚠️ SHAP failed: {e}")
    
    # ── Build bundle ──
    # Use CatBoost as "reg" (compatible with existing predict path)
    # Store all 3 models in "_ensemble_models" for averaging at predict time
    bundle = {
        "reg": m3,  # CatBoost as primary (for SHAP + fallback)
        "scaler": scaler,
        "feature_cols": feature_names,
        "explainer": explainer,
        "bias_correction": bias,
        "cv_mae": round(mae, 4),
        "n_train": len(X),
        "n_historical": len(df),
        "n_current": 0,
        "model_type": "Ensemble_v8_Lasso+EN+CatBoost",
        "architecture": "Lasso_0.01 + ElasticNet_a0.1_r0.5 + CatBoost_d6_i200_lr0.03",
        "trained_at": pd.Timestamp.now().isoformat(),
        "mae_cv": round(mae, 4),
        "rl_thresholds": rl_thresholds,
        "clf": None,
        "isotonic": None,
        # Ensemble: predict path averages all 3
        "_ensemble_models": [m1, m2, m3],
    }
    
    local_path = "mlb_model_v8.pkl"
    with open(local_path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    sz = os.path.getsize(local_path)
    print(f"\n  Saved: {local_path} ({sz // 1024} KB)")
    
    if upload:
        print("\n  Uploading to Supabase as 'mlb'...")
        import requests
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        
        buf = io.BytesIO()
        pickle.dump(bundle, buf, protocol=4)
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        meta = {
            "mae_cv": bundle["cv_mae"],
            "n_train": bundle["n_train"],
            "model_type": bundle["model_type"],
            "n_features": len(feature_names),
            "size_bytes": len(buf.getvalue()),
            "trained_at": bundle["trained_at"],
            "rl_thresholds": rl_thresholds,
        }
        
        headers = {
            "apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        resp = requests.post(
            f"{url}/rest/v1/model_store",
            headers=headers,
            json={"name": "mlb", "data": b64, "metadata": meta},
        )
        if resp.status_code < 300:
            print(f"  ✅ Uploaded ({len(buf.getvalue()) // 1024} KB)")
        else:
            print(f"  ❌ Upload failed: {resp.status_code} {resp.text[:200]}")
        
        try:
            r = requests.post("https://sports-predictor-api-production.up.railway.app/debug/reload-model/mlb", timeout=30)
            print(f"  Railway reload: {r.json()}")
        except Exception as e:
            print(f"  Railway reload: {e}")
    
    print(f"\n{'='*70}")
    print(f"  MLB v8 ENSEMBLE COMPLETE")
    print(f"  Architecture: Lasso + ElasticNet + CatBoost")
    print(f"  Features: {len(feature_names)}")
    print(f"  Walk-forward MAE: {mae:.4f}")
    print(f"  Win accuracy: {win_acc:.2f}%")
    print(f"  Bias: {bias:+.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
