#!/usr/bin/env python3
"""
mlb_ou_retrain.py — MLB Over/Under total-runs prediction model
═══════════════════════════════════════════════════════════════
Separate model targeting TOTAL RUNS (not margin).
Uses COMBINED features (sums, averages) instead of diffs.

Architecture: Lasso + ElasticNet + CatBoost → averaged ensemble
Target: actual_home_runs + actual_away_runs

Usage:
    python3 mlb_ou_retrain.py                # Train + evaluate
    python3 mlb_ou_retrain.py --upload       # Train + upload to Supabase as 'mlb_ou'
    python3 mlb_ou_retrain.py --refresh      # Re-pull data from Supabase first
"""
import sys, os, time, warnings, pickle, io, base64
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from catboost import CatBoostRegressor

SEED = 42
N_FOLDS = 20

# ── O/U Feature set ──
# Key difference from margin model: COMBINED (sum) features instead of DIFFS
# Environment features are weighted more heavily for totals
OU_FEATURE_COLS = [
    # ── Total offensive quality (sum, not diff) ──
    "woba_combined",          # home_woba + away_woba
    "woba_diff",              # lopsided matchups regress to mean → lower totals
    # ── Total pitching quality ──
    "fip_combined",           # home_starter_fip + away_starter_fip (lower = less runs)
    "fip_diff",               # ace vs ace creates different total than avg vs avg
    "sp_fip_spread",          # matchup lopsidedness
    "bullpen_combined",       # home_bullpen_era + away_bullpen_era
    "k_bb_combined",          # combined strikeout/walk rate
    "sp_ip_combined",         # combined starter innings (more IP = less bullpen exposure)
    "bp_exposure_combined",   # combined bullpen exposure
    # ── Environment (critical for O/U) ──
    "park_factor",            # #2 predictor after market_total
    "temp_f",
    "wind_mph",
    "wind_out",
    "is_warm",
    "is_cold",
    "temp_x_park",            # interaction: hot + hitter park = high scoring
    "lg_rpg",                 # league run environment
    # ── Umpire ──
    "ump_run_env",            # umpire's historical average total runs
    # ── Rolling totals context ──
    "scoring_entropy_combined",  # combined scoring diversity
    "first_inn_rate_combined",   # combined early-aggression tendency
    # ── Market signal (strongest predictor) ──
    "market_total",           # Vegas O/U line
    "has_market",
    # ── Heuristic total ──
    "total_pred",             # heuristic engine's predicted total
    "has_heuristic",
    # ── Context ──
    "rest_combined",          # combined rest days (more rest = sharper pitching?)
    "series_game_num",        # game in series (bullpen fatigue accumulates)
]

# Import the data loader from mlb_retrain
from mlb_retrain import load_data


def build_ou_features(df):
    """Build O/U-specific features: COMBINED (sums) instead of diffs."""
    df = df.copy()

    # ── Ensure raw columns with defaults ──
    raw_defaults = {
        "home_woba": 0.315, "away_woba": 0.315,
        "home_sp_fip": 4.25, "away_sp_fip": 4.25,
        "home_fip": 4.25, "away_fip": 4.25,
        "home_bullpen_era": 4.10, "away_bullpen_era": 4.10,
        "home_k9": 8.5, "away_k9": 8.5,
        "home_bb9": 3.2, "away_bb9": 3.2,
        "home_sp_ip": 5.5, "away_sp_ip": 5.5,
        "park_factor": 1.0, "temp_f": 70.0, "wind_mph": 5.0,
        "wind_out_flag": 0, "home_rest_days": 4.0, "away_rest_days": 4.0,
        "home_lineup_confirmed": 0, "away_lineup_confirmed": 0,
        "home_platoon_delta": 0, "away_platoon_delta": 0,
    }
    for col, default in raw_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # ── SP FIP fallback ──
    if "home_sp_fip_known" in df.columns:
        hk = pd.to_numeric(df["home_sp_fip_known"], errors="coerce").fillna(0).astype(bool)
        ak = pd.to_numeric(df["away_sp_fip_known"], errors="coerce").fillna(0).astype(bool)
        df["home_starter_fip"] = np.where(hk, df["home_sp_fip"], df["home_fip"])
        df["away_starter_fip"] = np.where(ak, df["away_sp_fip"], df["away_fip"])
    else:
        df["home_starter_fip"] = df["home_sp_fip"].where(df["home_sp_fip"] != 4.25, df["home_fip"])
        df["away_starter_fip"] = df["away_sp_fip"].where(df["away_sp_fip"] != 4.25, df["away_fip"])

    # ═══════════════════════════════════════════════════════════
    # COMBINED features (sums — what drives TOTAL runs)
    # ═══════════════════════════════════════════════════════════
    df["woba_combined"] = df["home_woba"] + df["away_woba"]
    df["woba_diff"] = df["home_woba"] - df["away_woba"]
    df["fip_combined"] = df["home_starter_fip"] + df["away_starter_fip"]
    df["fip_diff"] = df["home_starter_fip"] - df["away_starter_fip"]
    df["sp_fip_spread"] = (df["home_starter_fip"] - df["away_starter_fip"]).abs()
    df["bullpen_combined"] = df["home_bullpen_era"] + df["away_bullpen_era"]

    # K-BB combined
    df["k_bb_home"] = df["home_k9"] - df["home_bb9"]
    df["k_bb_away"] = df["away_k9"] - df["away_bb9"]
    df["k_bb_combined"] = df["k_bb_home"] + df["k_bb_away"]

    # SP innings + bullpen exposure (combined)
    df["sp_ip_combined"] = df["home_sp_ip"] + df["away_sp_ip"]
    df["home_bp_exposure"] = np.maximum(0, 5.5 - df["home_sp_ip"]) * (df["home_bullpen_era"] / 4.10)
    df["away_bp_exposure"] = np.maximum(0, 5.5 - df["away_sp_ip"]) * (df["away_bullpen_era"] / 4.10)
    df["bp_exposure_combined"] = df["home_bp_exposure"] + df["away_bp_exposure"]

    # ── Environment ──
    df["wind_out"] = df["wind_out_flag"].astype(int)
    df["is_warm"] = (df["temp_f"] > 75).astype(int)
    df["is_cold"] = (df["temp_f"] < 45).astype(int)
    df["temp_x_park"] = ((df["temp_f"] - 70) / 30.0) * df["park_factor"]

    # ── League RPG ──
    SEASON_RPG = {
        2015: 4.25, 2016: 4.48, 2017: 4.65, 2018: 4.45, 2019: 4.83,
        2022: 4.28, 2023: 4.62, 2024: 4.38, 2025: 4.30, 2026: 4.30,
    }
    if "season" in df.columns:
        df["lg_rpg"] = df["season"].map(lambda s: SEASON_RPG.get(int(s), 4.30) if pd.notna(s) else 4.30)
    else:
        game_year = pd.to_datetime(df["game_date"]).dt.year
        df["lg_rpg"] = game_year.map(lambda y: SEASON_RPG.get(y, 4.30))

    # ── Heuristic total ──
    for col, default in [("pred_home_runs", 0.0), ("pred_away_runs", 0.0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default
    df["total_pred"] = df["pred_home_runs"] + df["pred_away_runs"]
    df["has_heuristic"] = (df["total_pred"] > 0).astype(int)

    # ── Market total ──
    df["market_total"] = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else pd.Series(0, index=df.index),
        errors="coerce"
    ).fillna(0)
    df["has_market"] = (df["market_total"] > 0).astype(int)

    # ── Context ──
    df["rest_combined"] = df["home_rest_days"] + df["away_rest_days"]

    # ── Advanced rolling (from backfill) ──
    for col, default in [("scoring_entropy_combined", 5.0),
                         ("first_inn_rate_combined", 0.8),
                         ("ump_run_env", 8.5),
                         ("series_game_num", 1.0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    return df[OU_FEATURE_COLS].fillna(0)


def walk_forward_ou(X, y, market_total, weights, n_folds=N_FOLDS):
    """Walk-forward validation targeting total runs."""
    n = len(X)
    fold_size = n // n_folds
    preds = np.full(n, np.nan)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        if fold == n_folds - 1:
            test_end = n

        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) < 200:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = weights[train_idx] if weights is not None else None

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        # 3-model ensemble
        m1 = Lasso(alpha=0.01, random_state=SEED, max_iter=5000)
        m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=5000)
        m3 = CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.03,
            subsample=0.8, random_seed=SEED, verbose=0,
        )

        m1.fit(X_tr_s, y_train, sample_weight=w_train)
        m2.fit(X_tr_s, y_train, sample_weight=w_train)
        m3.fit(X_tr_s, y_train, sample_weight=w_train)

        p1 = m1.predict(X_te_s)
        p2 = m2.predict(X_te_s)
        p3 = m3.predict(X_te_s)
        preds[test_idx] = (p1 + p2 + p3) / 3

        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{n_folds}")

    return preds


def evaluate_ou(preds, y_total, market_total):
    """Evaluate O/U accuracy at various edge thresholds."""
    valid = ~np.isnan(preds) & (market_total > 0)
    pv = preds[valid]
    tv = y_total[valid]
    mt = market_total[valid]

    mae = np.mean(np.abs(pv - tv))
    print(f"\n  Walk-forward MAE: {mae:.3f} runs")
    print(f"  Market MAE:       {np.mean(np.abs(mt - tv)):.3f} runs")
    print(f"  Bias:             {np.mean(pv - tv):+.3f}")

    # ── O/U accuracy at various edge thresholds ──
    print(f"\n  {'='*65}")
    print(f"  O/U ACCURACY BY EDGE THRESHOLD")
    print(f"  {'='*65}")
    print(f"  {'Edge':>6} {'Games':>7} {'Correct':>8} {'Acc%':>7} {'ROI%':>7} {'Verdict':<10}")
    print(f"  {'─'*6} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*10}")

    thresholds = {}
    for edge in [0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
        model_over = pv > (mt + edge)
        model_under = pv < (mt - edge)
        has_pick = model_over | model_under

        actual_over = tv > mt
        actual_under = tv < mt
        push = tv == mt

        correct = (model_over & actual_over) | (model_under & actual_under)
        decided = has_pick & ~push

        if decided.sum() < 20:
            continue

        acc = correct[decided].sum() / decided.sum()
        roi = (acc * 1.91 - 1) * 100  # -110 odds
        n_games = decided.sum()

        # Directional split
        over_picks = (model_over & ~push).sum()
        under_picks = (model_under & ~push).sum()
        over_correct = (model_over & actual_over & ~push).sum()
        under_correct = (model_under & actual_under & ~push).sum()
        over_acc = over_correct / over_picks if over_picks > 0 else 0
        under_acc = under_correct / under_picks if under_picks > 0 else 0

        verdict = "🔥 STRONG" if acc >= 0.55 else ("✅ PROFIT" if roi > 0 else "❌ LOSS")
        print(f"  {edge:>5.1f}+ {n_games:>7d} {correct[decided].sum():>8.0f} "
              f"{acc*100:>6.1f}% {roi:>+6.1f}% {verdict}")
        print(f"         OVER: {over_acc:.1%} ({over_picks}g)  UNDER: {under_acc:.1%} ({under_picks}g)")

        thresholds[f"ou_{edge}+"] = {
            "games": int(n_games), "acc": round(acc * 100, 2),
            "roi": round(roi, 2),
            "over_acc": round(over_acc * 100, 1), "under_acc": round(under_acc * 100, 1),
        }

    return mae, thresholds


def main():
    upload = "--upload" in sys.argv

    print("=" * 70)
    print("  MLB O/U MODEL — TOTAL RUNS PREDICTOR")
    print("  Lasso + ElasticNet + CatBoost ensemble")
    print(f"  {len(OU_FEATURE_COLS)} features (combined/environment-focused)")
    print("=" * 70)

    # ── Load data ──
    df = load_data(refresh="--refresh" in sys.argv)

    # Target: total runs
    df["target_total"] = df["actual_home_runs"].astype(float) + df["actual_away_runs"].astype(float)
    y = df["target_total"].values
    weights = df["season_weight"].values if "season_weight" in df.columns else None

    # Build O/U features
    X_df = build_ou_features(df)
    X = X_df.values
    feature_cols = list(X_df.columns)

    # Market total for evaluation
    market_total = pd.to_numeric(df.get("market_ou_total", pd.Series(0, index=df.index)),
                                  errors="coerce").fillna(0).values

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Games: {len(df)}")
    print(f"  Games with market O/U: {(market_total > 0).sum()}")
    print(f"  Total runs mean: {y.mean():.2f}, std: {y.std():.2f}")

    # ── Walk-forward validation ──
    print(f"\n  Walk-forward validation ({N_FOLDS}-fold ensemble)...")
    t0 = time.time()
    preds = walk_forward_ou(X, y, market_total, weights, N_FOLDS)
    elapsed = time.time() - t0
    print(f"  Complete in {elapsed:.0f}s")

    mae, thresholds = evaluate_ou(preds, y, market_total)

    # ── Train production model ──
    print(f"\n  Training production ensemble on {len(df)} games...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    m1 = Lasso(alpha=0.01, random_state=SEED, max_iter=5000)
    m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=5000)
    m3 = CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.03,
        subsample=0.8, random_seed=SEED, verbose=0,
    )

    m1.fit(X_scaled, y, sample_weight=weights)
    m2.fit(X_scaled, y, sample_weight=weights)
    m3.fit(X_scaled, y, sample_weight=weights)

    in_sample = (m1.predict(X_scaled) + m2.predict(X_scaled) + m3.predict(X_scaled)) / 3
    is_mae = np.mean(np.abs(in_sample - y))
    print(f"  In-sample MAE: {is_mae:.4f}")

    # Lasso feature selection insight
    n_kept = np.sum(np.abs(m1.coef_) > 1e-6)
    print(f"  Lasso kept: {n_kept}/{len(feature_cols)} features")
    dropped = [f for f, c in zip(feature_cols, m1.coef_) if abs(c) < 1e-6]
    if dropped:
        print(f"  Dropped: {', '.join(dropped)}")

    # Bias correction
    valid = ~np.isnan(preds)
    bias = float(np.mean(preds[valid] - y[valid])) if valid.sum() > 100 else 0.0
    print(f"  Bias correction: {bias:+.3f}")

    # SHAP
    import shap
    print("  Building SHAP explainer...")
    explainer = shap.TreeExplainer(m3)
    print("  ✅ TreeExplainer built")

    # ── Save bundle ──
    from datetime import datetime
    bundle = {
        "scaler": scaler,
        "_ensemble_models": [m1, m2, m3],
        "explainer": explainer,
        "feature_cols": feature_cols,
        "n_train": len(df),
        "mae_cv": round(mae, 4),
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": "OU_Ensemble_v1_Lasso+EN+CatBoost",
        "bias_correction": round(bias, 4),
        "thresholds": thresholds,
    }

    pkl_path = "mlb_ou_model_v1.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    size_kb = os.path.getsize(pkl_path) / 1024
    print(f"\n  Saved: {pkl_path} ({size_kb:.0f} KB)")

    if upload:
        print(f"\n  Uploading to Supabase as 'mlb_ou'...")
        from db import save_model, load_model
        save_model("mlb_ou", bundle)

        # Verify reload
        try:
            test = load_model("mlb_ou")
            if test:
                print(f"  Railway reload: {{'features': {len(test['feature_cols'])}, "
                      f"'mae': {test['mae_cv']}, 'status': 'ok'}}")
            else:
                print("  ⚠️ Reload verification failed")
        except Exception as e:
            print(f"  ⚠️ Reload check: {e}")

    print(f"\n{'='*70}")
    print(f"  MLB O/U MODEL COMPLETE")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Walk-forward MAE: {mae:.3f}")
    print(f"  Bias: {bias:+.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
