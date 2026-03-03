#!/usr/bin/env python3
"""
sp_fip_spread Isolation Backtest
================================
Compares 22-feature model (baseline) vs 25-feature model (with sp_fip_spread,
platoon_diff, both_lineups_confirmed) on identical walk-forward folds.

Since platoon_diff and both_lineups_confirmed are zero in historical data,
the ONLY difference is sp_fip_spread — this isolates its impact.

Run from your sports-predictor-api directory:
    python3 sp_fip_spread_backtest.py

Requires: Railway environment with Supabase access, or run on Railway directly.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, brier_score_loss

# ── Supabase client ──
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_KEY) env vars")
    sys.exit(1)

import requests

def sb_get(table, query=""):
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    all_data = []
    offset = 0
    limit = 1000
    while True:
        headers["Range"] = f"{offset}-{offset + limit - 1}"
        url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
        r = requests.get(url, headers=headers, timeout=30)
        if not r.ok:
            print(f"  Supabase error: {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
        print(f"  Fetched {len(all_data)} rows so far...")
    return all_data


# ── Season constants (same as main.py) ──
SEASON_CONSTANTS = {
    2015: {"lg_rpg": 4.25, "lg_woba": 0.313, "woba_scale": 1.24, "lg_fip": 3.97},
    2016: {"lg_rpg": 4.48, "lg_woba": 0.318, "woba_scale": 1.23, "lg_fip": 4.19},
    2017: {"lg_rpg": 4.65, "lg_woba": 0.321, "woba_scale": 1.24, "lg_fip": 4.36},
    2018: {"lg_rpg": 4.45, "lg_woba": 0.315, "woba_scale": 1.23, "lg_fip": 4.15},
    2019: {"lg_rpg": 4.83, "lg_woba": 0.320, "woba_scale": 1.17, "lg_fip": 4.51},
    2021: {"lg_rpg": 4.53, "lg_woba": 0.313, "woba_scale": 1.21, "lg_fip": 4.27},
    2022: {"lg_rpg": 4.28, "lg_woba": 0.310, "woba_scale": 1.24, "lg_fip": 4.05},
    2023: {"lg_rpg": 4.59, "lg_woba": 0.318, "woba_scale": 1.24, "lg_fip": 4.33},
    2024: {"lg_rpg": 4.38, "lg_woba": 0.317, "woba_scale": 1.25, "lg_fip": 4.17},
    2025: {"lg_rpg": 4.38, "lg_woba": 0.317, "woba_scale": 1.25, "lg_fip": 4.17},
}
DEFAULT_CONSTANTS = SEASON_CONSTANTS[2024]


def build_features_baseline(df):
    """22-feature model (no sp_fip_spread, platoon_diff, both_lineups_confirmed)."""
    df = df.copy()
    raw_cols = {
        "home_woba": 0.314, "away_woba": 0.314,
        "home_sp_fip": 4.25, "away_sp_fip": 4.25,
        "home_fip": 4.25, "away_fip": 4.25,
        "home_bullpen_era": 4.10, "away_bullpen_era": 4.10,
        "park_factor": 1.00, "temp_f": 70.0, "wind_mph": 5.0, "wind_out_flag": 0.0,
        "home_rest_days": 4.0, "away_rest_days": 4.0,
        "home_travel": 0.0, "away_travel": 0.0,
        "home_k9": 8.5, "away_k9": 8.5, "home_bb9": 3.2, "away_bb9": 3.2,
        "home_sp_ip": 5.5, "away_sp_ip": 5.5,
        "home_def_oaa": 0.0, "away_def_oaa": 0.0,
    }
    for col, default in raw_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # SP FIP fallback
    df["home_starter_fip"] = df["home_sp_fip"].where(df["home_sp_fip"] != 4.25, df["home_fip"])
    df["away_starter_fip"] = df["away_sp_fip"].where(df["away_sp_fip"] != 4.25, df["away_fip"])
    df["has_sp_fip"] = ((df["home_sp_fip"] != 4.25) & (df["away_sp_fip"] != 4.25)).astype(int)

    df["woba_diff"] = df["home_woba"] - df["away_woba"]
    df["fip_diff"] = df["home_starter_fip"] - df["away_starter_fip"]
    df["bullpen_era_diff"] = df["home_bullpen_era"] - df["away_bullpen_era"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["travel_diff"] = df["home_travel"] - df["away_travel"]
    df["is_warm"] = (df["temp_f"] > 75).astype(int)
    df["is_cold"] = (df["temp_f"] < 45).astype(int)
    df["wind_out"] = df["wind_out_flag"].astype(int)
    df["k_bb_diff"] = (df["home_k9"] - df["home_bb9"]) - (df["away_k9"] - df["away_bb9"])
    df["sp_ip_diff"] = df["home_sp_ip"] - df["away_sp_ip"]
    df["home_bp_exposure"] = np.maximum(0, 9.0 - df["home_sp_ip"]) * (df["home_bullpen_era"] / 4.10)
    df["away_bp_exposure"] = np.maximum(0, 9.0 - df["away_sp_ip"]) * (df["away_bullpen_era"] / 4.10)
    df["bp_exposure_diff"] = df["home_bp_exposure"] - df["away_bp_exposure"]
    df["def_oaa_diff"] = df["home_def_oaa"] - df["away_def_oaa"]
    df["fip_x_bullpen"] = df["fip_diff"] * df["bullpen_era_diff"]
    df["woba_x_park"] = df["woba_diff"] * df["park_factor"]
    df["wind_x_fip"] = df["wind_out"].astype(float) * df["fip_diff"]

    if "season" in df.columns:
        df["lg_rpg"] = df["season"].map(
            lambda s: SEASON_CONSTANTS.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
            if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
        )
    else:
        df["lg_rpg"] = DEFAULT_CONSTANTS["lg_rpg"]

    for col, default in [("pred_home_runs", 0.0), ("pred_away_runs", 0.0),
                         ("win_pct_home", 0.5), ("ou_total", 9.0), ("model_ml_home", 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    df["run_diff_pred"] = df["pred_home_runs"] - df["pred_away_runs"]
    df["has_heuristic"] = (df["pred_home_runs"] + df["pred_away_runs"] > 0).astype(int)

    feature_cols = [
        "woba_diff", "fip_diff", "has_sp_fip", "bullpen_era_diff", "k_bb_diff",
        "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff",
        "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold",
        "rest_diff", "travel_diff", "lg_rpg",
        "fip_x_bullpen", "woba_x_park", "wind_x_fip",
        "run_diff_pred", "has_heuristic",
    ]
    return df[feature_cols].fillna(0), feature_cols


def build_features_enhanced(df):
    """25-feature model (adds sp_fip_spread, platoon_diff, both_lineups_confirmed)."""
    df = df.copy()
    raw_cols = {
        "home_woba": 0.314, "away_woba": 0.314,
        "home_sp_fip": 4.25, "away_sp_fip": 4.25,
        "home_fip": 4.25, "away_fip": 4.25,
        "home_bullpen_era": 4.10, "away_bullpen_era": 4.10,
        "park_factor": 1.00, "temp_f": 70.0, "wind_mph": 5.0, "wind_out_flag": 0.0,
        "home_rest_days": 4.0, "away_rest_days": 4.0,
        "home_travel": 0.0, "away_travel": 0.0,
        "home_k9": 8.5, "away_k9": 8.5, "home_bb9": 3.2, "away_bb9": 3.2,
        "home_sp_ip": 5.5, "away_sp_ip": 5.5,
        "home_def_oaa": 0.0, "away_def_oaa": 0.0,
        "home_platoon_delta": 0.0, "away_platoon_delta": 0.0,
        "home_lineup_confirmed": 0.0, "away_lineup_confirmed": 0.0,
    }
    for col, default in raw_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    df["home_starter_fip"] = df["home_sp_fip"].where(df["home_sp_fip"] != 4.25, df["home_fip"])
    df["away_starter_fip"] = df["away_sp_fip"].where(df["away_sp_fip"] != 4.25, df["away_fip"])
    df["has_sp_fip"] = ((df["home_sp_fip"] != 4.25) & (df["away_sp_fip"] != 4.25)).astype(int)

    df["woba_diff"] = df["home_woba"] - df["away_woba"]
    df["fip_diff"] = df["home_starter_fip"] - df["away_starter_fip"]
    df["bullpen_era_diff"] = df["home_bullpen_era"] - df["away_bullpen_era"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["travel_diff"] = df["home_travel"] - df["away_travel"]
    df["is_warm"] = (df["temp_f"] > 75).astype(int)
    df["is_cold"] = (df["temp_f"] < 45).astype(int)
    df["wind_out"] = df["wind_out_flag"].astype(int)
    df["k_bb_diff"] = (df["home_k9"] - df["home_bb9"]) - (df["away_k9"] - df["away_bb9"])
    df["sp_ip_diff"] = df["home_sp_ip"] - df["away_sp_ip"]
    df["home_bp_exposure"] = np.maximum(0, 9.0 - df["home_sp_ip"]) * (df["home_bullpen_era"] / 4.10)
    df["away_bp_exposure"] = np.maximum(0, 9.0 - df["away_sp_ip"]) * (df["away_bullpen_era"] / 4.10)
    df["bp_exposure_diff"] = df["home_bp_exposure"] - df["away_bp_exposure"]
    df["def_oaa_diff"] = df["home_def_oaa"] - df["away_def_oaa"]
    df["fip_x_bullpen"] = df["fip_diff"] * df["bullpen_era_diff"]
    df["woba_x_park"] = df["woba_diff"] * df["park_factor"]
    df["wind_x_fip"] = df["wind_out"].astype(float) * df["fip_diff"]

    # NEW: Enhancement features
    df["platoon_diff"] = df["home_platoon_delta"] - df["away_platoon_delta"]
    df["sp_fip_spread"] = (df["home_starter_fip"] - df["away_starter_fip"]).abs()
    df["both_lineups_confirmed"] = (
        (df["home_lineup_confirmed"] == 1) & (df["away_lineup_confirmed"] == 1)
    ).astype(int)

    if "season" in df.columns:
        df["lg_rpg"] = df["season"].map(
            lambda s: SEASON_CONSTANTS.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
            if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
        )
    else:
        df["lg_rpg"] = DEFAULT_CONSTANTS["lg_rpg"]

    for col, default in [("pred_home_runs", 0.0), ("pred_away_runs", 0.0),
                         ("win_pct_home", 0.5), ("ou_total", 9.0), ("model_ml_home", 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    df["run_diff_pred"] = df["pred_home_runs"] - df["pred_away_runs"]
    df["has_heuristic"] = (df["pred_home_runs"] + df["pred_away_runs"] > 0).astype(int)

    feature_cols = [
        "woba_diff", "fip_diff", "has_sp_fip", "bullpen_era_diff", "k_bb_diff",
        "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff",
        "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold",
        "rest_diff", "travel_diff", "lg_rpg",
        "fip_x_bullpen", "woba_x_park", "wind_x_fip",
        "run_diff_pred", "has_heuristic",
        "platoon_diff", "sp_fip_spread", "both_lineups_confirmed",
    ]
    return df[feature_cols].fillna(0), feature_cols


def train_and_eval(X_train_s, X_test_s, y_train_margin, y_test_margin, y_test_win, weights, label):
    """Train stacked ensemble, return metrics."""
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08,
                                     subsample=0.8, min_samples_leaf=25, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=20,
                                    max_features=0.7, random_state=42, n_jobs=1)
    ridge = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=3)

    gbm.fit(X_train_s, y_train_margin, sample_weight=weights)
    rf_reg.fit(X_train_s, y_train_margin, sample_weight=weights)
    ridge.fit(X_train_s, y_train_margin, sample_weight=weights)

    meta_X_train = np.column_stack([gbm.predict(X_train_s), rf_reg.predict(X_train_s), ridge.predict(X_train_s)])
    meta_reg = Ridge(alpha=1.0)
    meta_reg.fit(meta_X_train, y_train_margin)

    meta_X_test = np.column_stack([gbm.predict(X_test_s), rf_reg.predict(X_test_s), ridge.predict(X_test_s)])
    pred_margin = meta_reg.predict(meta_X_test)

    # Win prob from logistic regression on margin
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train_s, (y_train_margin > 0).astype(int), sample_weight=weights)
    win_prob = clf.predict_proba(X_test_s)[:, 1]

    # Metrics
    ml_correct = ((pred_margin > 0) == (y_test_margin > 0))
    ml_acc = ml_correct.mean()
    mae = mean_absolute_error(y_test_margin, pred_margin)
    brier = brier_score_loss(y_test_win, win_prob)

    # Confidence tiers based on win_prob distance from 0.5
    conf = np.abs(win_prob - 0.5)
    tiers = {}
    for thresh_label, thresh in [("52%+", 0.02), ("55%+", 0.05), ("58%+", 0.08),
                                  ("60%+", 0.10), ("65%+", 0.15)]:
        mask = conf >= thresh
        n = mask.sum()
        if n > 0:
            tier_acc = ((pred_margin[mask] > 0) == (y_test_margin[mask] > 0)).mean()
            tiers[thresh_label] = {"n": int(n), "accuracy": round(tier_acc * 100, 1)}

    return {
        "ml_accuracy": round(ml_acc * 100, 2),
        "mae": round(mae, 3),
        "brier": round(brier, 4),
        "n_test": len(y_test_margin),
        "tiers": tiers,
    }


def main():
    print("=" * 70)
    print("sp_fip_spread ISOLATION BACKTEST")
    print("22-feature (baseline) vs 25-feature (enhanced)")
    print("=" * 70)

    # Fetch historical data
    print("\nFetching mlb_historical from Supabase...")
    rows = sb_get("mlb_historical",
                   "is_outlier_season=eq.0&actual_home_runs=not.is.null&select=*&order=season.desc&limit=100000")
    if not rows:
        print("ERROR: No historical data returned")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} rows")

    # Ensure numeric types
    for col in ["actual_home_runs", "actual_away_runs", "home_win",
                "home_woba", "away_woba", "home_sp_fip", "away_sp_fip",
                "home_fip", "away_fip", "home_bullpen_era", "away_bullpen_era",
                "park_factor", "temp_f", "wind_mph", "wind_out_flag",
                "home_rest_days", "away_rest_days", "home_travel", "away_travel",
                "season_weight", "home_k9", "away_k9", "home_bb9", "away_bb9",
                "home_sp_ip", "away_sp_ip", "home_def_oaa", "away_def_oaa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Zero out heuristic columns (pure ML comparison)
    df["pred_home_runs"] = 0.0
    df["pred_away_runs"] = 0.0
    df["win_pct_home"] = 0.5
    df["ou_total"] = 0.0
    df["model_ml_home"] = 0

    test_seasons = [2019, 2021, 2022, 2023, 2024, 2025]
    available = sorted(df["season"].dropna().astype(int).unique().tolist())
    print(f"  Available seasons: {available}")

    baseline_results = []
    enhanced_results = []

    for test_season in test_seasons:
        if test_season not in available:
            continue

        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] == test_season].copy()

        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        y_train_margin = (train_df["actual_home_runs"] - train_df["actual_away_runs"]).values
        y_test_margin = (test_df["actual_home_runs"] - test_df["actual_away_runs"]).values
        y_test_win = test_df["home_win"].astype(int).values
        weights = train_df["season_weight"].fillna(1.0).astype(float).values if "season_weight" in train_df.columns else np.ones(len(train_df))

        # ── Baseline (22 features) ──
        X_train_b, _ = build_features_baseline(train_df)
        X_test_b, _ = build_features_baseline(test_df)
        scaler_b = StandardScaler()
        X_train_bs = scaler_b.fit_transform(X_train_b)
        X_test_bs = scaler_b.transform(X_test_b)

        print(f"\n{'='*50}")
        print(f"Season {test_season}: {len(train_df)} train → {len(test_df)} test")

        t0 = time.time()
        b_result = train_and_eval(X_train_bs, X_test_bs, y_train_margin, y_test_margin, y_test_win, weights, "baseline")
        t1 = time.time()
        b_result["season"] = test_season
        baseline_results.append(b_result)
        print(f"  BASELINE (22f): {b_result['ml_accuracy']}% acc | MAE {b_result['mae']} | Brier {b_result['brier']} | {t1-t0:.1f}s")
        for tier, vals in b_result["tiers"].items():
            print(f"    {tier}: {vals['accuracy']}% ({vals['n']} games)")

        # ── Enhanced (25 features) ──
        X_train_e, _ = build_features_enhanced(train_df)
        X_test_e, _ = build_features_enhanced(test_df)
        scaler_e = StandardScaler()
        X_train_es = scaler_e.fit_transform(X_train_e)
        X_test_es = scaler_e.transform(X_test_e)

        t0 = time.time()
        e_result = train_and_eval(X_train_es, X_test_es, y_train_margin, y_test_margin, y_test_win, weights, "enhanced")
        t1 = time.time()
        e_result["season"] = test_season
        enhanced_results.append(e_result)
        print(f"  ENHANCED (25f): {e_result['ml_accuracy']}% acc | MAE {e_result['mae']} | Brier {e_result['brier']} | {t1-t0:.1f}s")
        for tier, vals in e_result["tiers"].items():
            print(f"    {tier}: {vals['accuracy']}% ({vals['n']} games)")

        # Delta
        delta_acc = e_result["ml_accuracy"] - b_result["ml_accuracy"]
        delta_mae = e_result["mae"] - b_result["mae"]
        delta_brier = e_result["brier"] - b_result["brier"]
        print(f"  DELTA: {delta_acc:+.2f}% acc | {delta_mae:+.3f} MAE | {delta_brier:+.4f} Brier")

        # 65%+ tier comparison
        b65 = b_result["tiers"].get("65%+", {})
        e65 = e_result["tiers"].get("65%+", {})
        if b65 and e65:
            print(f"  65%+ TIER: {b65['n']}→{e65['n']} games | {b65['accuracy']}%→{e65['accuracy']}% acc")

    # ── AGGREGATE ──
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    total_b = sum(r["n_test"] for r in baseline_results)
    total_e = sum(r["n_test"] for r in enhanced_results)
    avg_b_acc = np.mean([r["ml_accuracy"] for r in baseline_results])
    avg_e_acc = np.mean([r["ml_accuracy"] for r in enhanced_results])
    avg_b_mae = np.mean([r["mae"] for r in baseline_results])
    avg_e_mae = np.mean([r["mae"] for r in enhanced_results])
    avg_b_brier = np.mean([r["brier"] for r in baseline_results])
    avg_e_brier = np.mean([r["brier"] for r in enhanced_results])

    print(f"  BASELINE (22f): {avg_b_acc:.2f}% avg acc | {avg_b_mae:.3f} MAE | {avg_b_brier:.4f} Brier")
    print(f"  ENHANCED (25f): {avg_e_acc:.2f}% avg acc | {avg_e_mae:.3f} MAE | {avg_e_brier:.4f} Brier")
    print(f"  DELTA:          {avg_e_acc - avg_b_acc:+.2f}% acc | {avg_e_mae - avg_b_mae:+.3f} MAE | {avg_e_brier - avg_b_brier:+.4f} Brier")

    # 65%+ tier aggregate
    b65_total = sum(r["tiers"].get("65%+", {}).get("n", 0) for r in baseline_results)
    e65_total = sum(r["tiers"].get("65%+", {}).get("n", 0) for r in enhanced_results)
    b65_accs = [r["tiers"]["65%+"]["accuracy"] for r in baseline_results if "65%+" in r["tiers"]]
    e65_accs = [r["tiers"]["65%+"]["accuracy"] for r in enhanced_results if "65%+" in r["tiers"]]
    if b65_accs and e65_accs:
        print(f"\n  65%+ TIER AGGREGATE:")
        print(f"    BASELINE: {b65_total} games avg/season, {np.mean(b65_accs):.1f}% avg acc")
        print(f"    ENHANCED: {e65_total} games avg/season, {np.mean(e65_accs):.1f}% avg acc")
        print(f"    DELTA:    {e65_total - b65_total:+d} games | {np.mean(e65_accs) - np.mean(b65_accs):+.1f}% acc")

    print(f"\nNote: platoon_diff=0 and both_lineups_confirmed=0 for all historical rows.")
    print(f"The ONLY active new feature is sp_fip_spread (abs starter FIP gap).")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
