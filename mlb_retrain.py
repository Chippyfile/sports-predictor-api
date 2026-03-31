#!/usr/bin/env python3
"""
mlb_retrain.py — Local MLB model retrain with walk-forward validation
=====================================================================
Replaces Railway's /train/mlb which uses broken isotonic calibration.
Trains on mlb_historical (parquet) + current mlb_predictions.
Architecture: CatBoost solo → margin-based probability (σ=4.0).

Usage:
    python3 mlb_retrain.py                # Train + evaluate
    python3 mlb_retrain.py --upload       # Train + upload to Supabase as 'mlb'
    python3 mlb_retrain.py --refresh      # Re-pull data from Supabase first
"""
import sys, os, time, warnings, pickle, io, base64
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("ERROR: CatBoost required. pip install catboost")
    sys.exit(1)

# ── Season constants (match frontend sharedUtils.js) ──
SEASON_CONSTANTS = {
    2015: {"lg_rpg": 4.25}, 2016: {"lg_rpg": 4.48}, 2017: {"lg_rpg": 4.65},
    2018: {"lg_rpg": 4.45}, 2019: {"lg_rpg": 4.83}, 2021: {"lg_rpg": 4.53},
    2022: {"lg_rpg": 4.28}, 2023: {"lg_rpg": 4.62}, 2024: {"lg_rpg": 4.38},
    2025: {"lg_rpg": 4.30}, 2026: {"lg_rpg": 4.30},
}
DEFAULT_LG_RPG = 4.30

# ── Feature builder (mirrors sports/mlb.py mlb_build_features exactly) ──
FEATURE_COLS = [
    "woba_diff", "fip_diff", "has_sp_fip", "bullpen_era_diff", "k_bb_diff",
    "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff",
    "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold",
    "rest_diff", "travel_diff", "lg_rpg",
    "fip_x_bullpen", "woba_x_park", "wind_x_fip",
    "run_diff_pred", "has_heuristic",
    "platoon_diff", "sp_fip_spread", "both_lineups_confirmed",
    "market_spread", "market_total", "spread_vs_market", "has_market",
    # ── Advanced features (v7) ──
    "pyth_residual_diff", "babip_luck_diff", "scoring_entropy_diff",
    "first_inn_rate_diff", "clutch_divergence_diff", "opp_adj_form_diff",
    "ump_run_env", "series_game_num",
    "scoring_entropy_combined", "first_inn_rate_combined",
    "sp_relative_fip_diff", "temp_x_park",
]

def build_features(df):
    """Build 41 features from raw data. Mirrors sports/mlb.py exactly."""
    df = df.copy()

    # Raw inputs with defaults
    raw_defaults = {
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
    for col, default in raw_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # SP FIP fallback
    if "home_sp_fip_known" in df.columns:
        home_sp_known = pd.to_numeric(df["home_sp_fip_known"], errors="coerce").fillna(0).astype(bool)
        away_sp_known = pd.to_numeric(df["away_sp_fip_known"], errors="coerce").fillna(0).astype(bool)
        df["home_starter_fip"] = np.where(home_sp_known, df["home_sp_fip"], df["home_fip"])
        df["away_starter_fip"] = np.where(away_sp_known, df["away_sp_fip"], df["away_fip"])
        df["has_sp_fip"] = (home_sp_known & away_sp_known).astype(int)
    else:
        df["home_starter_fip"] = df["home_sp_fip"].where(df["home_sp_fip"] != 4.25, df["home_fip"])
        df["away_starter_fip"] = df["away_sp_fip"].where(df["away_sp_fip"] != 4.25, df["away_fip"])
        df["has_sp_fip"] = ((df["home_sp_fip"] != 4.25) & (df["away_sp_fip"] != 4.25)).astype(int)

    # Derived features
    df["woba_diff"] = df["home_woba"] - df["away_woba"]
    df["fip_diff"] = df["home_starter_fip"] - df["away_starter_fip"]
    df["bullpen_era_diff"] = df["home_bullpen_era"] - df["away_bullpen_era"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["travel_diff"] = df["home_travel"] - df["away_travel"]
    df["is_warm"] = (df["temp_f"] > 75).astype(int)
    df["is_cold"] = (df["temp_f"] < 45).astype(int)
    df["wind_out"] = df["wind_out_flag"].astype(int)

    # K-BB
    df["k_bb_home"] = df["home_k9"] - df["home_bb9"]
    df["k_bb_away"] = df["away_k9"] - df["away_bb9"]
    df["k_bb_diff"] = df["k_bb_home"] - df["k_bb_away"]

    # SP innings & bullpen exposure
    df["sp_ip_diff"] = df["home_sp_ip"] - df["away_sp_ip"]
    df["home_bp_exposure"] = np.maximum(0, 9.0 - df["home_sp_ip"]) * (df["home_bullpen_era"] / 4.10)
    df["away_bp_exposure"] = np.maximum(0, 9.0 - df["away_sp_ip"]) * (df["away_bullpen_era"] / 4.10)
    df["bp_exposure_diff"] = df["home_bp_exposure"] - df["away_bp_exposure"]
    df["def_oaa_diff"] = df["home_def_oaa"] - df["away_def_oaa"]

    # Enhancements
    df["platoon_diff"] = df["home_platoon_delta"] - df["away_platoon_delta"]
    df["sp_fip_spread"] = (df["home_starter_fip"] - df["away_starter_fip"]).abs()
    df["both_lineups_confirmed"] = (
        (df["home_lineup_confirmed"] == 1) & (df["away_lineup_confirmed"] == 1)
    ).astype(int)

    # Interactions
    df["fip_x_bullpen"] = df["fip_diff"] * df["bullpen_era_diff"]
    df["woba_x_park"] = df["woba_diff"] * df["park_factor"]
    df["wind_x_fip"] = df["wind_out"].astype(float) * df["fip_diff"]

    # League RPG
    if "season" in df.columns:
        df["lg_rpg"] = df["season"].map(lambda s: SEASON_CONSTANTS.get(int(s), {}).get("lg_rpg", DEFAULT_LG_RPG) if pd.notna(s) else DEFAULT_LG_RPG)
    else:
        df["lg_rpg"] = DEFAULT_LG_RPG

    # Heuristic signal
    if "pred_home_runs" in df.columns and "pred_away_runs" in df.columns:
        ph = pd.to_numeric(df["pred_home_runs"], errors="coerce").fillna(0)
        pa = pd.to_numeric(df["pred_away_runs"], errors="coerce").fillna(0)
        df["run_diff_pred"] = ph - pa
        df["has_heuristic"] = ((ph > 0) | (pa > 0)).astype(int)
    else:
        df["run_diff_pred"] = 0.0
        df["has_heuristic"] = 0

    # Market features
    df["market_spread"] = pd.to_numeric(
        df["market_spread_home"] if "market_spread_home" in df.columns else 0,
        errors="coerce"
    ).fillna(0)
    df["market_total"] = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else 0,
        errors="coerce"
    ).fillna(0)
    df["has_market"] = ((df["market_spread"] != 0) | (df["market_total"] != 0)).astype(int)
    df["spread_vs_market"] = df["run_diff_pred"] - df["market_spread"]

    # ── Advanced features (v7) — pre-computed in parquet, pass through ──
    advanced_cols = {
        "pyth_residual_diff": 0.0,
        "babip_luck_diff": 0.0,
        "scoring_entropy_diff": 0.0,
        "first_inn_rate_diff": 0.0,
        "clutch_divergence_diff": 0.0,
        "opp_adj_form_diff": 0.0,
        "ump_run_env": 8.5,       # league average total runs
        "series_game_num": 1.0,
        "scoring_entropy_combined": 5.0,  # ~average combined entropy
        "first_inn_rate_combined": 0.8,   # ~average combined first-inning rate
    }
    for col, default in advanced_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # sp_relative_fip: starter quality relative to own team
    if "sp_relative_fip_diff" in df.columns:
        df["sp_relative_fip_diff"] = pd.to_numeric(df["sp_relative_fip_diff"], errors="coerce").fillna(0)
    else:
        df["sp_relative_fip_diff"] = (
            (df.get("home_sp_fip", pd.Series(4.25)) - df.get("home_fip", pd.Series(4.25))) -
            (df.get("away_sp_fip", pd.Series(4.25)) - df.get("away_fip", pd.Series(4.25)))
        ).fillna(0)

    # temp × park interaction
    if "temp_x_park" in df.columns:
        df["temp_x_park"] = pd.to_numeric(df["temp_x_park"], errors="coerce").fillna(0)
    else:
        df["temp_x_park"] = ((df["temp_f"] - 70) / 30.0) * df["park_factor"]

    return df[FEATURE_COLS].fillna(0)


def load_data(refresh=False):
    """Load training data from local parquet or Supabase."""
    parquet_path = "mlb_training_data.parquet"

    if refresh or not os.path.exists(parquet_path):
        print("Pulling from Supabase...")
        import requests
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        all_rows = []
        offset = 0
        while True:
            r = requests.get(
                f"{url}/rest/v1/mlb_historical?select=*&order=game_date.asc&limit=1000&offset={offset}",
                headers=headers
            )
            rows = r.json()
            if not isinstance(rows, list) or not rows:
                break
            all_rows.extend(rows)
            offset += len(rows)
        df = pd.DataFrame(all_rows)
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved {len(df)} historical games")
    else:
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} games from {parquet_path}")

    # Filter: need actual scores
    df["actual_home_runs"] = pd.to_numeric(df["actual_home_runs"], errors="coerce")
    df["actual_away_runs"] = pd.to_numeric(df["actual_away_runs"], errors="coerce")
    df = df.dropna(subset=["actual_home_runs", "actual_away_runs"])

    # Target
    df["target_margin"] = df["actual_home_runs"] - df["actual_away_runs"]

    # Season
    df["season"] = pd.to_datetime(df["game_date"]).dt.year

    # Season weight
    if "season_weight" not in df.columns or df["season_weight"].isna().all():
        current = df["season"].max()
        df["season_weight"] = df["season"].map(lambda s: max(0.5, 1.0 - (current - s) * 0.1))

    print(f"Seasons: {sorted(df.season.unique())}")
    print(f"Games with scores: {len(df)}")

    return df


def main():
    upload = "--upload" in sys.argv
    refresh = "--refresh" in sys.argv

    print("=" * 70)
    print("  MLB LOCAL RETRAIN — CatBoost + Walk-Forward Validation")
    print("  No isotonic calibrator (margin-based σ=4.0 probability)")
    print("=" * 70)

    # ── Load data ──
    df = load_data(refresh=refresh)
    y = df["target_margin"].values
    weights = df["season_weight"].values if "season_weight" in df.columns else None

    # ── Build features ──
    X_df = build_features(df)
    feature_names = list(X_df.columns)
    X = X_df.values
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Target mean: {y.mean():.3f}, std: {y.std():.3f}")

    # Feature coverage audit
    for f in feature_names:
        nz = (X_df[f] != 0).sum()
        if nz < len(df) * 0.1:
            print(f"  ⚠️ {f}: only {nz}/{len(df)} non-zero ({100*nz/len(df):.1f}%)")

    # ── Walk-forward validation ──
    print(f"\n  Walk-forward validation (20-fold, time-ordered)...")
    n_folds = 20
    fold_size = len(X) // (n_folds + 3)
    min_train = fold_size * 3
    all_preds = np.full(len(X), np.nan)
    t0 = time.time()

    for fold in range(n_folds):
        te_s = min_train + fold * fold_size
        te_e = min(te_s + fold_size, len(X))
        if te_s >= len(X):
            break
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:te_s])
        Xte = sc.transform(X[te_s:te_e])
        wt = weights[:te_s] if weights is not None else None
        m = CatBoostRegressor(
            depth=4, iterations=50, learning_rate=0.06,
            subsample=0.8, min_data_in_leaf=20,
            random_seed=42, verbose=0,
        )
        m.fit(Xtr, y[:te_s], sample_weight=wt)
        all_preds[te_s:te_e] = m.predict(Xte)
        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{n_folds} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Complete in {elapsed:.0f}s")

    # ── Evaluate ──
    valid = ~np.isnan(all_preds)
    pv, tv = all_preds[valid], y[valid]
    mae = np.mean(np.abs(pv - tv))
    bias = float(np.mean(tv - pv))
    win_correct = ((pv > 0) == (tv > 0)).sum()
    win_total = (tv != 0).sum()
    win_acc = win_correct / win_total * 100

    print(f"\n  Walk-forward MAE: {mae:.3f}")
    print(f"  Win prediction accuracy: {win_acc:.1f}%")
    print(f"  Bias: {bias:+.3f}")

    # Market RL accuracy
    mkt_col = X_df["market_spread"].values
    has_mkt = mkt_col[valid] != 0
    if has_mkt.sum() > 100:
        model_side = pv[has_mkt] + mkt_col[valid][has_mkt]
        actual_side = tv[has_mkt] + mkt_col[valid][has_mkt]
        rl_correct = ((model_side > 0) == (actual_side > 0)).sum()
        not_push = actual_side != 0
        rl_acc = rl_correct / not_push.sum() * 100 if not_push.sum() > 0 else 0
        print(f"  Run Line accuracy (with market): {rl_acc:.1f}% on {not_push.sum()} games")

    # O/U accuracy (predict total from margin? no — separate model needed)

    # ── Train production model ──
    print(f"\n  Training production model on {len(X)} games...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    prod = CatBoostRegressor(
        depth=4, iterations=50, learning_rate=0.06,
        subsample=0.8, min_data_in_leaf=20,
        random_seed=42, verbose=0,
    )
    prod.fit(X_s, y, sample_weight=weights)
    in_mae = np.mean(np.abs(prod.predict(X_s) - y))
    print(f"  In-sample MAE: {in_mae:.3f}")

    # ── SHAP explainer (CatBoost built-in) ──
    # CatBoost has native SHAP support — much better than the old LinearExplainer
    print("  Building SHAP explainer...")
    try:
        import shap
        explainer = shap.TreeExplainer(prod)
        print("  ✅ TreeExplainer built")
    except Exception as e:
        explainer = None
        print(f"  ⚠️ SHAP explainer failed: {e} (will use feature importance fallback)")

    # ── Save bundle ──
    bundle = {
        "reg": prod,
        "scaler": scaler,
        "feature_cols": feature_names,
        "explainer": explainer,
        "bias_correction": bias,
        "cv_mae": round(mae, 4),
        "n_train": len(X),
        "n_historical": len(df),
        "n_current": 0,
        "model_type": "CatBoost_v7_41feat",
        "trained_at": pd.Timestamp.now().isoformat(),
        "mae_cv": round(mae, 4),
        # No clf or isotonic — probability computed from margin at serve time
        "clf": None,
        "isotonic": None,
    }

    local_path = "mlb_model_local.pkl"
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
            "size_bytes": len(buf.getvalue()),
            "trained_at": bundle["trained_at"],
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

        # Reload on Railway
        try:
            r = requests.post("https://sports-predictor-api-production.up.railway.app/debug/reload-model/mlb", timeout=30)
            print(f"  Railway reload: {r.json()}")
        except Exception as e:
            print(f"  Railway reload failed: {e}")

    print(f"\n{'='*70}")
    print(f"  MLB RETRAIN COMPLETE")
    print(f"  Features: {len(feature_names)}")
    print(f"  Walk-forward MAE: {mae:.3f}")
    print(f"  Win accuracy: {win_acc:.1f}%")
    print(f"  Bias: {bias:+.3f}")
    print(f"  No isotonic calibrator — margin-based probability (σ=4.0)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
