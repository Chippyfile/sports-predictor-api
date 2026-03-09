#!/usr/bin/env python3
"""
mega_sweep.py — Multi-Sport ATS Accuracy Improvement Sweep
═══════════════════════════════════════════════════════════
Tests 5 dimensions across MLB, NBA, and NCAA:
  1. TARGET VARIABLE:  actual_margin vs residual (actual_margin - market_spread)
  2. FEATURE SETS:     baseline vs +enhanced (Elo residual, rolling windows)
  3. BASE LEARNERS:    XGB, CAT, RF, LightGBM, GBM — all combos
  4. ESTIMATOR COUNT:  50, 100, 150, 200
  5. EVALUATION:       MAE, ATS accuracy, Brier score, calibration error

Run from sports-predictor-api root:
    python3 mega_sweep.py --sport nba
    python3 mega_sweep.py --sport ncaa
    python3 mega_sweep.py --sport mlb
    python3 mega_sweep.py --sport all
"""
import argparse, time, sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.special import expit  # sigmoid
from db import sb_get
from ml_utils import HAS_XGB, _time_series_oof

warnings.filterwarnings("ignore")

if HAS_XGB:
    from xgboost import XGBRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ═══════════════════════════════════════════════════════════════
# ENHANCED FEATURE BUILDERS (wrap existing ones + add new features)
# ═══════════════════════════════════════════════════════════════

def _add_enhanced_features_nba(df, feature_df):
    """
    Add enhanced features on top of the baseline NBA feature set.
    - elo_residual: model spread minus market spread (KenPom vs Vegas disagreement)
    - line_movement: closing - opening spread (sharp money signal)
    - games_last_7/14: cumulative fatigue proxy
    - win_streak: momentum signal
    - market_spread_abs: magnitude of spread (blowout vs tossup)
    """
    enhanced = feature_df.copy()

    # ── Elo/KenPom residual (disagreement between model and market) ──
    score_diff = pd.to_numeric(df.get("pred_home_score", 0), errors="coerce").fillna(0) - \
                 pd.to_numeric(df.get("pred_away_score", 0), errors="coerce").fillna(0)
    mkt_spread = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    enhanced["elo_residual"] = score_diff - mkt_spread

    # ── Market spread magnitude (tossup vs blowout regime) ──
    enhanced["market_spread_abs"] = mkt_spread.abs()

    # ── Win percentage differential (form signal) ──
    h_wins = pd.to_numeric(df.get("home_wins", 20), errors="coerce").fillna(20)
    h_losses = pd.to_numeric(df.get("home_losses", 20), errors="coerce").fillna(20)
    a_wins = pd.to_numeric(df.get("away_wins", 20), errors="coerce").fillna(20)
    a_losses = pd.to_numeric(df.get("away_losses", 20), errors="coerce").fillna(20)
    h_wpct = h_wins / (h_wins + h_losses).clip(lower=1)
    a_wpct = a_wins / (a_wins + a_losses).clip(lower=1)
    enhanced["wpct_diff"] = h_wpct - a_wpct

    # ── Net rating squared interaction (non-linear quality gap) ──
    net_diff = pd.to_numeric(enhanced.get("net_rtg_diff", 0), errors="coerce").fillna(0)
    enhanced["net_rtg_diff_sq"] = net_diff ** 2 * np.sign(net_diff)

    # ── Tempo mismatch (pace differential) ──
    h_tempo = pd.to_numeric(df.get("home_tempo", 100), errors="coerce").fillna(100)
    a_tempo = pd.to_numeric(df.get("away_tempo", 100), errors="coerce").fillna(100)
    enhanced["tempo_mismatch"] = (h_tempo - a_tempo).abs()

    return enhanced


def _add_enhanced_features_ncaa(df, feature_df):
    """NCAA-specific enhanced features."""
    enhanced = feature_df.copy()

    # ── EM residual (KenPom spread vs market) ──
    h_em = pd.to_numeric(df.get("home_adj_em", 0), errors="coerce").fillna(0)
    a_em = pd.to_numeric(df.get("away_adj_em", 0), errors="coerce").fillna(0)
    em_diff = h_em - a_em
    mkt_spread = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    enhanced["em_residual"] = em_diff - mkt_spread

    # ── Market spread magnitude ──
    enhanced["market_spread_abs"] = mkt_spread.abs()

    # ── Rank interaction (both ranked = tighter game) ──
    h_rank = pd.to_numeric(df.get("home_rank", 200), errors="coerce").fillna(200)
    a_rank = pd.to_numeric(df.get("away_rank", 200), errors="coerce").fillna(200)
    enhanced["both_ranked"] = ((h_rank <= 50) & (a_rank <= 50)).astype(int)
    enhanced["rank_diff"] = a_rank - h_rank  # positive = home ranked higher

    # ── SOS differential ──
    h_sos = pd.to_numeric(df.get("home_sos", 0.5), errors="coerce").fillna(0.5)
    a_sos = pd.to_numeric(df.get("away_sos", 0.5), errors="coerce").fillna(0.5)
    enhanced["sos_diff"] = h_sos - a_sos

    # ── EM squared interaction ──
    enhanced["em_diff_sq"] = em_diff ** 2 * np.sign(em_diff)

    # ── Tempo mismatch ──
    h_tempo = pd.to_numeric(df.get("home_tempo", 68), errors="coerce").fillna(68)
    a_tempo = pd.to_numeric(df.get("away_tempo", 68), errors="coerce").fillna(68)
    enhanced["tempo_mismatch"] = (h_tempo - a_tempo).abs()

    return enhanced


def _add_enhanced_features_mlb(df, feature_df):
    """MLB-specific enhanced features."""
    enhanced = feature_df.copy()

    # ── Model vs market residual ──
    run_diff = pd.to_numeric(df.get("pred_home_runs", 0), errors="coerce").fillna(0) - \
               pd.to_numeric(df.get("pred_away_runs", 0), errors="coerce").fillna(0)
    mkt_spread = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    enhanced["model_residual"] = run_diff - mkt_spread

    # ── SP FIP squared (non-linear starter quality) ──
    h_fip = pd.to_numeric(df.get("home_sp_fip", df.get("home_fip", 4.25)),
                          errors="coerce").fillna(4.25)
    a_fip = pd.to_numeric(df.get("away_sp_fip", df.get("away_fip", 4.25)),
                          errors="coerce").fillna(4.25)
    fip_diff = a_fip - h_fip  # positive = home has better SP
    enhanced["fip_diff_sq"] = fip_diff ** 2 * np.sign(fip_diff)

    # ── wOBA × bullpen interaction ──
    h_woba = pd.to_numeric(df.get("home_woba", 0.314), errors="coerce").fillna(0.314)
    a_woba = pd.to_numeric(df.get("away_woba", 0.314), errors="coerce").fillna(0.314)
    h_bp = pd.to_numeric(df.get("home_bullpen_era", 4.1), errors="coerce").fillna(4.1)
    a_bp = pd.to_numeric(df.get("away_bullpen_era", 4.1), errors="coerce").fillna(4.1)
    enhanced["woba_x_opp_bp"] = (h_woba - a_woba) * (a_bp - h_bp)

    # ── Market spread magnitude ──
    enhanced["market_spread_abs"] = mkt_spread.abs()

    return enhanced


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def load_nba_data():
    """Load NBA data and return (df, X_baseline, y_margin, market_spread, weights)."""
    from sports.nba import nba_build_features, _nba_merge_historical

    rows = sb_get("nba_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df, weights = _nba_merge_historical(current_df)

    X_base = nba_build_features(df)
    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)

    mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    has_mkt = (mkt != 0)

    w = weights if weights is not None else np.ones(len(df))
    return df, X_base, y_margin, mkt, has_mkt, w


def load_ncaa_data():
    """Load NCAA data and return (df, X_baseline, y_margin, market_spread, weights)."""
    from sports.ncaa import ncaa_build_features, _ncaa_merge_historical

    rows = sb_get("ncaa_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df, weights, n_hist = _ncaa_merge_historical(current_df)

    X_base = ncaa_build_features(df)
    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)

    mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    has_mkt = (mkt != 0)

    w = weights if weights is not None else np.ones(len(df))
    return df, X_base, y_margin, mkt, has_mkt, w


def load_mlb_data():
    """Load MLB data and return (df, X_baseline, y_margin, market_spread, weights)."""
    from sports.mlb import mlb_build_features, _mlb_merge_historical

    rows = sb_get("mlb_predictions",
                  "result_entered=eq.true&actual_home_runs=not.is.null&game_type=eq.R&select=*")
    current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df, weights = _mlb_merge_historical(current_df)

    X_base = mlb_build_features(df)
    y_margin = df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)

    mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    has_mkt = (mkt != 0)

    w = weights if weights is not None else np.ones(len(df))
    return df, X_base, y_margin, mkt, has_mkt, w


# ═══════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════

def compute_ats_accuracy(pred_margin, actual_margin, market_spread):
    """
    ATS accuracy: does the model correctly predict whether the actual margin
    beats the spread? Only evaluated on games WITH market data.
    """
    mask = (market_spread != 0)
    if mask.sum() < 10:
        return np.nan

    model_pick = pred_margin - market_spread  # positive = model says home covers
    actual_cover = actual_margin - market_spread  # positive = home actually covered

    # Exclude pushes
    non_push = (actual_cover != 0)
    combined = mask & non_push

    if combined.sum() < 10:
        return np.nan

    correct = ((model_pick[combined] > 0) == (actual_cover[combined] > 0))
    return float(correct.mean())


def compute_calibration_error(pred_probs, actual_wins, n_bins=10):
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(pred_probs)
    for i in range(n_bins):
        mask = (pred_probs >= bins[i]) & (pred_probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = pred_probs[mask].mean()
        avg_actual = actual_wins[mask].mean()
        ece += (mask.sum() / total) * abs(avg_pred - avg_actual)
    return ece


def compute_clv(pred_margin, market_spread, actual_margin):
    """
    Closing Line Value: when model disagrees with market,
    does the actual result move in the model's direction?
    """
    mask = (market_spread != 0)
    if mask.sum() < 10:
        return np.nan

    disagree = (pred_margin - market_spread)  # model's "edge"
    reality = (actual_margin - market_spread)  # what actually happened

    # CLV = correlation between model's edge and actual residual
    valid = mask & np.isfinite(disagree) & np.isfinite(reality)
    if valid.sum() < 10:
        return np.nan

    return float(np.corrcoef(disagree[valid], reality[valid])[0, 1])


# ═══════════════════════════════════════════════════════════════
# CORE SWEEP ENGINE
# ═══════════════════════════════════════════════════════════════

def build_models(combo, n_est):
    """Create model dict from combo list."""
    models = {}
    if "RF" in combo:
        models["rf"] = RandomForestRegressor(
            n_estimators=n_est, max_depth=6, min_samples_leaf=15,
            max_features=0.7, random_state=42, n_jobs=-1)
    if "XGB" in combo and HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=42, tree_method="hist", verbosity=0)
    if "CAT" in combo and HAS_CAT:
        models["cat"] = CatBoostRegressor(
            iterations=n_est, depth=4, learning_rate=0.06,
            subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0)
    if "GBM" in combo:
        models["gbm"] = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=20, random_state=42)
    if "LGBM" in combo and HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, verbose=-1)
    return models


def run_single_config(X, y_target, y_actual_margin, market_spread, df, weights,
                      combo, n_est, cv_folds=10, sigma=None):
    """
    Run one sweep configuration. Returns dict of all metrics.

    y_target: what the model trains on (actual_margin or residual)
    y_actual_margin: always the true margin (for ATS evaluation)
    market_spread: Vegas line (for ATS + CLV)
    sigma: sport-specific sigma for win prob (NBA~11, NCAA~11, MLB~4)
    """
    t0 = time.time()
    models = build_models(combo, n_est)
    if not models:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # OOF predictions
    oof = _time_series_oof(models, X_scaled, y_target, df,
                           n_splits=cv_folds, weights=weights)

    # Stack if multiple models
    if len(models) == 1:
        name = list(models.keys())[0]
        oof_pred = oof[name]
    else:
        # Full fits for stacking
        for name, model in models.items():
            model.fit(X_scaled, y_target, sample_weight=weights)
        meta_X = np.column_stack([oof[k] for k in models.keys()])
        meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        meta_reg.fit(meta_X, y_target)
        oof_pred = meta_reg.predict(meta_X)

    # ── Convert OOF predictions back to actual margin space ──
    # If we trained on residuals, add market_spread back
    is_residual = not np.allclose(y_target.values, y_actual_margin.values, atol=0.01)
    if is_residual:
        pred_margin = oof_pred + market_spread.values
    else:
        pred_margin = oof_pred

    # ── Metrics ──
    # 1. MAE on actual margin
    mae = float(mean_absolute_error(y_actual_margin.values, pred_margin))

    # 2. MAE on target (what model actually optimized)
    mae_target = float(mean_absolute_error(y_target.values, oof_pred))

    # 3. ATS accuracy
    ats = compute_ats_accuracy(pred_margin, y_actual_margin.values, market_spread.values)

    # 4. Win probability + Brier score
    if sigma is None:
        sigma = 11.0  # default for basketball
    pred_wp = expit(pred_margin / sigma)
    y_win = (y_actual_margin > 0).astype(int).values
    brier = float(brier_score_loss(y_win, pred_wp))

    # 5. Calibration error
    ece = compute_calibration_error(pred_wp, y_win)

    # 6. CLV
    clv = compute_clv(pred_margin, market_spread.values, y_actual_margin.values)

    # 7. SU accuracy (straight up — does model pick the winner?)
    su_correct = ((pred_margin > 0) == (y_actual_margin.values > 0))
    # exclude exact ties
    non_tie = (y_actual_margin.values != 0)
    su_acc = float(su_correct[non_tie].mean()) if non_tie.sum() > 10 else np.nan

    # 8. High-confidence tier ATS (model disagrees with market by > 2pts)
    disagree = np.abs(pred_margin - market_spread.values)
    has_mkt = (market_spread.values != 0)
    high_conf = has_mkt & (disagree > 2.0)
    if high_conf.sum() >= 10:
        hc_cover = y_actual_margin.values[high_conf] - market_spread.values[high_conf]
        hc_pick = pred_margin[high_conf] - market_spread.values[high_conf]
        non_push = (hc_cover != 0)
        if non_push.sum() >= 10:
            hc_ats = float(((hc_pick[non_push] > 0) == (hc_cover[non_push] > 0)).mean())
        else:
            hc_ats = np.nan
        hc_n = int(high_conf.sum())
    else:
        hc_ats = np.nan
        hc_n = 0

    elapsed = time.time() - t0

    return {
        "mae": mae,
        "mae_target": mae_target,
        "ats_pct": ats,
        "su_acc": su_acc,
        "brier": brier,
        "ece": ece,
        "clv": clv,
        "hc_ats": hc_ats,
        "hc_n": hc_n,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# SPORT-SPECIFIC SWEEP RUNNERS
# ═══════════════════════════════════════════════════════════════

def sweep_sport(sport):
    """Run the full multi-dimensional sweep for one sport."""
    print(f"\n{'='*90}")
    print(f"  MEGA SWEEP: {sport.upper()}")
    print(f"{'='*90}")

    # ── Load data ──
    print(f"\n  Loading {sport.upper()} data...")
    if sport == "nba":
        df, X_base, y_margin, mkt, has_mkt, weights = load_nba_data()
        enhance_fn = _add_enhanced_features_nba
        sigma = 11.0
    elif sport == "ncaa":
        df, X_base, y_margin, mkt, has_mkt, weights = load_ncaa_data()
        enhance_fn = _add_enhanced_features_ncaa
        sigma = 11.0
    elif sport == "mlb":
        df, X_base, y_margin, mkt, has_mkt, weights = load_mlb_data()
        enhance_fn = _add_enhanced_features_mlb
        sigma = 4.0
    else:
        print(f"  Unknown sport: {sport}")
        return

    n = len(df)
    n_mkt = int(has_mkt.sum())
    print(f"  Dataset: {n} rows, {len(X_base.columns)} base features, "
          f"{n_mkt} with market data ({n_mkt/n*100:.0f}%)")

    # ── Build feature sets ──
    X_enhanced = enhance_fn(df, X_base)
    print(f"  Enhanced features: {len(X_enhanced.columns)} "
          f"(+{len(X_enhanced.columns) - len(X_base.columns)} new)")

    # ── Define sweep dimensions ──
    target_configs = [
        ("margin", y_margin),   # predict actual margin directly
    ]
    # Only add residual target if we have enough market data
    if n_mkt > n * 0.5:
        y_residual = y_margin - mkt
        target_configs.append(("residual", y_residual))
        print(f"  Residual target enabled ({n_mkt}/{n} games have market data)")
    else:
        print(f"  Residual target SKIPPED (only {n_mkt}/{n} games have market data)")

    feature_configs = [
        ("baseline", X_base),
        ("enhanced", X_enhanced),
    ]

    # Learner combos — ordered by expected quality
    learner_combos = [
        ["XGB", "CAT", "RF"],          # current best for NCAA
        ["XGB", "CAT", "LGBM"],        # swap RF for LightGBM
        ["XGB", "CAT", "RF", "LGBM"],  # add LightGBM to existing stack
        ["XGB", "CAT", "GBM", "RF"],   # current full stack
        ["LGBM"],                       # LightGBM solo
        ["CAT"],                        # CatBoost solo (current MLB best)
        ["XGB", "LGBM"],               # gradient boosters only
    ]

    estimator_counts = [50, 100, 150, 200]

    # ── Run sweep ──
    results = []
    total_configs = (len(target_configs) * len(feature_configs) *
                     len(learner_combos) * len(estimator_counts))
    print(f"\n  Total configurations: {total_configs}")
    print(f"  (may take 10-30 min depending on dataset size)\n")

    # Header
    print(f"  {'#':>3} | {'Target':>8} | {'Features':>9} | {'Combo':<20} | {'Est':>4} | "
          f"{'MAE':>7} | {'ATS%':>6} | {'SU%':>6} | {'Brier':>7} | {'ECE':>6} | "
          f"{'CLV':>6} | {'HC-ATS':>7} | {'HC-N':>5} | {'Time':>5}")
    print(f"  {'-'*130}")

    idx = 0
    for target_name, y_target in target_configs:
        for feat_name, X_feat in feature_configs:
            for combo in learner_combos:
                # Check if required libraries are available
                if "XGB" in combo and not HAS_XGB:
                    continue
                if "CAT" in combo and not HAS_CAT:
                    continue
                if "LGBM" in combo and not HAS_LGBM:
                    continue

                for n_est in estimator_counts:
                    idx += 1
                    label = "+".join(combo)

                    try:
                        r = run_single_config(
                            X_feat, y_target, y_margin, mkt, df, weights,
                            combo, n_est, cv_folds=10, sigma=sigma
                        )
                        if r is None:
                            continue

                        r["target"] = target_name
                        r["features"] = feat_name
                        r["combo"] = label
                        r["n_est"] = n_est
                        results.append(r)

                        # Print row
                        ats_str = f"{r['ats_pct']*100:.1f}" if not np.isnan(r['ats_pct']) else "  N/A"
                        su_str = f"{r['su_acc']*100:.1f}" if not np.isnan(r['su_acc']) else "  N/A"
                        clv_str = f"{r['clv']:.3f}" if not np.isnan(r['clv']) else "  N/A"
                        hc_str = f"{r['hc_ats']*100:.1f}" if not np.isnan(r['hc_ats']) else "  N/A"

                        # Mark if this is current best MAE
                        best_mae = min(rr["mae"] for rr in results)
                        marker = " ★" if r["mae"] <= best_mae else ""

                        print(f"  {idx:>3} | {target_name:>8} | {feat_name:>9} | {label:<20} | "
                              f"{n_est:>4} | {r['mae']:>7.3f} | {ats_str:>6} | {su_str:>6} | "
                              f"{r['brier']:>7.4f} | {r['ece']:>.4f} | {clv_str:>6} | "
                              f"{hc_str:>7} | {r['hc_n']:>5} | {r['elapsed']:>4.0f}s{marker}")

                    except Exception as e:
                        print(f"  {idx:>3} | {target_name:>8} | {feat_name:>9} | {label:<20} | "
                              f"{n_est:>4} | ERROR: {str(e)[:50]}")

    # ═══════════════════════════════════════════════════════════
    # RESULTS ANALYSIS
    # ═══════════════════════════════════════════════════════════
    if not results:
        print("\n  No results to analyze!")
        return

    rdf = pd.DataFrame(results)

    print(f"\n{'='*90}")
    print(f"  RESULTS ANALYSIS: {sport.upper()}")
    print(f"{'='*90}")

    # ── 1. Best by MAE ──
    print(f"\n  ── TOP 5 BY MAE (margin prediction accuracy) ──")
    top_mae = rdf.nsmallest(5, "mae")
    for _, r in top_mae.iterrows():
        ats_str = f"{r['ats_pct']*100:.1f}%" if not np.isnan(r['ats_pct']) else "N/A"
        print(f"    MAE={r['mae']:.3f}  ATS={ats_str:>6}  "
              f"{r['target']:>8}/{r['features']:>9}  {r['combo']:<20}  est={r['n_est']}")

    # ── 2. Best by ATS ──
    ats_valid = rdf[rdf["ats_pct"].notna()]
    if len(ats_valid) > 0:
        print(f"\n  ── TOP 5 BY ATS ACCURACY (against the spread) ──")
        top_ats = ats_valid.nlargest(5, "ats_pct")
        for _, r in top_ats.iterrows():
            print(f"    ATS={r['ats_pct']*100:.1f}%  MAE={r['mae']:.3f}  "
                  f"{r['target']:>8}/{r['features']:>9}  {r['combo']:<20}  est={r['n_est']}")

    # ── 3. Best by Brier (calibration) ──
    print(f"\n  ── TOP 5 BY BRIER SCORE (lower = better calibrated) ──")
    top_brier = rdf.nsmallest(5, "brier")
    for _, r in top_brier.iterrows():
        ats_str = f"{r['ats_pct']*100:.1f}%" if not np.isnan(r['ats_pct']) else "N/A"
        print(f"    Brier={r['brier']:.4f}  ATS={ats_str:>6}  "
              f"{r['target']:>8}/{r['features']:>9}  {r['combo']:<20}  est={r['n_est']}")

    # ── 4. Best by CLV ──
    clv_valid = rdf[rdf["clv"].notna()]
    if len(clv_valid) > 0:
        print(f"\n  ── TOP 5 BY CLV (closing line value correlation) ──")
        top_clv = clv_valid.nlargest(5, "clv")
        for _, r in top_clv.iterrows():
            ats_str = f"{r['ats_pct']*100:.1f}%" if not np.isnan(r['ats_pct']) else "N/A"
            print(f"    CLV={r['clv']:.3f}  ATS={ats_str:>6}  MAE={r['mae']:.3f}  "
                  f"{r['target']:>8}/{r['features']:>9}  {r['combo']:<20}")

    # ── 5. Best high-confidence ATS ──
    hc_valid = rdf[(rdf["hc_ats"].notna()) & (rdf["hc_n"] >= 20)]
    if len(hc_valid) > 0:
        print(f"\n  ── TOP 5 HIGH-CONFIDENCE ATS (model disagrees w/ market by >2pts) ──")
        top_hc = hc_valid.nlargest(5, "hc_ats")
        for _, r in top_hc.iterrows():
            print(f"    HC-ATS={r['hc_ats']*100:.1f}% ({r['hc_n']:.0f} games)  "
                  f"MAE={r['mae']:.3f}  {r['target']:>8}/{r['features']:>9}  {r['combo']:<20}")

    # ── 6. Dimension-level analysis ──
    print(f"\n  ── DIMENSION ANALYSIS (average across all configs) ──")

    # Target variable impact
    print(f"\n    Target Variable:")
    for t in rdf["target"].unique():
        sub = rdf[rdf["target"] == t]
        ats_mean = sub["ats_pct"].dropna().mean()
        ats_str = f"{ats_mean*100:.1f}%" if not np.isnan(ats_mean) else "N/A"
        print(f"      {t:>10}: MAE={sub['mae'].mean():.3f}  ATS={ats_str}  "
              f"Brier={sub['brier'].mean():.4f}")

    # Feature set impact
    print(f"\n    Feature Set:")
    for f in rdf["features"].unique():
        sub = rdf[rdf["features"] == f]
        ats_mean = sub["ats_pct"].dropna().mean()
        ats_str = f"{ats_mean*100:.1f}%" if not np.isnan(ats_mean) else "N/A"
        print(f"      {f:>10}: MAE={sub['mae'].mean():.3f}  ATS={ats_str}  "
              f"Brier={sub['brier'].mean():.4f}")

    # Learner combo impact
    print(f"\n    Learner Combos (avg across estimator counts):")
    combo_stats = rdf.groupby("combo").agg(
        mae_mean=("mae", "mean"),
        ats_mean=("ats_pct", lambda x: x.dropna().mean()),
        brier_mean=("brier", "mean"),
    ).sort_values("mae_mean")
    for combo, row in combo_stats.iterrows():
        ats_str = f"{row['ats_mean']*100:.1f}%" if not np.isnan(row['ats_mean']) else "N/A"
        print(f"      {combo:<25}: MAE={row['mae_mean']:.3f}  ATS={ats_str}  "
              f"Brier={row['brier_mean']:.4f}")

    # ── 7. Overall best (weighted score) ──
    print(f"\n  ── OVERALL BEST (0.4×norm_MAE + 0.3×(1-ATS) + 0.3×Brier) ──")
    scored = rdf.copy()
    # Normalize MAE to 0-1 range
    mae_min, mae_max = scored["mae"].min(), scored["mae"].max()
    if mae_max > mae_min:
        scored["norm_mae"] = (scored["mae"] - mae_min) / (mae_max - mae_min)
    else:
        scored["norm_mae"] = 0

    # ATS: higher is better, invert
    scored["ats_score"] = 1.0 - scored["ats_pct"].fillna(0.5)

    # Brier: lower is better
    brier_min, brier_max = scored["brier"].min(), scored["brier"].max()
    if brier_max > brier_min:
        scored["norm_brier"] = (scored["brier"] - brier_min) / (brier_max - brier_min)
    else:
        scored["norm_brier"] = 0

    scored["composite"] = (0.4 * scored["norm_mae"] +
                           0.3 * scored["ats_score"] +
                           0.3 * scored["norm_brier"])

    best5 = scored.nsmallest(5, "composite")
    for _, r in best5.iterrows():
        ats_str = f"{r['ats_pct']*100:.1f}%" if not np.isnan(r['ats_pct']) else "N/A"
        print(f"    Score={r['composite']:.4f}  MAE={r['mae']:.3f}  ATS={ats_str}  "
              f"Brier={r['brier']:.4f}  {r['target']:>8}/{r['features']:>9}  "
              f"{r['combo']:<20}  est={r['n_est']:.0f}")

    # ── Summary ──
    print(f"\n  {'='*90}")
    overall_best = scored.iloc[scored["composite"].idxmin()]
    ats_str = f"{overall_best['ats_pct']*100:.1f}%" if not np.isnan(overall_best['ats_pct']) else "N/A"
    print(f"  ★ RECOMMENDED CONFIG FOR {sport.upper()}:")
    print(f"    Target:   {overall_best['target']}")
    print(f"    Features: {overall_best['features']}")
    print(f"    Combo:    {overall_best['combo']}")
    print(f"    Est:      {overall_best['n_est']:.0f}")
    print(f"    MAE:      {overall_best['mae']:.3f}")
    print(f"    ATS:      {ats_str}")
    print(f"    Brier:    {overall_best['brier']:.4f}")
    clv_str = f"{overall_best['clv']:.3f}" if not np.isnan(overall_best['clv']) else "N/A"
    print(f"    CLV:      {clv_str}")
    print(f"  {'='*90}\n")

    return rdf


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-sport ATS accuracy sweep")
    parser.add_argument("--sport", default="all",
                        choices=["nba", "ncaa", "mlb", "all"],
                        help="Which sport to sweep (default: all)")
    args = parser.parse_args()

    print("=" * 90)
    print("  MEGA SWEEP — Multi-Dimensional ATS Accuracy Improvement")
    print("=" * 90)
    print(f"  Libraries: XGB={'✓' if HAS_XGB else '✗'}  "
          f"CatBoost={'✓' if HAS_CAT else '✗'}  "
          f"LightGBM={'✓' if HAS_LGBM else '✗'}")
    print(f"  Dimensions: target × features × combos × estimators")
    print(f"  Metrics: MAE, ATS%, SU%, Brier, ECE, CLV, High-Conf ATS")

    all_results = {}
    sports = ["nba", "ncaa", "mlb"] if args.sport == "all" else [args.sport]

    for sport in sports:
        try:
            rdf = sweep_sport(sport)
            if rdf is not None:
                all_results[sport] = rdf
        except Exception as e:
            print(f"\n  ERROR in {sport}: {e}")
            import traceback
            traceback.print_exc()

    # ── Cross-sport summary ──
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print(f"  CROSS-SPORT SUMMARY")
        print(f"{'='*90}")
        for sport, rdf in all_results.items():
            best = rdf.loc[rdf["mae"].idxmin()]
            ats_str = f"{best['ats_pct']*100:.1f}%" if not np.isnan(best['ats_pct']) else "N/A"
            print(f"  {sport.upper():>5}: Best MAE={best['mae']:.3f}  ATS={ats_str}  "
                  f"{best['target']}/{best['features']}  {best['combo']}@{best['n_est']}")
