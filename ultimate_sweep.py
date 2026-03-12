#!/usr/bin/env python3
"""
ultimate_sweep.py — Exhaustive Multi-Sport Model Optimization
═════════════════════════════════════════════════════════════════
The most thorough sweep possible. Tests EVERY dimension:

  1. TARGET VARIABLE:     margin, residual (margin - market_spread)
  2. FEATURE SELECTION:   full, ablation (drop each group), correlation-filtered
  3. BASE LEARNERS:       all 31 combos of {XGB, CAT, RF, GBM, LGBM}
  4. META-LEARNER:        Ridge, ElasticNet, simple average
  5. ESTIMATOR COUNT:     50, 75, 100, 125, 150, 175, 200
  6. MAX DEPTH:           3, 4, 5, 6
  7. LEARNING RATE:       0.03, 0.06, 0.10
  8. CV FOLDS:            5, 10, 15, 20
  9. EVALUATION:          MAE, ATS%, SU%, Brier, ECE, CLV, O/U accuracy

Estimated runtime: 4-8 hours per sport locally (no timeout)
Run from sports-predictor-api root:
    python3 ultimate_sweep.py --sport ncaa
    python3 ultimate_sweep.py --sport mlb
    python3 ultimate_sweep.py --sport nba
    python3 ultimate_sweep.py --sport all

    # Quick mode (subset of combos for validation):
    python3 ultimate_sweep.py --sport ncaa --quick
"""
import argparse, time, sys, os, warnings, itertools, json
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.special import expit
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
# FEATURE GROUP DEFINITIONS (for ablation testing)
# ═══════════════════════════════════════════════════════════════

# NCAA feature groups
NCAA_FEATURE_GROUPS = {
    "efficiency":    ["neutral_em_diff", "hca_pts", "neutral"],
    "shooting":      ["ppg_diff", "opp_ppg_diff", "fgpct_diff", "threepct_diff"],
    "four_factors":  ["orb_pct_diff", "fta_rate_diff", "ato_diff"],
    "defense":       ["def_fgpct_diff", "steals_diff", "blocks_diff"],
    "turnover_qual": ["to_margin_diff", "steals_to_diff"],
    "context":       ["sos_diff", "form_diff", "rank_diff", "win_pct_diff"],
    "tempo":         ["tempo_avg"],
    "matchup_type":  ["is_ranked_game", "is_top_matchup"],
    "conference":    ["is_conf_game", "season_phase"],
    "schedule":      ["rest_diff", "either_b2b"],
    "market":        ["market_spread", "market_total", "spread_vs_market", "has_market"],
    "injury":        ["injury_diff", "starters_diff", "any_injury_flag"],
    "tournament":    ["is_conf_tourney", "is_ncaa_tourney", "is_bubble", "is_early", "importance"],
}

# NBA feature groups (derived from nba_build_features)
NBA_FEATURE_GROUPS = {
    "heuristic":     ["score_diff_pred", "total_pred", "home_fav", "win_pct_home", "ou_gap"],
    "net_rating":    ["home_net_rtg", "away_net_rtg", "net_rtg_diff"],
    "shooting":      ["ppg_diff", "opp_ppg_diff", "fgpct_diff", "threepct_diff", "ftpct_diff"],
    "playmaking":    ["assists_diff", "turnovers_diff", "ato_diff"],
    "rebounding":    ["orb_pct_diff", "fta_rate_diff"],
    "defense":       ["opp_fgpct_diff", "opp_threepct_diff", "steals_diff", "blocks_diff"],
    "form":          ["form_diff", "win_pct_diff"],
    "schedule":      ["rest_diff", "away_travel"],
    "market":        ["market_spread", "market_total", "spread_vs_market", "has_market"],
}

# MLB feature groups
MLB_FEATURE_GROUPS = {
    "offense":       ["woba_diff"],
    "pitching":      ["fip_diff", "has_sp_fip", "bullpen_era_diff", "k_bb_diff"],
    "sp_workload":   ["sp_ip_diff", "bp_exposure_diff", "def_oaa_diff"],
    "park_weather":  ["park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold"],
    "context":       ["rest_diff", "travel_diff", "lg_rpg"],
    "interactions":  ["fip_x_bullpen", "woba_x_park", "wind_x_fip"],
    "heuristic":     ["run_diff_pred", "has_heuristic"],
    "enhancements":  ["platoon_diff", "sp_fip_spread", "both_lineups_confirmed"],
    "market":        ["market_spread", "market_total", "spread_vs_market", "has_market"],
}


# ═══════════════════════════════════════════════════════════════
# ENHANCED FEATURE BUILDERS
# ═══════════════════════════════════════════════════════════════

def add_enhanced_features(df, feature_df, sport):
    """Add enhanced features (Elo residual, interactions, etc.)."""
    enhanced = feature_df.copy()

    if sport == "nba":
        score_diff = pd.to_numeric(df.get("pred_home_score", 0), errors="coerce").fillna(0) - \
                     pd.to_numeric(df.get("pred_away_score", 0), errors="coerce").fillna(0)
        mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
        enhanced["elo_residual"] = score_diff - mkt
        enhanced["market_spread_abs"] = mkt.abs()
        net_diff = enhanced.get("net_rtg_diff", pd.Series(0, index=df.index))
        enhanced["net_rtg_diff_sq"] = net_diff ** 2 * np.sign(net_diff)

    elif sport == "ncaa":
        h_em = pd.to_numeric(df.get("home_adj_em", 0), errors="coerce").fillna(0)
        a_em = pd.to_numeric(df.get("away_adj_em", 0), errors="coerce").fillna(0)
        mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
        enhanced["em_residual"] = (h_em - a_em) - mkt
        enhanced["market_spread_abs"] = mkt.abs()
        enhanced["em_diff_sq"] = (h_em - a_em) ** 2 * np.sign(h_em - a_em)
        h_rank = pd.to_numeric(df.get("home_rank", 200), errors="coerce").fillna(200)
        a_rank = pd.to_numeric(df.get("away_rank", 200), errors="coerce").fillna(200)
        enhanced["both_ranked"] = ((h_rank <= 50) & (a_rank <= 50)).astype(int)

        # ── Player-derived features (from ESPN box scores, ~99.9% coverage) ──
        # Star dependency: how reliant is the team on its top scorer?
        h_star1 = pd.to_numeric(df.get("home_star1_pts_share", 0), errors="coerce").fillna(0)
        a_star1 = pd.to_numeric(df.get("away_star1_pts_share", 0), errors="coerce").fillna(0)
        enhanced["star1_dep_diff"] = h_star1 - a_star1

        # Top-3 concentration: are points distributed or top-heavy?
        h_top3 = pd.to_numeric(df.get("home_top3_pts_share", 0), errors="coerce").fillna(0)
        a_top3 = pd.to_numeric(df.get("away_top3_pts_share", 0), errors="coerce").fillna(0)
        enhanced["top3_dep_diff"] = h_top3 - a_top3

        # Bench depth: teams with strong benches handle fatigue/foul trouble better
        h_bench = pd.to_numeric(df.get("home_bench_pts_share", 0), errors="coerce").fillna(0)
        a_bench = pd.to_numeric(df.get("away_bench_pts_share", 0), errors="coerce").fillna(0)
        enhanced["bench_depth_diff"] = h_bench - a_bench

        # Bench raw points differential
        h_bench_pts = pd.to_numeric(df.get("home_bench_pts", 0), errors="coerce").fillna(0)
        a_bench_pts = pd.to_numeric(df.get("away_bench_pts", 0), errors="coerce").fillna(0)
        enhanced["bench_pts_diff"] = h_bench_pts - a_bench_pts

        # Rotation size: more players used = deeper roster
        h_used = pd.to_numeric(df.get("home_players_used", 0), errors="coerce").fillna(0)
        a_used = pd.to_numeric(df.get("away_players_used", 0), errors="coerce").fillna(0)
        enhanced["rotation_diff"] = h_used - a_used

        # Minutes concentration (HHI): higher = more concentrated minutes = less depth
        h_hhi = pd.to_numeric(df.get("home_minutes_hhi", 0), errors="coerce").fillna(0)
        a_hhi = pd.to_numeric(df.get("away_minutes_hhi", 0), errors="coerce").fillna(0)
        enhanced["minutes_hhi_diff"] = h_hhi - a_hhi

        # Interaction: star dependency × market spread (stars matter more in close games)
        enhanced["star_x_spread"] = (h_star1 - a_star1) * mkt.abs().clip(upper=20) / 20

    elif sport == "mlb":
        run_diff = pd.to_numeric(df.get("pred_home_runs", 0), errors="coerce").fillna(0) - \
                   pd.to_numeric(df.get("pred_away_runs", 0), errors="coerce").fillna(0)
        mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
        enhanced["model_residual"] = run_diff - mkt
        enhanced["market_spread_abs"] = mkt.abs()

    return enhanced


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def load_sport_data(sport, min_quality=0.0):
    """Load data for a sport. Returns (df, X_base, y_margin, market_spread, weights, feature_groups).
    
    min_quality: fraction (0.0-1.0) of core signal columns that must have REAL data
                 (not null/default). 0.0 = keep all rows, 1.0 = only fully populated rows.
    """
    # Core signal columns per sport — these are the ones where a null means
    # the model is learning from a league-average constant, not real data.
    QUALITY_COLS = {
        "ncaa": [
            "home_adj_em", "away_adj_em",           # efficiency (the big ones)
            "home_ppg", "away_ppg",                  # scoring
            "home_opp_ppg", "away_opp_ppg",          # defense
            "home_fgpct", "away_fgpct",              # shooting
            "home_threepct", "away_threepct",         # 3pt shooting
            "home_tempo", "away_tempo",               # pace
            "home_orb_pct", "away_orb_pct",           # offensive rebounding
            "home_fta_rate", "away_fta_rate",          # free throw rate
            "home_turnovers", "away_turnovers",        # ball control
            "home_steals", "away_steals",              # defensive disruption
            "home_blocks", "away_blocks",              # rim protection
            "home_ato_ratio", "away_ato_ratio",        # assist/turnover
            "market_spread_home",                      # spread (post ESPN fallback)
            "market_ou_total",                         # over/under (post ESPN fallback)
            "home_minutes_hhi", "away_minutes_hhi",    # rotation concentration
            "home_star1_pts_share", "away_star1_pts_share",  # star dependency
            "home_bench_pts_share", "away_bench_pts_share",  # bench depth
        ],
        "nba": [
            "home_ppg", "away_ppg",
            "home_opp_ppg", "away_opp_ppg",
            "home_fgpct", "away_fgpct",
            "home_threepct", "away_threepct",
            "home_tempo", "away_tempo",
            "home_orb_pct", "away_orb_pct",
            "home_steals", "away_steals",
            "home_blocks", "away_blocks",
        ],
        "mlb": [
            "home_woba", "away_woba",
            "home_sp_fip", "away_sp_fip",
            "home_bullpen_era", "away_bullpen_era",
            "home_k9", "away_k9",
            "home_bb9", "away_bb9",
            "home_sp_ip", "away_sp_ip",
            "park_factor",
        ],
    }

    if sport == "nba":
        from sports.nba import nba_build_features, _nba_merge_historical
        rows = sb_get("nba_predictions", "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        df, weights, _ = _nba_merge_historical(current_df)
        build_fn = nba_build_features
        y_col_h, y_col_a = "actual_home_score", "actual_away_score"
        groups = NBA_FEATURE_GROUPS
        sigma = 11.0

    elif sport == "ncaa":
        from sports.ncaa import ncaa_build_features, _ncaa_merge_historical
        rows = sb_get("ncaa_predictions", "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        df, weights, _ = _ncaa_merge_historical(current_df)
        build_fn = ncaa_build_features
        y_col_h, y_col_a = "actual_home_score", "actual_away_score"
        groups = NCAA_FEATURE_GROUPS
        sigma = 11.0

    elif sport == "mlb":
        from sports.mlb import mlb_build_features, _mlb_merge_historical
        rows = sb_get("mlb_predictions", "result_entered=eq.true&actual_home_runs=not.is.null&game_type=eq.R&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        df, weights = _mlb_merge_historical(current_df)
        build_fn = mlb_build_features
        y_col_h, y_col_a = "actual_home_runs", "actual_away_runs"
        groups = MLB_FEATURE_GROUPS
        sigma = 4.0
    else:
        raise ValueError(f"Unknown sport: {sport}")

    # ── ESPN/DraftKings odds fallback (NCAA) ──
    # market_spread_home from The Odds API has only ~4% coverage.
    # ESPN extraction has DraftKings lines (espn_spread, espn_over_under) at ~63% coverage.
    # Same sign convention confirmed: negative = home favored.
    # Fill market columns from ESPN where Odds API is missing.
    if sport == "ncaa":
        espn_spread_col = "espn_spread"
        espn_ou_col = "espn_over_under"
        espn_ml_home_col = "espn_ml_home"
        espn_ml_away_col = "espn_ml_away"

        n_before_spread = int(pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").notna().sum() -
                              (pd.to_numeric(df.get("market_spread_home", 0), errors="coerce") == 0).sum())
        # More accurate: count rows where market_spread_home is not null AND not 0
        mkt_real = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
        n_odds_api = int(mkt_real.notna().sum() & (mkt_real != 0).sum()) if "market_spread_home" in df.columns else 0
        # Simpler: just count non-null, non-zero
        if "market_spread_home" in df.columns:
            mkt_vals = pd.to_numeric(df["market_spread_home"], errors="coerce")
            n_odds_api = int((mkt_vals.notna() & (mkt_vals != 0)).sum())
        else:
            n_odds_api = 0

        filled_spread = 0
        filled_ou = 0

        if espn_spread_col in df.columns:
            espn_s = pd.to_numeric(df[espn_spread_col], errors="coerce")

            if "market_spread_home" not in df.columns:
                df["market_spread_home"] = np.nan
            mkt_s = pd.to_numeric(df["market_spread_home"], errors="coerce")

            # Fill where Odds API is missing but ESPN has data
            missing_mask = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
            df.loc[missing_mask, "market_spread_home"] = espn_s[missing_mask]
            filled_spread = int(missing_mask.sum())

        if espn_ou_col in df.columns:
            espn_ou = pd.to_numeric(df[espn_ou_col], errors="coerce")

            if "market_ou_total" not in df.columns:
                df["market_ou_total"] = np.nan
            mkt_ou = pd.to_numeric(df["market_ou_total"], errors="coerce")

            missing_mask_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
            df.loc[missing_mask_ou, "market_ou_total"] = espn_ou[missing_mask_ou]
            filled_ou = int(missing_mask_ou.sum())

        n_after_spread = int((pd.to_numeric(df["market_spread_home"], errors="coerce").notna() &
                              (pd.to_numeric(df["market_spread_home"], errors="coerce") != 0)).sum())

        print(f"\n  ESPN/DRAFTKINGS ODDS FALLBACK:")
        print(f"    Spread:  {n_odds_api:,} Odds API + {filled_spread:,} ESPN backfill = {n_after_spread:,} total ({n_after_spread/len(df)*100:.1f}%)")
        print(f"    O/U:     {filled_ou:,} ESPN backfill applied")

    # ── Data Quality Report (BEFORE feature building fills defaults) ──
    qcols = QUALITY_COLS.get(sport, [])
    present_in_df = [c for c in qcols if c in df.columns]
    n_total = len(df)

    if present_in_df:
        # Count non-null values per quality column
        print(f"\n  DATA QUALITY REPORT — {sport.upper()} ({n_total:,} games)")
        print(f"  {'Column':<30} {'Real':>8} {'Default':>8} {'Coverage':>8}")
        print(f"  {'-'*58}")

        # Build a matrix: True = real data, False = null/will-be-defaulted
        quality_matrix = pd.DataFrame(index=df.index)
        for col in present_in_df:
            is_real = df[col].notna()
            # Also flag sentinel values that are effectively "no data"
            if col in ["home_adj_em", "away_adj_em"]:
                is_real = is_real & (pd.to_numeric(df[col], errors="coerce") != 0)
            if col in ["market_spread_home", "market_ou_total"]:
                is_real = is_real & (pd.to_numeric(df[col], errors="coerce") != 0)
            n_real = int(is_real.sum())
            n_default = n_total - n_real
            pct = n_real / n_total * 100
            print(f"  {col:<30} {n_real:>8,} {n_default:>8,} {pct:>7.1f}%")
            quality_matrix[col] = is_real

        # Per-row quality score: fraction of signal columns with real data
        row_quality = quality_matrix.mean(axis=1)  # 0.0 to 1.0

        # Quality distribution
        print(f"\n  ROW QUALITY DISTRIBUTION:")
        for threshold in [1.0, 0.9, 0.8, 0.7, 0.5, 0.25, 0.0]:
            n_above = int((row_quality >= threshold).sum())
            print(f"    ≥{threshold*100:>5.0f}% real: {n_above:>8,} games ({n_above/n_total*100:.1f}%)")

        # Apply quality filter
        if min_quality > 0:
            quality_mask = (row_quality >= min_quality)
            n_keep = int(quality_mask.sum())
            n_drop = n_total - n_keep
            print(f"\n  QUALITY FILTER: min_quality={min_quality:.0%}")
            print(f"    Keeping {n_keep:,} games, dropping {n_drop:,} "
                  f"({n_drop/n_total*100:.1f}% below threshold)")

            df = df.loc[quality_mask].reset_index(drop=True)
            if weights is not None:
                if isinstance(weights, np.ndarray):
                    weights = weights[quality_mask.values]
                elif isinstance(weights, pd.Series):
                    weights = weights.loc[quality_mask].reset_index(drop=True).values
        else:
            print(f"\n  QUALITY FILTER: OFF (min_quality=0%, all {n_total:,} games used)")

    # Build features AFTER filtering
    X = build_fn(df)
    y = df[y_col_h].astype(float) - df[y_col_a].astype(float)

    mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    w = weights if weights is not None else np.ones(len(df))
    if isinstance(w, pd.Series):
        w = w.values

    return df, X, y, mkt, w, groups, sigma


# ═══════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════

def evaluate_predictions(pred_margin, y_actual, market_spread, sigma):
    """Compute all metrics for a set of predictions."""
    results = {}

    # MAE
    results["mae"] = float(mean_absolute_error(y_actual, pred_margin))

    # SU accuracy
    non_tie = (y_actual != 0)
    if non_tie.sum() > 10:
        results["su_acc"] = float(((pred_margin > 0) == (y_actual > 0))[non_tie].mean())
    else:
        results["su_acc"] = np.nan

    # Win probability + Brier
    pred_wp = expit(pred_margin / sigma)
    y_win = (y_actual > 0).astype(int)
    results["brier"] = float(brier_score_loss(y_win, pred_wp))

    # ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (pred_wp >= bins[i]) & (pred_wp < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(pred_wp)) * abs(pred_wp[mask].mean() - y_win[mask].mean())
    results["ece"] = ece

    # ATS accuracy (only games with market data)
    has_mkt = (market_spread != 0)
    if has_mkt.sum() >= 20:
        model_edge = pred_margin - market_spread
        actual_cover = y_actual - market_spread
        non_push = has_mkt & (actual_cover != 0)
        if non_push.sum() >= 20:
            results["ats"] = float(((model_edge[non_push] > 0) == (actual_cover[non_push] > 0)).mean())
        else:
            results["ats"] = np.nan
    else:
        results["ats"] = np.nan

    # O/U accuracy (using total prediction vs market total)
    # We use the heuristic's total prediction (pred_home + pred_away) since
    # the sweep optimizes margin, not total. This measures whether the model's
    # implied total (via margin prediction + base total) beats the market.
    results["ou_acc"] = np.nan  # requires total data passed separately

    # CLV correlation
    if has_mkt.sum() >= 20:
        edge = pred_margin - market_spread
        reality = y_actual - market_spread
        valid = has_mkt & np.isfinite(edge) & np.isfinite(reality)
        if valid.sum() >= 20:
            results["clv"] = float(np.corrcoef(edge[valid], reality[valid])[0, 1])
        else:
            results["clv"] = np.nan
    else:
        results["clv"] = np.nan

    # High-confidence ATS (disagree with market by > 2pts)
    if has_mkt.sum() >= 20:
        disagree = np.abs(pred_margin - market_spread)
        high_conf = has_mkt & (disagree > 2.0)
        if high_conf.sum() >= 15:
            hc_cover = y_actual[high_conf] - market_spread[high_conf]
            hc_pick = pred_margin[high_conf] - market_spread[high_conf]
            non_push = (hc_cover != 0)
            if non_push.sum() >= 10:
                results["hc_ats"] = float(((hc_pick[non_push] > 0) == (hc_cover[non_push] > 0)).mean())
                results["hc_n"] = int(high_conf.sum())
            else:
                results["hc_ats"] = np.nan
                results["hc_n"] = 0
        else:
            results["hc_ats"] = np.nan
            results["hc_n"] = 0
    else:
        results["hc_ats"] = np.nan
        results["hc_n"] = 0

    return results


# ═══════════════════════════════════════════════════════════════
# MODEL BUILDER
# ═══════════════════════════════════════════════════════════════

def build_models(combo, n_est, max_depth, lr):
    """Create model dict from combo spec."""
    models = {}
    if "RF" in combo:
        models["rf"] = RandomForestRegressor(
            n_estimators=n_est, max_depth=max_depth + 2,  # RF benefits from deeper trees
            min_samples_leaf=15, max_features=0.7, random_state=42, n_jobs=-1)
    if "XGB" in combo and HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=42, tree_method="hist", verbosity=0)
    if "CAT" in combo and HAS_CAT:
        models["cat"] = CatBoostRegressor(
            iterations=n_est, depth=max_depth, learning_rate=lr,
            subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0)
    if "GBM" in combo:
        models["gbm"] = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
            subsample=0.8, min_samples_leaf=20, random_state=42)
    if "LGBM" in combo and HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, verbose=-1)
    return models


def run_config(X, y_target, y_actual, market_spread, df, weights,
               combo, n_est, max_depth, lr, cv_folds, meta_type, sigma):
    """Run one sweep configuration and return metrics dict."""
    t0 = time.time()
    models = build_models(combo, n_est, max_depth, lr)
    if not models:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        oof = _time_series_oof(models, X_scaled, y_target, df,
                               n_splits=cv_folds, weights=weights)
    except Exception as e:
        return {"error": str(e)}

    # Stack or single
    if len(models) == 1:
        name = list(models.keys())[0]
        oof_pred = oof[name]
    else:
        for name, model in models.items():
            model.fit(X_scaled, y_target, sample_weight=weights)

        meta_X = np.column_stack([oof[k] for k in models.keys()])

        if meta_type == "ridge":
            meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            meta.fit(meta_X, y_target)
            oof_pred = meta.predict(meta_X)
        elif meta_type == "elasticnet":
            meta = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                alphas=[0.01, 0.1, 1.0, 10.0], cv=5, random_state=42)
            meta.fit(meta_X, y_target)
            oof_pred = meta.predict(meta_X)
        else:  # simple average
            oof_pred = meta_X.mean(axis=1)

    # Convert back to margin space if residual target
    is_residual = not np.allclose(y_target.values[:100], y_actual.values[:100], atol=0.01)
    pred_margin = oof_pred + market_spread.values if is_residual else oof_pred

    metrics = evaluate_predictions(pred_margin, y_actual.values, market_spread.values, sigma)
    metrics["elapsed"] = time.time() - t0

    return metrics


# ═══════════════════════════════════════════════════════════════
# PHASE 1: FEATURE CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def feature_correlation_analysis(X, y_margin, market_spread, sport):
    """Analyze correlation of each feature with target variables."""
    print(f"\n{'='*80}")
    print(f"  PHASE 1: FEATURE CORRELATION ANALYSIS — {sport.upper()}")
    print(f"{'='*80}")

    y_ats = y_margin.values - market_spread.values
    has_mkt = (market_spread.values != 0)

    results = []
    for col in X.columns:
        vals = X[col].values
        if np.std(vals) < 1e-10:
            results.append({"feature": col, "margin_corr": 0, "ats_corr": 0, "variance": 0, "flag": "ZERO_VAR"})
            continue

        margin_corr = float(np.corrcoef(vals, y_margin.values)[0, 1]) if np.std(y_margin) > 0 else 0
        ats_corr = float(np.corrcoef(vals[has_mkt], y_ats[has_mkt])[0, 1]) if has_mkt.sum() > 20 and np.std(y_ats[has_mkt]) > 0 else 0

        flag = ""
        if abs(margin_corr) < 0.01 and abs(ats_corr) < 0.01:
            flag = "WEAK"
        elif margin_corr * ats_corr < 0 and abs(margin_corr) > 0.05:
            flag = "CONFLICTING"

        results.append({
            "feature": col,
            "margin_corr": round(margin_corr, 4),
            "ats_corr": round(ats_corr, 4),
            "variance": round(float(np.std(vals)), 4),
            "flag": flag,
        })

    # Sort by absolute margin correlation
    results.sort(key=lambda r: abs(r["margin_corr"]), reverse=True)

    print(f"\n  {'Feature':<30} {'Margin r':>10} {'ATS r':>10} {'Std':>10} {'Flag':>12}")
    print(f"  {'-'*75}")
    for r in results:
        flag_str = f"  ⚠️ {r['flag']}" if r["flag"] else ""
        print(f"  {r['feature']:<30} {r['margin_corr']:>10.4f} {r['ats_corr']:>10.4f} "
              f"{r['variance']:>10.4f}{flag_str}")

    # Identify features to potentially drop
    weak = [r["feature"] for r in results if r["flag"] == "WEAK"]
    zero_var = [r["feature"] for r in results if r["flag"] == "ZERO_VAR"]
    conflicting = [r["feature"] for r in results if r["flag"] == "CONFLICTING"]

    print(f"\n  Summary:")
    print(f"    Total features: {len(results)}")
    print(f"    Zero variance (DEFINITELY drop): {len(zero_var)} — {zero_var[:5]}")
    print(f"    Weak signal (test dropping): {len(weak)} — {weak[:5]}")
    print(f"    Conflicting ATS vs margin: {len(conflicting)} — {conflicting[:5]}")

    return results, zero_var, weak, conflicting


# ═══════════════════════════════════════════════════════════════
# PHASE 2: FEATURE ABLATION
# ═══════════════════════════════════════════════════════════════

def feature_ablation(X, y_margin, y_residual, market_spread, df, weights,
                     feature_groups, sigma, sport):
    """Test dropping each feature group to see impact."""
    print(f"\n{'='*80}")
    print(f"  PHASE 2: FEATURE GROUP ABLATION — {sport.upper()}")
    print(f"{'='*80}")
    print(f"  Testing what happens when each feature group is removed.\n")

    # Baseline: all features, default config
    combo = ["XGB", "CAT", "RF"]
    combo = [c for c in combo if (c != "XGB" or HAS_XGB) and (c != "CAT" or HAS_CAT)]
    if not combo:
        combo = ["GBM", "RF"]

    print(f"  Baseline model: {'+'.join(combo)} @ 100 est, depth=4, lr=0.06, 10-fold")

    baseline = run_config(X, y_margin, y_margin, market_spread, df, weights,
                          combo, 100, 4, 0.06, 10, "ridge", sigma)
    if baseline is None or "error" in baseline:
        print(f"  Baseline failed!")
        return {}

    ats_str = f"{baseline['ats']*100:.1f}%" if not np.isnan(baseline.get('ats', np.nan)) else "N/A"
    print(f"  Baseline: MAE={baseline['mae']:.3f}  ATS={ats_str}  Brier={baseline['brier']:.4f}\n")

    print(f"  {'Group':<20} {'MAE':>8} {'ΔMAE':>8} {'ATS':>8} {'ΔATS':>8} {'Brier':>8} {'Verdict':>10}")
    print(f"  {'-'*78}")

    ablation_results = {"_baseline": baseline}

    for group_name, group_cols in feature_groups.items():
        # Find which columns actually exist in X
        drop_cols = [c for c in group_cols if c in X.columns]
        if not drop_cols:
            print(f"  {group_name:<20} — no matching columns in feature set")
            continue

        keep_cols = [c for c in X.columns if c not in drop_cols]
        if len(keep_cols) < 3:
            print(f"  {group_name:<20} — would leave <3 features, skipping")
            continue

        X_ablated = X[keep_cols]
        r = run_config(X_ablated, y_margin, y_margin, market_spread, df, weights,
                       combo, 100, 4, 0.06, 10, "ridge", sigma)

        if r is None or "error" in r:
            print(f"  {group_name:<20} — ERROR")
            continue

        d_mae = r["mae"] - baseline["mae"]
        d_ats = (r.get("ats", 0.5) - baseline.get("ats", 0.5)) * 100 if not np.isnan(r.get("ats", np.nan)) else 0
        ats_str = f"{r['ats']*100:.1f}%" if not np.isnan(r.get("ats", np.nan)) else "N/A"

        if d_mae < -0.05:
            verdict = "✅ DROP"
        elif d_mae > 0.10:
            verdict = "🔒 KEEP"
        else:
            verdict = "— neutral"

        print(f"  {group_name:<20} {r['mae']:>8.3f} {d_mae:>+8.3f} {ats_str:>8} {d_ats:>+7.1f}% "
              f"{r['brier']:>8.4f} {verdict:>10}")

        ablation_results[group_name] = {**r, "dropped": drop_cols, "delta_mae": d_mae}

    return ablation_results


# ═══════════════════════════════════════════════════════════════
# PHASE 3: EXHAUSTIVE MODEL SWEEP
# ═══════════════════════════════════════════════════════════════

def exhaustive_sweep(X, y_margin, y_residual, market_spread, df, weights,
                     sigma, sport, quick=False, resume=False, feature_filter=None):
    """The big one. Tests all combinations."""
    print(f"\n{'='*80}")
    print(f"  PHASE 3: EXHAUSTIVE MODEL SWEEP — {sport.upper()}")
    print(f"{'='*80}")

    # ── Define all dimensions ──
    available_learners = []
    if HAS_XGB: available_learners.append("XGB")
    if HAS_CAT: available_learners.append("CAT")
    # RF and GBM dropped — too slow on 60K rows, XGB+CAT+LGBM outperforms
    # available_learners.append("RF")
    # available_learners.append("GBM")
    if HAS_LGBM: available_learners.append("LGBM")

    # All non-empty subsets of available learners
    all_combos = []
    for r in range(1, len(available_learners) + 1):
        for combo in itertools.combinations(available_learners, r):
            all_combos.append(list(combo))

    # Target configs
    targets = [("margin", y_margin)]
    has_mkt = (market_spread != 0).sum()
    if has_mkt > len(df) * 0.4:
        targets.append(("residual", y_residual))

    # Feature configs
    X_enhanced = add_enhanced_features(df, X, sport)

    # Build pruned feature set (drop zero-variance + weak from Phase 1)
    zero_var_cols = [col for col in X.columns if X[col].std() < 1e-10]
    weak_cols = []
    y_ats_check = y_margin.values - market_spread.values
    has_mkt_mask = (market_spread.values != 0)
    for col in X.columns:
        if col in zero_var_cols:
            continue
        vals = X[col].values
        if np.std(vals) < 1e-10:
            continue
        m_corr = abs(float(np.corrcoef(vals, y_margin.values)[0, 1]))
        a_corr = abs(float(np.corrcoef(vals[has_mkt_mask], y_ats_check[has_mkt_mask])[0, 1])) if has_mkt_mask.sum() > 20 else 0
        if m_corr < 0.01 and a_corr < 0.01:
            weak_cols.append(col)

    drop_cols = set(zero_var_cols + weak_cols)
    pruned_cols = [c for c in X.columns if c not in drop_cols]
    X_pruned = X[pruned_cols] if len(pruned_cols) >= 5 else X

    feature_configs = [("baseline", X), ("enhanced", X_enhanced)]
    if len(drop_cols) >= 2:
        feature_configs.append(("pruned", X_pruned))
        print(f"  Pruned feature set: dropped {len(drop_cols)} cols "
              f"({len(zero_var_cols)} zero-var, {len(weak_cols)} weak) → {len(pruned_cols)} remaining")

    # Filter to single feature set for parallel runs
    if feature_filter:
        feature_configs = [(n, x) for n, x in feature_configs if n == feature_filter]
        if not feature_configs:
            print(f"  ERROR: --features {feature_filter} not found. Available: baseline, enhanced, pruned")
            return []
        print(f"  PARALLEL MODE: running only '{feature_filter}' feature set")

    if quick:
        # Quick mode: fewer combos for validation
        all_combos = [
            ["XGB", "CAT", "RF"] if HAS_XGB and HAS_CAT else ["GBM", "RF"],
            ["LGBM"] if HAS_LGBM else ["GBM"],
            ["XGB", "CAT", "LGBM"] if HAS_XGB and HAS_CAT and HAS_LGBM else ["GBM", "RF"],
        ]
        estimators = [100, 150]
        depths = [4]
        lrs = [0.06]
        folds_list = [10]
        meta_types = ["ridge"]
    else:
        estimators = [50, 75, 100, 125, 150, 175]
        depths = [3, 4, 5]
        lrs = [0.03, 0.06, 0.10]
        folds_list = [5, 10]
        meta_types = ["ridge"]

    # Count total configs
    total = (len(targets) * len(feature_configs) * len(all_combos) *
             len(estimators) * len(depths) * len(lrs) * len(folds_list) * len(meta_types))
    # For single models, meta_type doesn't matter — reduce count
    single_combos = sum(1 for c in all_combos if len(c) == 1)
    multi_combos = len(all_combos) - single_combos
    actual_total = (len(targets) * len(feature_configs) *
                    (single_combos * len(estimators) * len(depths) * len(lrs) * len(folds_list) +
                     multi_combos * len(estimators) * len(depths) * len(lrs) * len(folds_list) * len(meta_types)))

    print(f"\n  Learner combos:  {len(all_combos)}")
    print(f"  Targets:         {len(targets)}")
    print(f"  Feature sets:    {len(feature_configs)}")
    print(f"  Estimators:      {estimators}")
    print(f"  Depths:          {depths}")
    print(f"  Learning rates:  {lrs}")
    print(f"  CV folds:        {folds_list}")
    print(f"  Meta-learners:   {meta_types}")
    print(f"  Total configs:   ~{actual_total:,}")

    # Estimate time
    est_per_config = 3.0 if sport == "mlb" else 1.5  # seconds
    est_hours = (actual_total * est_per_config) / 3600
    print(f"  Est. runtime:    ~{est_hours:.1f} hours\n")

    # ── Resume from checkpoint ──
    results = []
    completed_keys = set()
    skip_count = 0

    # Use feature-specific checkpoint for parallel runs
    cp_suffix = f"_{feature_filter}" if feature_filter else ""
    cp_partial = f"sweep_checkpoint_{sport}{cp_suffix}_partial.json"

    if resume:
        if os.path.exists(cp_partial):
            with open(cp_partial) as f:
                results = json.load(f)
            # Build set of completed config keys for fast lookup
            for r in results:
                key = (r.get("target",""), r.get("features",""), r.get("combo",""),
                       r.get("n_est",0), r.get("depth",0), r.get("lr",0),
                       r.get("folds",0), r.get("meta",""))
                completed_keys.add(key)
            best_mae = min((r["mae"] for r in results), default=999)
            best_ats = max((r.get("ats",0) or 0 for r in results), default=0)
            print(f"  ▶ RESUMING: loaded {len(results)} completed configs from {cp_partial}")
            print(f"    Best so far: MAE={best_mae:.3f}, ATS={best_ats*100:.1f}%")
        else:
            print(f"  ▶ Resume requested but no checkpoint found ({cp_partial}) — starting fresh")

    idx = 0
    if not resume:
        best_mae = 999
        best_ats = 0
    start_time = time.time()

    for target_name, y_target in targets:
        for feat_name, X_feat in feature_configs:
            for combo in all_combos:
                # Skip unavailable combos
                if "XGB" in combo and not HAS_XGB: continue
                if "CAT" in combo and not HAS_CAT: continue
                if "LGBM" in combo and not HAS_LGBM: continue

                is_single = len(combo) == 1
                meta_list = ["ridge"] if is_single else meta_types

                for n_est in estimators:
                    for depth in depths:
                        for lr in lrs:
                            for folds in folds_list:
                                for meta in meta_list:
                                    idx += 1
                                    label = "+".join(combo)

                                    # Skip if already completed (resume mode)
                                    if resume and completed_keys:
                                        key = (target_name, feat_name, label,
                                               n_est, depth, lr, folds, meta)
                                        if key in completed_keys:
                                            skip_count += 1
                                            continue

                                    try:
                                        r = run_config(
                                            X_feat, y_target, y_margin, market_spread,
                                            df, weights, combo, n_est, depth, lr,
                                            folds, meta, sigma
                                        )
                                        if r is None or "error" in r:
                                            continue

                                        r.update({
                                            "target": target_name,
                                            "features": feat_name,
                                            "combo": label,
                                            "n_est": n_est,
                                            "depth": depth,
                                            "lr": lr,
                                            "folds": folds,
                                            "meta": meta,
                                        })
                                        results.append(r)

                                        # Track bests
                                        is_best_mae = r["mae"] < best_mae
                                        is_best_ats = (r.get("ats", 0) or 0) > best_ats
                                        if is_best_mae:
                                            best_mae = r["mae"]
                                        if (r.get("ats", 0) or 0) > best_ats:
                                            best_ats = r.get("ats", 0) or 0

                                        # Print progress every 50 configs + any new bests
                                        if idx % 50 == 0 or is_best_mae or is_best_ats:
                                            elapsed = time.time() - start_time
                                            rate = idx / elapsed if elapsed > 0 else 0
                                            eta = (actual_total - idx) / rate / 60 if rate > 0 else 0
                                            ats_s = f"{r['ats']*100:.1f}%" if not np.isnan(r.get('ats', np.nan)) else "N/A"
                                            marker = ""
                                            if is_best_mae: marker += " ★MAE"
                                            if is_best_ats: marker += " ★ATS"
                                            print(f"  [{idx:>6}/{actual_total}] MAE={r['mae']:.3f} ATS={ats_s:>6} "
                                                  f"{target_name}/{feat_name} {label} "
                                                  f"e={n_est} d={depth} lr={lr} f={folds} m={meta} "
                                                  f"({eta:.0f}min left){marker}")

                                        # Checkpoint every 500 configs
                                        if idx % 500 == 0 and results:
                                            with open(cp_partial, "w") as f:
                                                json.dump(results, f)
                                            print(f"  [CHECKPOINT] {len(results)} results saved to {cp_partial}")

                                    except Exception as e:
                                        if idx % 100 == 0:
                                            print(f"  [{idx}] ERROR: {str(e)[:60]}")

    total_time = time.time() - start_time
    resumed_msg = f" (skipped {skip_count} already-completed)" if skip_count else ""
    print(f"\n  Sweep complete: {len(results)} total configs in {total_time/60:.1f} minutes{resumed_msg}")

    # Save checkpoint
    if results:
        checkpoint_path = f"sweep_checkpoint_{sport}{cp_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)
        print(f"  Final checkpoint saved: {checkpoint_path}")

    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 4: RESULTS ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_results(results, sport):
    """Comprehensive analysis of sweep results."""
    if not results:
        print("  No results to analyze!")
        return

    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"  PHASE 4: RESULTS ANALYSIS — {sport.upper()}")
    print(f"  ({len(df)} configurations tested)")
    print(f"{'='*80}")

    # ── Top 10 by each metric ──
    for metric, ascending, label in [
        ("mae", True, "MAE (lower = better)"),
        ("ats", False, "ATS Accuracy (higher = better)"),
        ("brier", True, "Brier Score (lower = better)"),
        ("clv", False, "CLV Correlation (higher = better)"),
        ("hc_ats", False, "High-Confidence ATS (higher = better)"),
    ]:
        valid = df[df[metric].notna()] if metric in df.columns else pd.DataFrame()
        if len(valid) < 5:
            continue

        top = valid.nsmallest(10, metric) if ascending else valid.nlargest(10, metric)
        print(f"\n  ── TOP 10 BY {label} ──")
        for i, (_, r) in enumerate(top.iterrows()):
            ats_s = f"{r['ats']*100:.1f}%" if not np.isnan(r.get('ats', np.nan)) else "N/A"
            print(f"    {i+1:>2}. {metric}={r[metric]:.4f}  MAE={r['mae']:.3f}  ATS={ats_s}  "
                  f"{r['target']}/{r['features']} {r['combo']} "
                  f"e={r['n_est']:.0f} d={r['depth']:.0f} lr={r['lr']} f={r['folds']:.0f} m={r['meta']}")

    # ── Dimension impact analysis ──
    print(f"\n  ── DIMENSION IMPACT (average across configs) ──")
    for dim, label in [("target", "Target Variable"), ("features", "Feature Set"),
                       ("combo", "Learner Combo"), ("depth", "Max Depth"),
                       ("lr", "Learning Rate"), ("folds", "CV Folds"),
                       ("meta", "Meta-Learner"), ("n_est", "Estimators")]:
        if dim not in df.columns:
            continue
        print(f"\n    {label}:")
        grouped = df.groupby(dim).agg(
            mae_mean=("mae", "mean"), mae_min=("mae", "min"),
            ats_mean=("ats", lambda x: x.dropna().mean()),
            brier_mean=("brier", "mean"),
            count=("mae", "count"),
        ).sort_values("mae_mean")
        for val, row in grouped.iterrows():
            ats_s = f"{row['ats_mean']*100:.1f}%" if not np.isnan(row['ats_mean']) else "N/A"
            print(f"      {str(val):<25} MAE={row['mae_mean']:.3f} (best {row['mae_min']:.3f})  "
                  f"ATS={ats_s}  Brier={row['brier_mean']:.4f}  n={row['count']:.0f}")

    # ── Composite best ──
    print(f"\n  ── OVERALL BEST (0.4×MAE + 0.3×(1-ATS) + 0.3×Brier) ──")
    scored = df.copy()
    mae_min, mae_max = scored["mae"].min(), scored["mae"].max()
    scored["norm_mae"] = (scored["mae"] - mae_min) / (mae_max - mae_min) if mae_max > mae_min else 0
    scored["ats_inv"] = 1.0 - scored["ats"].fillna(0.5)
    b_min, b_max = scored["brier"].min(), scored["brier"].max()
    scored["norm_brier"] = (scored["brier"] - b_min) / (b_max - b_min) if b_max > b_min else 0
    scored["composite"] = 0.4 * scored["norm_mae"] + 0.3 * scored["ats_inv"] + 0.3 * scored["norm_brier"]

    best10 = scored.nsmallest(10, "composite")
    for i, (_, r) in enumerate(best10.iterrows()):
        ats_s = f"{r['ats']*100:.1f}%" if not np.isnan(r.get('ats', np.nan)) else "N/A"
        print(f"    {i+1:>2}. Score={r['composite']:.4f}  MAE={r['mae']:.3f}  ATS={ats_s}  "
              f"Brier={r['brier']:.4f}")
        print(f"        {r['target']}/{r['features']} {r['combo']} "
              f"e={r['n_est']:.0f} d={r['depth']:.0f} lr={r['lr']} f={r['folds']:.0f} m={r['meta']}")

    # ── Save full results to CSV ──
    csv_path = f"sweep_results_{sport}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Full results saved to: {csv_path}")

    return scored


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_sport(sport, quick=False, resume=False, min_quality=0.0, feature_filter=None):
    """Run all phases for one sport."""
    print(f"\n{'#'*80}")
    print(f"  ULTIMATE SWEEP: {sport.upper()}")
    print(f"  {'Quick mode' if quick else 'Full exhaustive mode'}{'  +RESUME' if resume else ''}")
    if min_quality > 0:
        print(f"  Quality filter: ≥{min_quality*100:.0f}% of core signal columns must be real data")
    print(f"{'#'*80}")

    # Load data
    print(f"\n  Loading {sport.upper()} data...")
    df, X, y_margin, market_spread, weights, feature_groups, sigma = load_sport_data(sport, min_quality=min_quality)
    print(f"  Dataset: {len(df)} rows, {len(X.columns)} features")

    # Residual target
    y_residual = y_margin - market_spread
    has_mkt = (market_spread != 0).sum()
    print(f"  Market data: {has_mkt}/{len(df)} ({has_mkt/len(df)*100:.0f}%)")

    # Phase 1: Correlation analysis
    corr_results, zero_var, weak, conflicting = feature_correlation_analysis(
        X, y_margin, market_spread, sport)

    # Phase 2: Feature ablation
    ablation = feature_ablation(X, y_margin, y_residual, market_spread, df, weights,
                                feature_groups, sigma, sport)

    # Phase 3: Exhaustive sweep
    results = exhaustive_sweep(X, y_margin, y_residual, market_spread, df, weights,
                               sigma, sport, quick=quick, resume=resume,
                               feature_filter=feature_filter)

    # Phase 4: Analysis
    scored = analyze_results(results, sport)

    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate multi-sport model sweep")
    parser.add_argument("--sport", default="all", choices=["nba", "ncaa", "mlb", "all"])
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer combos)")
    parser.add_argument("--resume", action="store_true", help="Resume from partial checkpoint")
    parser.add_argument("--min-quality", type=float, default=0.0,
                        help="Min fraction of core signal columns with real data (0.0-1.0). "
                             "E.g., 0.5 = at least 50%% of signal columns must be non-null. "
                             "Default 0.0 keeps all rows.")
    parser.add_argument("--features", type=str, default=None,
                        choices=["baseline", "enhanced", "pruned"],
                        help="Run only one feature set (for parallel runs across terminals)")
    args = parser.parse_args()

    print("=" * 80)
    print("  ULTIMATE SWEEP — Exhaustive Multi-Dimensional Optimization")
    print("=" * 80)
    print(f"  Libraries: XGB={'✓' if HAS_XGB else '✗'}  CAT={'✓' if HAS_CAT else '✗'}  "
          f"LGBM={'✓' if HAS_LGBM else '✗'}")
    print(f"  Mode: {'Quick' if args.quick else 'Full Exhaustive'}"
          f"{'  +RESUME' if args.resume else ''}"
          f"{'  Quality≥' + str(int(args.min_quality*100)) + '%' if args.min_quality > 0 else ''}"
          f"{'  Features=' + args.features if args.features else ''}")

    sports = ["ncaa", "mlb", "nba"] if args.sport == "all" else [args.sport]

    for sport in sports:
        try:
            run_sport(sport, quick=args.quick, resume=args.resume,
                      min_quality=args.min_quality, feature_filter=args.features)
        except Exception as e:
            print(f"\n  FATAL ERROR in {sport}: {e}")
            import traceback
            traceback.print_exc()
