#!/usr/bin/env python3
"""
nba_parity_audit.py — Full forensic audit of train/serve feature parity
========================================================================
Checks EVERY feature for:
  1. Default value mismatches (train defaults != serve defaults)
  2. Formula mismatches (different computation paths)
  3. Static constants / hardcoded guesses
  4. Data source gaps (train has data, serve doesn't)
  5. Feature coverage (% of games where feature is non-zero)

Covers: ATS model (38 features) + O/U model (45 res + 12 cls + 60 ats)

Usage:
    python3 nba_parity_audit.py                # Full audit
    python3 nba_parity_audit.py --live-test    # Also test live API
"""
import sys, os, argparse, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RED = "\033[91m"; YEL = "\033[93m"; GRN = "\033[92m"; RST = "\033[0m"


def audit_defaults():
    """Audit hardcoded defaults across all three files."""
    print("=" * 70)
    print("  AUDIT 1: HARDCODED DEFAULTS & STATIC CONSTANTS")
    print("=" * 70)

    # Every default value used in each file
    # Format: (feature_context, file, default_value, concern)
    mismatches = [
        # PPG defaults
        ("home_ppg / away_ppg", "nba_build_features_v27.py", "0 (no explicit default)",
         "nba_v27_features_live.py", "110",
         "nba_full_predict.py setdefault", "112",
         "THREE different defaults for same stat"),

        # FG% defaults
        ("home_fgpct / away_fgpct", "nba_build_features_v27.py", "0.46",
         "nba_v27_features_live.py", "0.46",
         "nba_full_predict.py setdefault", "0.471",
         "0.46 vs 0.471 — serve path overrides with different league avg"),

        # 3PT% defaults
        ("home_threepct / away_threepct", "nba_build_features_v27.py", "0.36",
         "nba_v27_features_live.py", "0.365",
         "nba_full_predict.py", "0.365",
         "0.36 vs 0.365 — small but systematic"),

        # OPP PPG defaults
        ("home_opp_ppg / away_opp_ppg", "nba_build_features_v27.py", "0",
         "nba_v27_features_live.py", "112",
         "nba_full_predict.py", "112",
         "0 vs 112 — training defaults to 0, serve to 112"),

        # FT% defaults
        ("home_ftpct / away_ftpct", "nba_build_features_v27.py", "0 (no default)",
         "nba_v27_features_live.py", "0.77",
         "nba_full_predict.py", "0.78",
         "0.77 vs 0.78"),

        # Market O/U total
        ("market_ou_total", "nba_build_features_v27.py", "0",
         "nba_v27_features_live.py", "0",
         "nba_full_predict.py", "228",
         "228 is a HARDCODED GUESS when no market data"),

        # TS% league average
        ("league_avg_ts (for ts_regression)", "training", "computed from data",
         "nba_v27_features_live.py", "0.575 (parameter)",
         "nba_full_predict.py", "0.575 (static fallback)",
         "Static 0.575 if dynamic_constants.py unavailable"),
    ]

    print(f"\n  {'Context':40s} {'Train':>10s} {'Live':>10s} {'Predict':>10s} {'Issue'}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*30}")

    n_issues = 0
    for ctx, f1, v1, f2, v2, f3, v3, issue in mismatches:
        is_mismatch = not (v1 == v2 == v3)
        color = RED if is_mismatch else GRN
        symbol = "❌" if is_mismatch else "✅"
        print(f"  {color}{ctx:40s} {v1:>10s} {v2:>10s} {v3:>10s}{RST}  {symbol} {issue}")
        if is_mismatch:
            n_issues += 1

    # Additional static constants
    print(f"\n  STATIC CONSTANTS (hardcoded values that should be dynamic):")
    statics = [
        ("crowd_pct fill rates", "nba_full_predict.py", "Team-specific dict", "OK if updated seasonally"),
        ("VENUE_CAPACITY dict", "nba_full_predict.py", "30 teams hardcoded", "Should update if arena changes"),
        ("PUBLIC_TEAMS set", "nba_v27_features_live.py", "15 teams", "Subjective, rarely changes"),
        ("NBA_CONFERENCES", "nba_v27_features_live.py", "30 teams", "Static, correct"),
        ("TRAIN_RANGES clamps", "nba_v27_features_live.py", "36 feature ranges", "OK — safety net, not default"),
        ("days_since_loss streak×2.2", "nba_v27_features_live.py", "2.2 multiplier", "HEURISTIC — should be validated"),
        ("three_pt_regression fallback", "nba_v27_features_live.py", "3pct - 0.365", "OK — regression to mean"),
        ("sigma=7.0 (win prob)", "nba_full_predict.py", "7.0", "VALIDATED via Brier sweep ✅"),
    ]
    for name, file, val, concern in statics:
        print(f"    {name:40s} {val:>20s}  {concern}")

    print(f"\n  Total default mismatches: {n_issues}")
    return n_issues


def audit_feature_formulas():
    """Compare feature computation formulas between train and serve."""
    print("\n" + "=" * 70)
    print("  AUDIT 2: FORMULA MISMATCHES (train vs serve computation)")
    print("=" * 70)

    formulas = [
        # (feature, train_formula, serve_formula, match?)
        ("efg_diff",
         "train: (home_fgpct+0.2*home_3p) - (away_fgpct+0.2*away_3p), defaults 0.46/0.36",
         "serve: (home_fgpct+0.2*home_3p) - (away_fgpct+0.2*away_3p), defaults 0.46/0.365",
         False, "3P default: 0.36 vs 0.365"),

        ("elo_diff",
         "train: home_form - away_form (raw diff from parquet)",
         "serve: form diff with auto-detect raw Elo (>10) and normalize /200",
         False, "Training uses raw form values; serve normalizes if >10"),

        ("crowd_pct",
         "train: attendance / venue_capacity (real data, varies 0.80-1.05)",
         "serve: team-specific fill rate dict (fixed per team per season)",
         False, "Train uses actual attendance; serve uses seasonal avg proxy"),

        ("consistency_diff",
         "train: ceiling_diff - floor_diff",
         "serve: ceiling_diff - floor_diff",
         True, "Match ✅"),

        ("matchup_efg",
         "train: home_three_fg_rate - away_three_fg_rate",
         "serve: home_three_fg_rate - away_three_fg_rate (from row or enrichment)",
         True, "Match ✅ — but serve may get 0 if enrichment stale"),

        ("ou_gap",
         "train: home_ppg + away_ppg - market_total (ppg defaults 0)",
         "serve: home_ppg + away_ppg - market_total (ppg defaults 110)",
         False, "If ppg missing: train=0+0-mkt=-mkt, serve=110+110-mkt=220-mkt"),

        ("scoring_hhi_diff",
         "train: _safe_diff(df, 'home_scoring_hhi', 'away_scoring_hhi')",
         "serve: h_enr('scoring_hhi',0.15) - a_enr('scoring_hhi',0.15)",
         False, "Default: train=0, serve=0.15 (league avg proxy)"),

        ("lineup_value_diff",
         "train: _safe_diff(df, 'home_lineup_value', 'away_lineup_value')",
         "serve: h_enr('lineup_value',0) - a_enr('lineup_value',0) OR star_ppg*fg*2 override",
         False, "Serve has ESPN star player fallback that training doesn't"),

        ("ts_regression_diff",
         "train: home_ts_regression - away_ts_regression",
         "serve: (home_ts - league_avg) - (away_ts - league_avg), OR 0 if no TS data",
         False, "Train reads pre-computed regression; serve computes from TS%"),

        ("days_since_loss_diff",
         "train: home_days_since_loss - away_days_since_loss (from parquet)",
         "serve: derived from streak * 2.2 heuristic when stale",
         False, "2.2 multiplier is a guess, not empirically validated"),

        ("roll_dreb_diff",
         "train: home_roll_dreb - away_roll_dreb (from parquet, 96.8% coverage)",
         "serve: from get_rolling_diffs() → nba_team_rolling table (just added)",
         True, "Match ✅ after fix — but needs backfill to populate"),

        ("three_value_diff",
         "train: h_3fg_rate*(h_3fg_rate-0.20) - a_3fg_rate*(a_3fg_rate-0.20)",
         "serve: same formula but uses enrichment three_fg_rate or 0",
         True, "Match ✅ — but enrichment may be stale"),
    ]

    n_mismatch = 0
    for feat, train, serve, match, note in formulas:
        color = GRN if match else RED
        symbol = "✅" if match else "❌"
        print(f"\n  {color}{symbol} {feat}{RST}")
        print(f"    Train: {train}")
        print(f"    Serve: {serve}")
        if not match:
            print(f"    {YEL}⚠ {note}{RST}")
            n_mismatch += 1

    print(f"\n  Formula mismatches: {n_mismatch}/{len(formulas)}")
    return n_mismatch


def audit_feature_coverage():
    """Check training data coverage for all ATS + O/U features."""
    print("\n" + "=" * 70)
    print("  AUDIT 3: TRAINING DATA COVERAGE")
    print("=" * 70)

    try:
        from nba_build_features_v27 import load_training_data, build_features
    except ImportError:
        print("  Cannot load training data — run from sports-predictor-api directory")
        return

    df = load_training_data("nba_training_data.parquet")
    X, feat_names = build_features(df)

    # ATS V27 features
    V27 = [
        "lineup_value_diff", "win_pct_diff", "scoring_hhi_diff", "espn_pregame_wp",
        "ceiling_diff", "matchup_efg", "ml_implied_spread", "sharp_spread_signal",
        "efg_diff", "opp_suppression_diff", "net_rtg_diff", "steals_to_diff",
        "threepct_diff", "b2b_diff", "ftpct_diff", "ou_gap",
        "roll_dreb_diff", "ts_regression_diff", "roll_paint_pts_diff", "ref_home_whistle",
        "opp_ppg_diff", "roll_max_run_avg", "away_is_public_team", "away_after_loss",
        "games_last_14_diff", "h2h_total_games", "three_pt_regression_diff", "games_diff",
        "ref_foul_proxy", "roll_fast_break_diff", "crowd_pct", "matchup_to",
        "overround", "roll_ft_trip_rate_diff", "home_after_loss", "rest_diff",
        "spread_juice_imbalance", "vig_uncertainty",
    ]

    print(f"\n  ATS Model Features ({len(V27)}):")
    print(f"  {'Feature':40s} {'Non-zero%':>10s} {'Mean':>10s} {'Std':>10s} {'Status'}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for feat in sorted(V27):
        if feat in X.columns:
            col = X[feat]
            nz_pct = (col != 0).mean() * 100
            mean_val = col.mean()
            std_val = col.std()
            if nz_pct < 5:
                status = f"{RED}LOW{RST}"
            elif nz_pct < 50:
                status = f"{YEL}MEDIUM{RST}"
            else:
                status = f"{GRN}OK{RST}"
            print(f"  {feat:40s} {nz_pct:9.1f}% {mean_val:>10.4f} {std_val:>10.4f} {status}")
        else:
            print(f"  {feat:40s} {RED}MISSING FROM TRAINING{RST}")

    # Check raw parquet columns for serve-time data availability
    print(f"\n  RAW PARQUET: Key columns for serve-time features:")
    serve_critical = [
        "home_form", "away_form", "espn_pregame_wp",
        "home_ml_close", "away_ml_close", "spread_open", "spread_close",
        "ref_home_whistle", "ref_foul_proxy", "ref_ou_bias",
        "home_scoring_hhi", "away_scoring_hhi",
        "home_lineup_value", "away_lineup_value",
        "home_ceiling", "away_ceiling",
        "home_opp_suppression", "away_opp_suppression",
        "home_roll_dreb", "away_roll_dreb",
        "home_roll_paint_pts", "away_roll_paint_pts",
        "home_roll_fast_break_pts", "away_roll_fast_break_pts",
        "attendance", "venue_capacity",
        "home_three_fg_rate", "away_three_fg_rate",
        "home_ts_regression", "away_ts_regression",
        "home_after_loss", "away_after_loss",
    ]
    for col in serve_critical:
        if col in df.columns:
            nz = (pd.to_numeric(df[col], errors="coerce").fillna(0) != 0).sum()
            pct = nz / len(df) * 100
            status = f"{GRN}OK{RST}" if pct > 50 else (f"{YEL}SPARSE{RST}" if pct > 10 else f"{RED}EMPTY{RST}")
            print(f"    {col:40s} {pct:6.1f}% non-zero  {status}")
        else:
            print(f"    {col:40s} {RED}NOT IN PARQUET{RST}")


def audit_live_prediction(game_id="401810997"):
    """Test a live prediction and compare feature values to expectations."""
    print("\n" + "=" * 70)
    print(f"  AUDIT 4: LIVE PREDICTION TEST (game_id={game_id})")
    print("=" * 70)

    import requests
    try:
        r = requests.post(
            "https://sports-predictor-api-production.up.railway.app/predict/nba/full",
            json={"game_id": game_id}, timeout=30
        )
        if not r.ok:
            print(f"  {RED}API error: {r.status_code}{RST}")
            return
        result = r.json()
    except Exception as e:
        print(f"  {RED}API call failed: {e}{RST}")
        return

    print(f"\n  Game: {result.get('away_team')} @ {result.get('home_team')}")
    print(f"  Coverage: {result.get('feature_coverage')}")
    print(f"  Model: {result.get('model_meta', {}).get('model_type')}")
    print(f"  ATS margin: {result.get('ml_margin')}")
    print(f"  O/U pick: {result.get('ou_pick')} {result.get('ou_tier')}u")
    print(f"  O/U residual: {result.get('ou_res_avg')}")
    print(f"  O/U P(under): {result.get('ou_cls_avg')}")

    # Check for zero-value features that shouldn't be zero
    shap = result.get("shap", [])
    zero_features = [s for s in shap if s["value"] == 0]
    nonzero_features = [s for s in shap if s["value"] != 0]

    print(f"\n  Non-zero features: {len(nonzero_features)}/{len(shap)}")

    # Categorize zero features
    always_zero_ok = {"b2b_diff", "home_after_loss", "away_after_loss", "reverse_line_movement",
                      "games_diff", "is_early_season", "altitude_factor"}
    market_zero_ok = {"sharp_spread_signal", "spread_juice_imbalance", "vig_uncertainty",
                      "overround", "ml_implied_spread"}
    ref_zero_ok = {"ref_home_whistle", "ref_foul_proxy", "ref_ou_bias", "ref_pace_impact"}

    unexpected_zeros = []
    for s in zero_features:
        f = s["feature"]
        if f in always_zero_ok:
            continue  # legitimately zero sometimes
        elif f in market_zero_ok:
            if not result.get("market_spread"):
                continue  # no market data yet
        elif f in ref_zero_ok:
            continue  # refs not assigned
        else:
            unexpected_zeros.append(f)

    if unexpected_zeros:
        print(f"\n  {RED}UNEXPECTED ZEROS ({len(unexpected_zeros)}):{RST}")
        for f in unexpected_zeros:
            print(f"    {RED}❌ {f} = 0 (should have data){RST}")
    else:
        print(f"\n  {GRN}✅ No unexpected zero features{RST}")

    # Check diagnostics
    warnings_list = result.get("diagnostics", {}).get("warnings", [])
    if warnings_list:
        print(f"\n  Warnings ({len(warnings_list)}):")
        for w in warnings_list:
            print(f"    ⚠ {w}")

    sources = result.get("diagnostics", {}).get("sources", [])
    print(f"\n  Data sources ({len(sources)}):")
    for s in sources:
        print(f"    ✓ {s}")


def audit_ou_combined_features():
    """Check O/U combined features match between train and serve."""
    print("\n" + "=" * 70)
    print("  AUDIT 5: O/U COMBINED FEATURES (train vs serve)")
    print("=" * 70)

    # O/U combined features from nba_retrain_ou_v2.py build_ou_combined()
    # vs nba_full_predict.py inline computation
    ou_features = [
        ("market_total", "train: cascade market_ou_total→ou_total→dk_ou",
         "serve: row.get('market_ou_total') with 228 fallback",
         "228 fallback is a guess"),

        ("ppg_combined", "train: home_ppg(default 112) + away_ppg(default 112)",
         "serve: row home_ppg(default 112 from setdefault) + away_ppg",
         "Match after setdefault — but training default is 0 not 112"),

        ("pace_min", "train: min(home_tempo(100), away_tempo(100))",
         "serve: min(home_tempo(100), away_tempo(100))",
         "Match ✅"),

        ("ref_ou_bias", "train: from ref_ou_bias column in parquet",
         "serve: from ref_profile.get('ou_bias') — scraped or Supabase",
         "Depends on ref assignment timing"),

        ("ceiling_combined", "train: home_ceiling + away_ceiling (from parquet)",
         "serve: enrichment home ceiling + away ceiling",
         "Match if enrichment populated"),

        ("oreb_combined", "train: home_orb_pct + away_orb_pct (default 0.25)",
         "serve: same from row (default 0.25)",
         "Match ✅"),

        ("blowout_risk", "train: abs(home_wp - away_wp)",
         "serve: same formula from row wins/losses",
         "Match ✅"),
    ]

    for feat, train, serve, note in ou_features:
        has_issue = "mismatch" in note.lower() or "guess" in note.lower() or "depends" in note.lower()
        color = RED if has_issue else GRN
        symbol = "❌" if "guess" in note.lower() else ("⚠" if has_issue else "✅")
        print(f"\n  {color}{symbol} {feat}{RST}")
        print(f"    Train: {train}")
        print(f"    Serve: {serve}")
        print(f"    Note:  {note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-test", action="store_true", help="Include live API test")
    parser.add_argument("--coverage", action="store_true", help="Include data coverage check")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    if args.all:
        args.live_test = True
        args.coverage = True

    n1 = audit_defaults()
    n2 = audit_feature_formulas()
    audit_ou_combined_features()

    if args.coverage:
        audit_feature_coverage()

    if args.live_test:
        audit_live_prediction()

    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Default mismatches:  {n1}")
    print(f"  Formula mismatches:  {n2}")
    print(f"\n  Priority fixes:")
    print(f"    1. Align ppg default: train=0, serve=110/112 → pick one")
    print(f"    2. Align opp_ppg default: train=0, serve=112")
    print(f"    3. Align fgpct default: 0.46 vs 0.471")
    print(f"    4. Align 3pt default: 0.36 vs 0.365")
    print(f"    5. Align scoring_hhi default: 0 vs 0.15")
    print(f"    6. Remove market_ou_total=228 guess → use 0 (same as training)")
    print(f"    7. Validate days_since_loss 2.2× multiplier")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
