#!/usr/bin/env python3
"""
nba_v27_verify.py — Verify train/serve feature consistency.

Loads nba_training_data.parquet, picks N sample games, computes features via:
  1. Training path: nba_build_features_v27.build_features()
  2. Live path:     nba_v27_features_live.build_v27_features()

Compares the 38 v27 features side-by-side and flags any divergence > tolerance.

Usage:
    python3 nba_v27_verify.py                  # 20 sample games
    python3 nba_v27_verify.py --n 100          # 100 sample games
    python3 nba_v27_verify.py --game-idx 5000  # specific game index
    python3 nba_v27_verify.py --all            # all games (slow)
"""
import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from nba_build_features_v27 import load_training_data, build_features
from nba_v27_features_live import build_v27_features
from nba_v27_train import V27_FEATURE_SET

TOLERANCE_ABS = 0.01    # absolute tolerance per feature
TOLERANCE_REL = 0.05    # relative tolerance (5%)


def map_parquet_row_to_game_dict(row, df_row):
    """
    Map a single row from nba_training_data.parquet into the game dict
    format that nba_v27_features_live.build_v27_features() expects.

    This is the CRITICAL mapping — if anything here doesn't match how
    nba_full_predict.py assembles the game dict, features will diverge.
    """
    def s(key, default=0):
        val = df_row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except:
            return default

    game = {
        # ── Basic stats ──
        "home_ppg": s("home_ppg", 110),
        "away_ppg": s("away_ppg", 110),
        "home_opp_ppg": s("home_opp_ppg", 110),
        "away_opp_ppg": s("away_opp_ppg", 110),
        "home_fgpct": s("home_fgpct", 0.46),
        "away_fgpct": s("away_fgpct", 0.46),
        "home_threepct": s("home_threepct", 0),
        "away_threepct": s("away_threepct", 0),
        "home_ftpct": s("home_ftpct", 0),
        "away_ftpct": s("away_ftpct", 0),
        "home_steals": s("home_steals", 7.5),
        "away_steals": s("away_steals", 7.5),
        "home_turnovers": s("home_turnovers", 14),
        "away_turnovers": s("away_turnovers", 14),

        # ── Records ──
        "home_wins": s("home_wins", 20),
        "home_losses": s("home_losses", 20),
        "away_wins": s("away_wins", 20),
        "away_losses": s("away_losses", 20),

        # ── Net rating ──
        "home_net_rtg": s("home_net_rtg", 0),
        "away_net_rtg": s("away_net_rtg", 0),

        # ── Market ──
        "market_spread_home": s("market_spread_home", 0),
        "market_ou_total": s("market_ou_total", 0) or s("dk_ou", 0) or s("espn_over_under", 0) or s("ou_total", 0),

        # ── Rest ──
        "home_days_rest": s("home_days_rest", 2),
        "away_days_rest": s("away_days_rest", 2),

        # ── Opponent FG% ──
        "home_opp_fgpct": s("home_opp_fgpct", 0.46),
        "away_opp_fgpct": s("away_opp_fgpct", 0.46),

        # ── ESPN pregame WP ──
        "espn_pregame_wp": s("espn_pregame_wp", 0.5),

        # ── Market signals ──
        "sharp_spread_signal": s("sharp_spread_signal", 0),
        "spread_juice_imbalance": s("spread_juice_imbalance", 0),
        "vig_uncertainty": s("vig_uncertainty", 0),
        "home_moneyline": s("home_moneyline", 0),
        "away_moneyline": s("away_moneyline", 0),

        # ── Crowd ──
        "crowd_pct": s("crowd_pct", 0.9),

        # ── After loss ──
        "home_last_result": s("home_last_result", 0),
        "away_last_result": s("away_last_result", 0),

        # ── Schedule density ──
        "home_games_last_14": s("home_games_last_14", 6),
        "away_games_last_14": s("away_games_last_14", 6),

        # ── H2H ──
        "h2h_total_games": s("h2h_total_games", 0),

        # ── Games diff (derived from wins+losses) ──
        # (handled in live builder from wins/losses)

        # ── Rolling stats ──
        "home_roll_dreb": s("home_roll_dreb", 0),
        "away_roll_dreb": s("away_roll_dreb", 0),
        "home_opp_suppression": s("home_opp_suppression", 0),
        "away_opp_suppression": s("away_opp_suppression", 0),
        "home_moneyline": s("dk_home_ml", 0) or s("home_moneyline", 0),
        "away_moneyline": s("dk_away_ml", 0) or s("away_moneyline", 0),
        "home_spread_odds": s("dk_home_spread_odds", 0) or s("mgm_home_spread_odds", 0),
        "away_spread_odds": s("dk_away_spread_odds", 0) or s("mgm_away_spread_odds", 0),
        "spread_open": s("spread_open", 0) or s("dk_spread_open", 0) or s("odds_api_spread_open", 0),
        "spread_close": s("spread_close", 0) or s("dk_spread_close", 0) or s("odds_api_spread_close", 0),
        "market_spread_home": s("market_spread_home", 0),  # original column, not backfilled market_spread
        "implied_prob_home": s("implied_prob_home", 0),
        "home_three_fg_rate": s("home_three_fg_rate", 0),
        "away_three_fg_rate": s("away_three_fg_rate", 0),
        "home_three_pt_regression": s("home_three_pt_regression", 0),
        "away_three_pt_regression": s("away_three_pt_regression", 0),
        "away_is_public_team": s("away_is_public_team", 0),
        "attendance": s("attendance", 0),
        "venue_capacity": s("venue_capacity", 19000),
        "home_after_loss": s("home_after_loss", 0),
        "away_after_loss": s("away_after_loss", 0),
        "home_roll_paint_pts": s("home_roll_paint_pts", 0),
        "away_roll_paint_pts": s("away_roll_paint_pts", 0),
        "home_roll_max_run": s("home_roll_max_run", 0),
        "away_roll_max_run": s("away_roll_max_run", 0),
        "home_roll_fast_break_pts": s("home_roll_fast_break_pts", 0),
        "away_roll_fast_break_pts": s("away_roll_fast_break_pts", 0),
        "home_roll_ft_trip_rate": s("home_roll_ft_trip_rate", 0),
        "away_roll_ft_trip_rate": s("away_roll_ft_trip_rate", 0),

        # ── Team abbr (for public team detection) ──
        "away_team_abbr": df_row.get("away_team_abbr", "")
                          or df_row.get("away_team", "")
                          or df_row.get("away_team_name", ""),
    }

    # ── Enrichment data ──
    # In training, these come from parquet columns.
    # In live, they come from nba_team_enrichment table.
    enrichment = {
        "home": {
            "lineup_value": s("home_lineup_value", 0),
            "scoring_hhi": s("home_scoring_hhi", 0.15),
            "ceiling": s("home_ceiling", 0),
            "efg_pct": s("home_efg_pct", 0) or s("home_efg", 0),
            "opp_efg_pct": s("home_opp_efg_pct", 0),
            "ts_pct": s("home_ts_pct", 0.56),
        },
        "away": {
            "lineup_value": s("away_lineup_value", 0),
            "scoring_hhi": s("away_scoring_hhi", 0.15),
            "ceiling": s("away_ceiling", 0),
            "efg_pct": s("away_efg_pct", 0) or s("away_efg", 0),
            "opp_efg_pct": s("away_opp_efg_pct", 0),
            "ts_pct": s("away_ts_pct", 0.56),
        },
    }

    # ── Referee profile ──
    ref_profile = {
        "home_whistle": s("ref_home_whistle", 0),
        "foul_rate": s("ref_foul_rate", 0) or s("ref_foul_proxy", 0),
    }

    return game, enrichment, ref_profile


def compare_features(train_vals, live_vals, feature_names, game_idx, verbose=True):
    """
    Compare training vs live feature values.
    Returns list of (feature, train_val, live_val, abs_diff, status) tuples.
    """
    results = []
    for feat in feature_names:
        tv = float(train_vals.get(feat, 0))
        lv = float(live_vals.get(feat, 0))
        abs_diff = abs(tv - lv)

        # Determine tolerance
        denom = max(abs(tv), abs(lv), 1e-6)
        rel_diff = abs_diff / denom

        if abs_diff < TOLERANCE_ABS:
            status = "OK"
        elif rel_diff < TOLERANCE_REL:
            status = "OK (within rel)"
        elif abs_diff < 0.1:
            status = "WARN"
        else:
            status = "FAIL"

        results.append((feat, tv, lv, abs_diff, rel_diff, status))

    return results


def run_verification(n_samples=20, game_idx=None, run_all=False):
    """Main verification loop."""
    print("=" * 80)
    print("  NBA v27 TRAIN/SERVE VERIFICATION")
    print("=" * 80)

    # ── Load data ──
    df = load_training_data("nba_training_data.parquet")

    # ── Build ALL training features (this is the ground truth) ──
    print("\nBuilding training features (all 102)...")
    X_train_full, all_features = build_features(df)
    print(f"  Built {len(all_features)} features for {len(df)} games")

    # Extract just the 38 v27 features from training
    v27_features = [f for f in V27_FEATURE_SET if f in all_features]
    missing_in_train = [f for f in V27_FEATURE_SET if f not in all_features]
    if missing_in_train:
        print(f"\n  WARNING: {len(missing_in_train)} v27 features NOT in training builder:")
        for f in missing_in_train:
            print(f"    - {f}")

    X_train_v27 = X_train_full[v27_features]
    print(f"  v27 features available in training: {len(v27_features)}")

    # ── Discover which parquet columns the live builder needs ──
    raw_cols = list(df.columns)
    print(f"\n  Raw parquet columns: {len(raw_cols)}")

    # ── Check which raw columns exist for the mapping ──
    mapping_cols = {
        "enrichment": ["home_lineup_value", "away_lineup_value",
                       "home_scoring_hhi", "away_scoring_hhi",
                       "home_ceiling", "away_ceiling",
                       "home_efg_pct", "away_efg_pct",
                       "home_opp_efg_pct", "away_opp_efg_pct",
                       "home_ts_pct", "away_ts_pct"],
        "rolling": ["home_roll_dreb", "away_roll_dreb",
                     "home_roll_paint_pts", "away_roll_paint_pts",
                     "home_roll_max_run", "away_roll_max_run",
                     "home_roll_fast_break", "away_roll_fast_break",
                     "home_roll_ft_trip_rate", "away_roll_ft_trip_rate"],
        "referee": ["ref_home_whistle", "ref_foul_rate", "ref_foul_proxy"],
        "market": ["sharp_spread_signal", "spread_juice_imbalance",
                    "vig_uncertainty", "home_moneyline", "away_moneyline"],
        "context": ["espn_pregame_wp", "crowd_pct",
                     "home_last_result", "away_last_result",
                     "home_games_last_14", "away_games_last_14",
                     "h2h_total_games"],
    }

    print(f"\n  Column availability check:")
    for category, cols in mapping_cols.items():
        present = [c for c in cols if c in raw_cols]
        missing = [c for c in cols if c not in raw_cols]
        status = "✓" if not missing else "⚠"
        print(f"    {status} {category}: {len(present)}/{len(cols)} present")
        if missing:
            for m in missing:
                # Try fuzzy match
                close = [c for c in raw_cols if m.replace("home_", "").replace("away_", "") in c]
                hint = f" → maybe: {close[:3]}" if close else ""
                print(f"        MISSING: {m}{hint}")

    # ── Select sample games ──
    if run_all:
        indices = range(len(df))
        print(f"\n  Running on ALL {len(df)} games...")
    elif game_idx is not None:
        indices = [game_idx]
        print(f"\n  Running on game index {game_idx}")
    else:
        np.random.seed(42)
        indices = sorted(np.random.choice(len(df), min(n_samples, len(df)), replace=False))
        print(f"\n  Running on {len(indices)} sample games")

    # ── Compare features game by game ──
    n_ok = 0
    n_warn = 0
    n_fail = 0
    feature_fail_counts = {f: 0 for f in v27_features}
    feature_warn_counts = {f: 0 for f in v27_features}
    worst_diffs = {f: 0 for f in v27_features}
    example_failures = []

    for idx in indices:
        # Training path: already computed
        train_vals = {f: float(X_train_v27.iloc[idx][f]) for f in v27_features}

        # Live path: map parquet row → game dict → live builder
        df_row = df.iloc[idx].to_dict()
        game_dict, enrichment, ref_profile = map_parquet_row_to_game_dict(None, df_row)
        live_df = build_v27_features(game_dict, enrichment=enrichment, ref_profile=ref_profile)
        live_vals = {f: float(live_df[f].iloc[0]) if f in live_df.columns else 0
                     for f in v27_features}

        # Compare
        results = compare_features(train_vals, live_vals, v27_features, idx, verbose=False)

        game_has_fail = False
        for feat, tv, lv, abs_diff, rel_diff, status in results:
            if status == "FAIL":
                n_fail += 1
                feature_fail_counts[feat] += 1
                game_has_fail = True
            elif status == "WARN":
                n_warn += 1
                feature_warn_counts[feat] += 1
            else:
                n_ok += 1
            worst_diffs[feat] = max(worst_diffs[feat], abs_diff)

        # Store a detailed example for the first few failures
        if game_has_fail and len(example_failures) < 5:
            fails = [(feat, tv, lv, abs_diff) for feat, tv, lv, abs_diff, _, status in results
                     if status == "FAIL"]
            example_failures.append((idx, df_row.get("game_id", "?"), fails))

    total = n_ok + n_warn + n_fail
    n_games = len(indices)

    # ── Report ──
    print(f"\n{'='*80}")
    print(f"  RESULTS — {n_games} games × {len(v27_features)} features = {total} comparisons")
    print(f"{'='*80}")
    print(f"  ✓ OK:   {n_ok} ({n_ok/total*100:.1f}%)")
    print(f"  ⚠ WARN: {n_warn} ({n_warn/total*100:.1f}%)")
    print(f"  ✗ FAIL: {n_fail} ({n_fail/total*100:.1f}%)")

    # Features with failures
    failing_features = [(f, c) for f, c in feature_fail_counts.items() if c > 0]
    if failing_features:
        failing_features.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  FAILING FEATURES ({len(failing_features)}):")
        print(f"  {'Feature':<35} | {'Fail#':>6} | {'Fail%':>7} | {'MaxDiff':>10}")
        print(f"  {'-'*70}")
        for feat, count in failing_features:
            pct = count / n_games * 100
            maxd = worst_diffs[feat]
            print(f"  {feat:<35} | {count:>6} | {pct:>6.1f}% | {maxd:>10.4f}")

    # Warning features
    warning_features = [(f, c) for f, c in feature_warn_counts.items() if c > 0 and feature_fail_counts[f] == 0]
    if warning_features:
        warning_features.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  WARN-ONLY FEATURES ({len(warning_features)}):")
        for feat, count in warning_features[:10]:
            pct = count / n_games * 100
            print(f"    {feat:<35} {count} games ({pct:.1f}%), max_diff={worst_diffs[feat]:.4f}")

    # Example failures
    if example_failures:
        print(f"\n  EXAMPLE FAILURES (first {len(example_failures)} games):")
        for idx, game_id, fails in example_failures:
            print(f"\n    Game idx={idx}, id={game_id}:")
            for feat, tv, lv, diff in fails:
                print(f"      {feat:<35} train={tv:>+10.4f}  live={lv:>+10.4f}  diff={diff:>8.4f}")

    # Perfect features
    perfect = [f for f, c in feature_fail_counts.items()
               if c == 0 and feature_warn_counts[f] == 0]
    if perfect:
        print(f"\n  PERFECT FEATURES ({len(perfect)}/{len(v27_features)}):")
        for f in perfect:
            print(f"    ✓ {f}")

    # ── Action items ──
    if failing_features:
        print(f"\n{'='*80}")
        print(f"  ACTION ITEMS")
        print(f"{'='*80}")
        for feat, count in failing_features:
            pct = count / n_games * 100
            if pct > 50:
                priority = "CRITICAL"
            elif pct > 10:
                priority = "HIGH"
            else:
                priority = "MEDIUM"

            print(f"\n  [{priority}] {feat} (fails {pct:.0f}% of games)")
            print(f"    → Check how nba_build_features_v27.py computes this")
            print(f"    → Compare with nba_v27_features_live.py computation")
            print(f"    → Align the formulas or mapping")
    else:
        print(f"\n  ✓ ALL FEATURES MATCH — safe to deploy!")

    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA v27 Train/Serve Verification")
    parser.add_argument("--n", type=int, default=20, help="Number of sample games")
    parser.add_argument("--game-idx", type=int, default=None, help="Specific game index")
    parser.add_argument("--all", action="store_true", help="All games (slow)")
    args = parser.parse_args()

    ok = run_verification(n_samples=args.n, game_idx=args.game_idx, run_all=args.all)
    sys.exit(0 if ok else 1)
