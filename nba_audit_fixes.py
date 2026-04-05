#!/usr/bin/env python3
"""
nba_audit_fixes.py — Validation + calibration for NBA audit findings
====================================================================
1. Verify CRIT-1 fix (home_after_loss, away_after_loss now non-zero)
2. Verify CRIT-2 fix (crowd_pct varies by team, not constant)
3. Sigma calibration sweep (HIGH-1)
4. Train/serve parity spot check

Usage:
    python3 nba_audit_fixes.py --verify     # Check CRIT fixes
    python3 nba_audit_fixes.py --sigma      # Brier sweep for optimal sigma
    python3 nba_audit_fixes.py --parity     # Train/serve parity check
    python3 nba_audit_fixes.py --all        # Everything
"""
import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def verify_crit_fixes():
    """Verify CRIT-1 and CRIT-2 fixes in the live builder."""
    print("=" * 70)
    print("  VERIFY CRIT FIXES")
    print("=" * 70)

    from nba_v27_features_live import build_v27_features, FEATURES_69

    # ── CRIT-1: home_after_loss / away_after_loss ──
    # Test: home team on losing streak → home_after_loss should be 1
    game_loss = {
        "home_team": "CHI", "away_team": "BOS", "game_date": "2026-04-01",
        "home_ppg": 108, "away_ppg": 118, "home_wins": 20, "home_losses": 50,
        "away_wins": 55, "away_losses": 15,
        "home_streak": -3, "away_streak": 5,  # home on 3-game losing streak
        "market_spread_home": 8.5, "home_days_rest": 1, "away_days_rest": 2,
    }
    feat_df = build_v27_features(game_loss)

    hal = float(feat_df["home_after_loss"].iloc[0]) if "home_after_loss" in feat_df.columns else -1
    aal = float(feat_df["away_after_loss"].iloc[0]) if "away_after_loss" in feat_df.columns else -1
    ale = float(feat_df["after_loss_either"].iloc[0]) if "after_loss_either" in feat_df.columns else -1

    print(f"\n  CRIT-1: home_after_loss / away_after_loss")
    print(f"    home_after_loss:  {hal}  (expected: 1)  {'✅' if hal == 1 else '❌'}")
    print(f"    away_after_loss:  {aal}  (expected: 0)  {'✅' if aal == 0 else '❌'}")
    print(f"    after_loss_either: {ale} (expected: 1)  {'✅' if ale == 1 else '❌'}")

    # Test: neither team after loss
    game_win = {
        "home_team": "BOS", "away_team": "NYK", "game_date": "2026-04-01",
        "home_ppg": 118, "away_ppg": 112, "home_wins": 55, "home_losses": 15,
        "away_wins": 45, "away_losses": 25,
        "home_streak": 5, "away_streak": 3,  # both on winning streaks
        "market_spread_home": -6.5, "home_days_rest": 2, "away_days_rest": 2,
    }
    feat_df2 = build_v27_features(game_win)
    hal2 = float(feat_df2["home_after_loss"].iloc[0])
    aal2 = float(feat_df2["away_after_loss"].iloc[0])
    ale2 = float(feat_df2["after_loss_either"].iloc[0])
    print(f"\n    Control (both winning):")
    print(f"    home_after_loss:  {hal2}  (expected: 0)  {'✅' if hal2 == 0 else '❌'}")
    print(f"    away_after_loss:  {aal2}  (expected: 0)  {'✅' if aal2 == 0 else '❌'}")
    print(f"    after_loss_either: {ale2} (expected: 0)  {'✅' if ale2 == 0 else '❌'}")

    # ── CRIT-2: crowd_pct varies by team ──
    print(f"\n  CRIT-2: crowd_pct varies by team")
    teams = ["BOS", "DET", "WAS", "LAL", "CHI"]
    vals = []
    for t in teams:
        g = {"home_team": t, "away_team": "MIA", "game_date": "2026-04-01",
             "home_ppg": 110, "away_ppg": 110, "home_wins": 30, "home_losses": 30,
             "away_wins": 30, "away_losses": 30, "market_spread_home": 0}
        df = build_v27_features(g)
        cp = float(df["crowd_pct"].iloc[0]) if "crowd_pct" in df.columns else -1
        vals.append(cp)
        print(f"    {t}: crowd_pct = {cp:.3f}")
    unique = len(set(vals))
    print(f"    Unique values: {unique}  {'✅ varies' if unique > 1 else '❌ constant'}")

    # ── Verify all V27 features are in FEATURES_69 ──
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
    missing = [f for f in V27 if f not in FEATURES_69]
    print(f"\n  V27 features missing from FEATURES_69: {len(missing)}")
    for f in missing:
        print(f"    ❌ {f}")
    if not missing:
        print(f"    ✅ All V27 features in FEATURES_69")

    # Check they're in the output DataFrame
    feat_cols_in_df = [f for f in V27 if f in feat_df.columns]
    feat_missing_df = [f for f in V27 if f not in feat_df.columns]
    print(f"\n  V27 features in output DataFrame: {len(feat_cols_in_df)}/{len(V27)}")
    for f in feat_missing_df:
        print(f"    ❌ {f} NOT in DataFrame")

    crit1_pass = (hal == 1 and aal == 0 and hal2 == 0 and aal2 == 0)
    crit2_pass = (unique > 1)
    print(f"\n  CRIT-1: {'✅ PASS' if crit1_pass else '❌ FAIL'}")
    print(f"  CRIT-2: {'✅ PASS' if crit2_pass else '❌ FAIL'}")
    return crit1_pass and crit2_pass


def sigma_sweep():
    """Walk-forward Brier sweep to find optimal sigma for NBA win probability."""
    print("=" * 70)
    print("  SIGMA CALIBRATION SWEEP (NBA)")
    print("=" * 70)

    from nba_build_features_v27 import load_training_data, build_features

    df = load_training_data("nba_training_data.parquet")
    X, feature_names = build_features(df)
    y = df["target_margin"].values

    # Sort by date
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort().values
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    df = df.iloc[sort_idx].reset_index(drop=True)
    spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values

    # Walk-forward OOF predictions
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso

    n_folds = 20
    fold_size = len(X) // (n_folds + 2)
    min_train = fold_size * 2
    oof_preds = np.full(len(X), np.nan)

    print(f"\n  Running {n_folds}-fold walk-forward (Lasso α=0.1)...")
    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, len(X))
        if ts >= len(X): break
        sc = StandardScaler()
        X_tr = sc.fit_transform(X.iloc[:ts])
        X_te = sc.transform(X.iloc[ts:te])
        model = Lasso(alpha=0.1, max_iter=5000)
        model.fit(X_tr, y[:ts])
        oof_preds[ts:te] = model.predict(X_te)

    valid = ~np.isnan(oof_preds)
    print(f"  OOF predictions: {valid.sum()} games")
    print(f"  MAE: {np.mean(np.abs(oof_preds[valid] - y[valid])):.3f}")

    # Sweep sigma
    sigmas = np.arange(4.0, 16.0, 0.5)
    actual_home_win = (y > 0).astype(float)

    print(f"\n  {'σ':>6}  {'Brier':>8}  {'Acc':>7}  {'Cal Gap':>9}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*9}")

    best_sigma = 8.0
    best_brier = 1.0

    for sigma in sigmas:
        probs = 1.0 / (1.0 + np.exp(-oof_preds[valid] / sigma))
        brier = float(np.mean((probs - actual_home_win[valid]) ** 2))
        acc = float(np.mean((probs > 0.5) == actual_home_win[valid].astype(bool)))

        # Calibration: avg predicted prob vs actual win rate
        cal_gap = abs(probs.mean() - actual_home_win[valid].mean())

        marker = " ◀" if brier < best_brier else ""
        print(f"  {sigma:6.1f}  {brier:8.4f}  {acc:6.1%}  {cal_gap:9.4f}{marker}")

        if brier < best_brier:
            best_brier = brier
            best_sigma = sigma

    print(f"\n  ══════════════════════════════════")
    print(f"  OPTIMAL σ = {best_sigma:.1f}  (Brier = {best_brier:.4f})")
    print(f"  Current σ = 8.0     (compare above)")
    print(f"  ══════════════════════════════════")

    # Also check base-10 vs base-e at optimal sigma
    probs_e = 1.0 / (1.0 + np.exp(-oof_preds[valid] / best_sigma))
    probs_10 = 1.0 / (1.0 + 10 ** (-oof_preds[valid] / best_sigma))
    brier_e = float(np.mean((probs_e - actual_home_win[valid]) ** 2))
    brier_10 = float(np.mean((probs_10 - actual_home_win[valid]) ** 2))
    print(f"\n  Base-e  Brier @ σ={best_sigma}: {brier_e:.4f}")
    print(f"  Base-10 Brier @ σ={best_sigma}: {brier_10:.4f}")
    print(f"  Δ: {abs(brier_e - brier_10):.4f} ({'base-e wins' if brier_e < brier_10 else 'base-10 wins'})")

    return best_sigma


def parity_check():
    """Spot-check train/serve feature parity on sample games."""
    print("=" * 70)
    print("  TRAIN/SERVE PARITY CHECK")
    print("=" * 70)

    from nba_build_features_v27 import load_training_data, build_features
    from nba_v27_features_live import build_v27_features

    df = load_training_data("nba_training_data.parquet")
    X_train, feature_names = build_features(df)

    # V27 feature set
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
    available_v27 = [f for f in V27 if f in X_train.columns]
    print(f"\n  V27 features in training: {len(available_v27)}/{len(V27)}")

    import random
    sample_indices = random.sample(range(len(df)), min(20, len(df)))

    total_fail = 0
    total_warn = 0
    feat_issues = {}

    for idx in sample_indices:
        row = df.iloc[idx]

        # Build training features
        train_vals = {f: float(X_train[f].iloc[idx]) for f in available_v27}

        # Map parquet row to game dict for live builder
        game = {
            "home_team": str(row.get("home_team", "")),
            "away_team": str(row.get("away_team", "")),
            "game_date": str(row.get("game_date", "")),
            "home_ppg": float(row.get("home_ppg", 110) or 110),
            "away_ppg": float(row.get("away_ppg", 110) or 110),
            "home_opp_ppg": float(row.get("home_opp_ppg", 110) or 110),
            "away_opp_ppg": float(row.get("away_opp_ppg", 110) or 110),
            "home_fgpct": float(row.get("home_fgpct", 0.46) or 0.46),
            "away_fgpct": float(row.get("away_fgpct", 0.46) or 0.46),
            "home_threepct": float(row.get("home_threepct", 0.36) or 0.36),
            "away_threepct": float(row.get("away_threepct", 0.36) or 0.36),
            "home_ftpct": float(row.get("home_ftpct", 0.77) or 0.77),
            "away_ftpct": float(row.get("away_ftpct", 0.77) or 0.77),
            "home_steals": float(row.get("home_steals", 7.5) or 7.5),
            "away_steals": float(row.get("away_steals", 7.5) or 7.5),
            "home_turnovers": float(row.get("home_turnovers", 14) or 14),
            "away_turnovers": float(row.get("away_turnovers", 14) or 14),
            "home_wins": int(row.get("home_wins", 20) or 20),
            "home_losses": int(row.get("home_losses", 20) or 20),
            "away_wins": int(row.get("away_wins", 20) or 20),
            "away_losses": int(row.get("away_losses", 20) or 20),
            "home_days_rest": float(row.get("home_days_rest", 2) or 2),
            "away_days_rest": float(row.get("away_days_rest", 2) or 2),
            "market_spread_home": float(row.get("market_spread_home", 0) or 0),
            "market_ou_total": float(row.get("market_ou_total", 0) or 0),
            "espn_pregame_wp": float(row.get("espn_pregame_wp", 0.5) or 0.5),
            "home_net_rtg": float(row.get("home_net_rtg", 0) or 0),
            "away_net_rtg": float(row.get("away_net_rtg", 0) or 0),
            "home_ato_ratio": float(row.get("home_ato_ratio", 1.8) or 1.8),
            "away_ato_ratio": float(row.get("away_ato_ratio", 1.8) or 1.8),
            "home_form": float(row.get("home_form", 0) or 0),
            "away_form": float(row.get("away_form", 0) or 0),
            "home_streak": int(row.get("home_streak", 0) or 0),
            "away_streak": int(row.get("away_streak", 0) or 0),
            "home_after_loss": int(row.get("home_after_loss", 0) or 0),
            "away_after_loss": int(row.get("away_after_loss", 0) or 0),
            "home_games_last_14": float(row.get("home_games_last_14", 0) or 0),
            "away_games_last_14": float(row.get("away_games_last_14", 0) or 0),
            "ref_home_whistle": float(row.get("ref_home_whistle", 0) or 0),
            "ref_foul_proxy": float(row.get("ref_foul_proxy", 0) or 0),
            "home_moneyline": float(row.get("home_ml_close", 0) or 0),
            "away_moneyline": float(row.get("away_ml_close", 0) or 0),
            "home_spread_odds": float(row.get("dk_home_spread_odds", -110) or -110),
            "away_spread_odds": float(row.get("dk_away_spread_odds", -110) or -110),
            "h2h_total_games": int(row.get("h2h_total_games", 0) or 0),
        }

        # Build enrichment dict from parquet columns
        enrichment = {"home": {}, "away": {}}
        for side in ["home", "away"]:
            for col in ["ceiling", "floor", "scoring_hhi", "scoring_entropy",
                        "lineup_value", "scoring_var", "bimodal", "score_kurtosis",
                        "margin_accel", "momentum_halflife", "win_aging",
                        "pyth_residual", "pyth_luck", "recovery",
                        "opp_suppression", "ts_pct", "three_fg_rate",
                        "ft_trip_rate", "three_pt_regression", "ts_regression"]:
                key = f"{side}_{col}"
                if key in row.index:
                    val = row[key]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        enrichment[side][col] = float(val)

        feat_df = build_v27_features(game, enrichment=enrichment)
        live_vals = {f: float(feat_df[f].iloc[0]) if f in feat_df.columns else 0
                     for f in available_v27}

        # Compare
        mismatches = []
        for feat in available_v27:
            tv = train_vals.get(feat, 0)
            lv = live_vals.get(feat, 0)
            diff = abs(tv - lv)
            denom = max(abs(tv), abs(lv), 1e-6)

            if diff < 0.05:
                continue
            elif diff / denom < 0.10:
                continue
            elif diff < 0.5:
                mismatches.append((feat, tv, lv, diff, "WARN"))
                total_warn += 1
            else:
                mismatches.append((feat, tv, lv, diff, "FAIL"))
                total_fail += 1

            feat_issues[feat] = feat_issues.get(feat, 0) + 1

        teams = f"{game['away_team']}@{game['home_team']}"
        tag = "✅ PASS" if not mismatches else f"❌ {len(mismatches)} issues"
        print(f"  [{idx:5d}] {teams:>10s} {game['game_date']}  {tag}")
        for feat, tv, lv, diff, status in mismatches[:5]:
            print(f"          {feat:35s}  train={tv:+8.3f}  live={lv:+8.3f}  Δ={diff:.3f}")

    print(f"\n  {'='*60}")
    print(f"  TOTAL: {total_fail} FAILs, {total_warn} WARNs across {len(sample_indices)} games")
    if feat_issues:
        print(f"\n  Most frequent mismatches:")
        for f, c in sorted(feat_issues.items(), key=lambda x: -x[1])[:10]:
            print(f"    {f:35s}  {c} games")
    print(f"  {'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify CRIT fixes")
    parser.add_argument("--sigma", action="store_true", help="Sigma calibration sweep")
    parser.add_argument("--parity", action="store_true", help="Train/serve parity check")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    if args.all:
        args.verify = args.sigma = args.parity = True

    if not (args.verify or args.sigma or args.parity):
        print("Usage: python3 nba_audit_fixes.py --verify --sigma --parity")
        print("  --verify   Check CRIT-1 and CRIT-2 fixes")
        print("  --sigma    Brier sweep for optimal sigma")
        print("  --parity   Train/serve feature parity")
        print("  --all      Run all checks")
        sys.exit(0)

    if args.verify:
        verify_crit_fixes()
    if args.sigma:
        sigma_sweep()
    if args.parity:
        parity_check()
