#!/usr/bin/env python3
"""
nba_walkforward_detail.py — Per-game walkforward prediction log

Runs the same Lasso v26 walkforward as nba_lasso_walkforward.py but saves
EVERY game's prediction with full metadata:
  - game_date, season, home_team, away_team
  - actual_margin, predicted_margin, error
  - market_spread, vegas_pred, disagree (model vs Vegas)
  - ats_correct, model_pick_side, actual_cover_side
  - win_prob, actual_winner, ml_correct

Outputs:
  1. nba_walkforward_detail.csv — per-game predictions
  2. Console analysis: ATS by team, month, spread range, season

Usage:
  python3 nba_walkforward_detail.py              # full walkforward + analysis
  python3 nba_walkforward_detail.py --analyze    # re-read CSV, skip training
"""

import pandas as pd
import numpy as np
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

OUTPUT_CSV = "nba_walkforward_detail.csv"


def load_data():
    """Load and prepare data — mirrors retrain_nba.py pipeline exactly."""
    df = pd.read_parquet("nba_training_data.parquet")
    print(f"  Loaded {len(df)} rows")

    # Elo
    try:
        from nba_elo import compute_all_elo, merge_elo_into_df
        elo_df, _ = compute_all_elo(df)
        df = merge_elo_into_df(df, elo_df)
        print(f"  Elo: {len(elo_df)} games")
    except Exception as e:
        print(f"  WARNING: Elo failed ({e})")
        df["home_elo"] = 1500; df["away_elo"] = 1500; df["elo_diff"] = 0.0

    # Heuristic backfill
    try:
        from sports.nba import _nba_backfill_heuristic
        df = _nba_backfill_heuristic(df)
    except Exception as e:
        print(f"  WARNING: Backfill failed ({e})")
        if "pred_home_score" not in df.columns: df["pred_home_score"] = df.get("home_ppg", 110)
        if "pred_away_score" not in df.columns: df["pred_away_score"] = df.get("away_ppg", 110)

    # Enrichment v2
    try:
        from enrich_nba_v2 import enrich as enrich_v2
        df = enrich_v2(df)
    except ImportError:
        print("  WARNING: enrich_nba_v2 not found")

    # Quality filter
    _quality_cols = ["home_ppg", "away_ppg", "home_fgpct", "away_fgpct",
                     "market_spread_home", "market_ou_total"]
    _qcols = [c for c in _quality_cols if c in df.columns]
    if _qcols:
        _qmat = pd.DataFrame({
            c: df[c].notna() & (df[c] != 0 if c in ["market_spread_home", "market_ou_total"] else True)
            for c in _qcols
        })
        df = df.loc[_qmat.mean(axis=1) >= 0.60].reset_index(drop=True)
    print(f"  After quality filter: {len(df)} games")

    # Build features
    from nba_build_features_v25 import nba_build_features
    X = nba_build_features(df)

    y_margin = pd.to_numeric(df["actual_home_score"], errors="coerce").fillna(0) - \
               pd.to_numeric(df["actual_away_score"], errors="coerce").fillna(0)

    # Season weights
    if "season" in df.columns:
        latest = pd.to_numeric(df["season"], errors="coerce").max()
        weights = df["season"].apply(
            lambda s: 1.0 if pd.to_numeric(s, errors="coerce") >= latest - 1
            else 0.85 if pd.to_numeric(s, errors="coerce") >= latest - 2
            else 0.7
        ).values
    else:
        weights = np.ones(len(df))
    weights = np.clip(weights, 0.1, 2.0)

    print(f"  Ready: {len(X)} games, {len(X.columns)} features")
    return X, y_margin, weights, df


def run_walkforward(X, y_margin, weights, df):
    """Walk-forward by season: train on all prior seasons, predict current."""

    gd = pd.to_datetime(df["game_date"])
    df["_season"] = gd.apply(lambda d: d.year + 1 if d.month >= 10 else d.year)
    seasons = sorted(df["_season"].unique())

    # Need at least 2 seasons of training
    test_seasons = [s for s in seasons if (df["_season"] < s).sum() >= 1000]
    print(f"\n  Test seasons: {test_seasons}")

    all_rows = []

    for season in test_seasons:
        train_mask = df["_season"] < season
        test_mask = df["_season"] == season

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_margin[train_mask], y_margin[test_mask]
        w_train = weights[train_mask]
        df_test = df[test_mask]

        if len(X_test) < 10:
            continue

        t0 = time.time()

        # Lasso feature selection on training data
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

        lasso_sel = Lasso(alpha=0.1, max_iter=5000)
        lasso_sel.fit(X_train_s, y_train, sample_weight=w_train)
        survivors = [f for f, c in zip(X_train.columns, lasso_sel.coef_) if abs(c) > 0.001]

        if len(survivors) < 5:
            print(f"  Season {season}: only {len(survivors)} features survived, skipping")
            continue

        # Train final Lasso on selected features
        X_train_slim = X_train[survivors]
        X_test_slim = X_test[survivors]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train_slim)
        X_te_s = sc.transform(X_test_slim)

        model = Lasso(alpha=0.1, max_iter=5000)
        model.fit(X_tr_s, y_train, sample_weight=w_train)

        preds = model.predict(X_te_s)

        # Isotonic calibration (train on OOF from training set)
        # Simple: use training set predictions for calibration fitting
        train_preds = model.predict(X_tr_s)
        train_prob = 1.0 / (1.0 + np.exp(-train_preds / 8.0))
        train_wins = (y_train > 0).astype(int).values
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_prob, train_wins)

        dt = time.time() - t0

        # Build per-game rows
        spread = pd.to_numeric(df_test["market_spread_home"], errors="coerce").fillna(0).values

        for i, (idx, row) in enumerate(df_test.iterrows()):
            pred_margin = float(preds[i])
            actual_margin = float(y_margin.iloc[df.index.get_loc(idx)])
            sp = float(spread[i])

            # Win probability
            raw_prob = 1.0 / (1.0 + np.exp(-pred_margin / 8.0))
            cal_prob = float(iso.predict([raw_prob])[0])

            # ATS
            has_spread = sp != 0
            if has_spread:
                vegas_pred = -sp
                disagree = abs(pred_margin - vegas_pred)
                actual_covers = (actual_margin + sp) > 0
                model_picks_cover = (pred_margin + sp) > 0
                push = (actual_margin + sp) == 0
                ats_correct = (model_picks_cover == actual_covers) if not push else np.nan
                model_side = "home" if pred_margin > vegas_pred else "away"
                actual_side = "home" if actual_covers else "away"
            else:
                vegas_pred = np.nan
                disagree = np.nan
                ats_correct = np.nan
                model_side = ""
                actual_side = ""
                push = False

            # ML correct
            ml_correct = (pred_margin > 0) == (actual_margin > 0) if actual_margin != 0 else np.nan

            game_row = {
                "game_date": row.get("game_date", ""),
                "season": int(season),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "actual_home_score": row.get("actual_home_score", 0),
                "actual_away_score": row.get("actual_away_score", 0),
                "actual_margin": round(actual_margin, 1),
                "predicted_margin": round(pred_margin, 2),
                "error": round(abs(actual_margin - pred_margin), 2),
                "market_spread": round(sp, 1) if has_spread else np.nan,
                "vegas_pred": round(vegas_pred, 1) if has_spread else np.nan,
                "disagree": round(disagree, 2) if has_spread else np.nan,
                "win_prob_home": round(cal_prob, 4),
                "ml_correct": ml_correct,
                "ats_correct": ats_correct,
                "push": push,
                "model_side": model_side,
                "actual_side": actual_side,
                "n_features": len(survivors),
            }
            all_rows.append(game_row)

        # Season summary
        season_df = pd.DataFrame([r for r in all_rows if r["season"] == season])
        mae = season_df["error"].mean()
        acc = season_df["ml_correct"].mean()
        ats_games = season_df.dropna(subset=["ats_correct"])
        ats_pct = ats_games["ats_correct"].mean() if len(ats_games) > 0 else float("nan")
        print(f"  Season {season}: {len(season_df)} games, MAE={mae:.3f}, "
              f"Acc={acc:.1%}, ATS={ats_pct:.1%} ({len(ats_games)}), "
              f"{len(survivors)} features, {dt:.1f}s")

    return pd.DataFrame(all_rows)


def analyze(detail_df):
    """Comprehensive analysis of walkforward predictions."""

    print("\n" + "=" * 70)
    print("  WALKFORWARD ANALYSIS")
    print("=" * 70)

    # Overall
    n = len(detail_df)
    print(f"\n  Total games: {n}")
    print(f"  MAE: {detail_df['error'].mean():.3f}")
    print(f"  Accuracy: {detail_df['ml_correct'].mean():.1%}")

    ats = detail_df.dropna(subset=["ats_correct"])
    ats_no_push = ats[ats["push"] == False]
    print(f"  ATS: {ats_no_push['ats_correct'].mean():.1%} ({len(ats_no_push)} games)")

    # ── ATS by disagreement threshold ──
    print(f"\n  {'Disagree':>10s}  {'ATS%':>6s}  {'Games':>6s}  {'ROI':>7s}")
    print(f"  {'-'*35}")
    for thresh in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]:
        mask = ats_no_push["disagree"] >= thresh
        subset = ats_no_push[mask]
        if len(subset) >= 20:
            pct = subset["ats_correct"].mean()
            roi = (pct * 1.9091 - 1) * 100  # -110 odds
            print(f"  {thresh:>8d}+   {pct:>5.1%}  {len(subset):>6d}  {roi:>+6.1f}%")

    # ── ATS by season ──
    print(f"\n  ATS by Season:")
    print(f"  {'Season':>8s}  {'ATS%':>6s}  {'Games':>6s}  {'MAE':>6s}  {'Acc':>6s}")
    print(f"  {'-'*40}")
    for szn in sorted(detail_df["season"].unique()):
        s = detail_df[detail_df["season"] == szn]
        s_ats = s.dropna(subset=["ats_correct"])
        s_ats = s_ats[s_ats["push"] == False]
        ats_pct = s_ats["ats_correct"].mean() if len(s_ats) > 0 else float("nan")
        print(f"  {szn:>8d}  {ats_pct:>5.1%}  {len(s_ats):>6d}  "
              f"{s['error'].mean():>5.2f}  {s['ml_correct'].mean():>5.1%}")

    # ── ATS by month ──
    print(f"\n  ATS by Month:")
    detail_df["_month"] = pd.to_datetime(detail_df["game_date"]).dt.strftime("%Y-%m")
    months = sorted(detail_df["_month"].unique())
    print(f"  {'Month':>8s}  {'ATS%':>6s}  {'Games':>6s}  {'MAE':>6s}")
    print(f"  {'-'*35}")
    for m in months[-24:]:  # last 24 months
        s = detail_df[detail_df["_month"] == m]
        s_ats = s.dropna(subset=["ats_correct"])
        s_ats = s_ats[s_ats["push"] == False]
        if len(s_ats) >= 10:
            ats_pct = s_ats["ats_correct"].mean()
            print(f"  {m:>8s}  {ats_pct:>5.1%}  {len(s_ats):>6d}  {s['error'].mean():>5.2f}")

    # ── ATS by spread range ──
    print(f"\n  ATS by Spread Range:")
    print(f"  {'Range':>12s}  {'ATS%':>6s}  {'Games':>6s}")
    print(f"  {'-'*30}")
    for lo, hi, label in [(-25, -10, "Big fav"), (-10, -4, "Med fav"),
                          (-4, -1, "Sm fav"), (-1, 1, "PK"),
                          (1, 4, "Sm dog"), (4, 10, "Med dog"), (10, 25, "Big dog")]:
        mask = (ats_no_push["market_spread"] >= lo) & (ats_no_push["market_spread"] < hi)
        subset = ats_no_push[mask]
        if len(subset) >= 20:
            pct = subset["ats_correct"].mean()
            print(f"  {label:>12s}  {pct:>5.1%}  {len(subset):>6d}")

    # ── ATS by team (best and worst) ──
    print(f"\n  ATS by Team (≥30 games, top 10 / bottom 10):")
    team_ats = []
    for team in detail_df["home_team"].unique():
        home = ats_no_push[ats_no_push["home_team"] == team]
        away = ats_no_push[ats_no_push["away_team"] == team]
        combined = pd.concat([home, away])
        if len(combined) >= 30:
            team_ats.append({
                "team": team,
                "ats_pct": combined["ats_correct"].mean(),
                "games": len(combined),
            })

    team_ats.sort(key=lambda x: x["ats_pct"], reverse=True)
    print(f"  {'Team':>6s}  {'ATS%':>6s}  {'Games':>6s}")
    print(f"  {'-'*22}")
    for t in team_ats[:10]:
        print(f"  {t['team']:>6s}  {t['ats_pct']:>5.1%}  {t['games']:>6d}")
    print(f"  {'...':>6s}")
    for t in team_ats[-10:]:
        print(f"  {t['team']:>6s}  {t['ats_pct']:>5.1%}  {t['games']:>6d}")

    # ── High-confidence picks (4+ disagree, last 2 seasons) ──
    recent = detail_df[detail_df["season"] >= detail_df["season"].max() - 1]
    recent_ats = recent.dropna(subset=["ats_correct"])
    recent_ats = recent_ats[(recent_ats["push"] == False) & (recent_ats["disagree"] >= 4)]
    if len(recent_ats) > 0:
        print(f"\n  High-Confidence Picks (4+ disagree, last 2 seasons): "
              f"{recent_ats['ats_correct'].mean():.1%} ({len(recent_ats)} games)")

    # ── Model vs Vegas MAE ──
    has_mkt = ats_no_push.dropna(subset=["vegas_pred"])
    if len(has_mkt) > 0:
        model_mae = has_mkt["error"].mean()
        vegas_mae = (has_mkt["actual_margin"] - has_mkt["vegas_pred"]).abs().mean()
        print(f"\n  Model MAE vs Vegas (games with spreads):")
        print(f"    Model: {model_mae:.3f}")
        print(f"    Vegas: {vegas_mae:.3f}")
        print(f"    Edge:  {vegas_mae - model_mae:+.3f}")

    # ── Calibration ──
    print(f"\n  Win Probability Calibration:")
    print(f"  {'Bucket':>12s}  {'Predicted':>9s}  {'Actual':>7s}  {'N':>5s}  {'Gap':>6s}")
    for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]:
        mask = (detail_df["win_prob_home"] >= lo) & (detail_df["win_prob_home"] < hi)
        subset = detail_df[mask]
        if len(subset) >= 20:
            pred_avg = subset["win_prob_home"].mean()
            act_avg = subset["ml_correct"].mean()
            gap = pred_avg - act_avg
            print(f"  {lo:.1f}-{hi:.1f}      {pred_avg:.3f}      {act_avg:.3f}  {len(subset):>5d}  {gap:+.3f}")


def main():
    args = sys.argv[1:]

    if "--analyze" in args:
        if not pd.io.common.file_exists(OUTPUT_CSV):
            print(f"ERROR: {OUTPUT_CSV} not found. Run walkforward first.")
            return
        detail_df = pd.read_csv(OUTPUT_CSV)
        analyze(detail_df)
        return

    print("=" * 70)
    print("  NBA Lasso v26 — Per-Game Walkforward Detail")
    print("=" * 70)

    t0 = time.time()
    X, y_margin, weights, df = load_data()
    detail_df = run_walkforward(X, y_margin, weights, df)

    # Save
    detail_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved {OUTPUT_CSV}: {len(detail_df)} rows")

    # Analyze
    analyze(detail_df)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
