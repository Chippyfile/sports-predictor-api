#!/usr/bin/env python3
"""
nba_lasso_walkforward.py — Walk-forward backtest for Lasso solo model

Tests the Lasso (70 Lasso-selected features) architecture with proper
walk-forward validation: train on all prior seasons, predict the next.

Outputs: accuracy, MAE, Brier, ATS by disagreement threshold, Vegas comparison.

Usage:
  python3 nba_lasso_walkforward.py
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


def load_and_prep():
    """Load data with exact retrain_nba.py pipeline."""
    df = pd.read_parquet("nba_training_data.parquet")
    print(f"  Loaded {len(df)} rows")

    # Elo
    try:
        from nba_elo import compute_all_elo, merge_elo_into_df
        elo_df, current_ratings = compute_all_elo(df)
        df = merge_elo_into_df(df, elo_df)
        print(f"  Elo: {len(elo_df)} games")
    except Exception as e:
        print(f"  WARNING: Elo failed ({e})")
        df["home_elo"] = 1500; df["away_elo"] = 1500; df["elo_diff"] = 0.0

    # Heuristic backfill
    try:
        from sports.nba import _nba_backfill_heuristic
        df = _nba_backfill_heuristic(df)
        print(f"  Heuristic backfill done")
    except Exception as e:
        print(f"  WARNING: Backfill failed ({e})")
        if "pred_home_score" not in df.columns: df["pred_home_score"] = df.get("home_ppg", 110)
        if "pred_away_score" not in df.columns: df["pred_away_score"] = df.get("away_ppg", 110)
        if "win_pct_home" not in df.columns: df["win_pct_home"] = 0.5
        if "model_ml_home" not in df.columns: df["model_ml_home"] = 0

    # Enrichment v2
    try:
        from enrich_nba_v2 import enrich as enrich_v2
        df = enrich_v2(df)
        print(f"  Enrichment v2 done")
    except ImportError:
        print("  WARNING: enrich_nba_v2.py not found")

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

    # Targets
    hs = pd.to_numeric(df["actual_home_score"], errors="coerce").fillna(0)
    aws = pd.to_numeric(df["actual_away_score"], errors="coerce").fillna(0)
    y_margin = hs - aws
    y_win = (y_margin > 0).astype(int)

    # Weights
    if "season_weight" in df.columns:
        weights = pd.to_numeric(df["season_weight"], errors="coerce").fillna(1.0).values
    else:
        weights = np.ones(len(df))
    weights = np.clip(weights, 0.1, 2.0)

    # Season labels
    gd = pd.to_datetime(df["game_date"])
    seasons = gd.apply(lambda d: d.year if d.month >= 9 else d.year - 1).astype(int) + 1
    # e.g. games in Oct 2024 -> Apr 2025 = season 2025

    # Market data
    spread = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    has_market = spread.notna() & (spread != 0)

    print(f"  Ready: {len(X)} games, {len(X.columns)} features")
    print(f"  Seasons: {sorted(seasons.unique())}")

    return X, y_margin, y_win, weights, seasons, spread, has_market, df


def lasso_select_features(X, y, w, alpha=0.1):
    """Run Lasso feature selection, return survivor list."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X_s, y, sample_weight=w)
    survivors = [f for f, c in zip(X.columns, lasso.coef_) if abs(c) > 0.001]
    return survivors


def walkforward():
    """Run walk-forward backtest: train on all prior seasons, test on next."""
    X, y_margin, y_win, weights, seasons, spread, has_market, df = load_and_prep()

    all_seasons = sorted(seasons.unique())
    # Need at least 2 seasons of training data
    test_seasons = [s for s in all_seasons if s >= all_seasons[2]]

    print(f"\n{'='*70}")
    print(f"  LASSO WALK-FORWARD BACKTEST")
    print(f"  Test seasons: {test_seasons}")
    print(f"{'='*70}")

    results = []

    for test_szn in test_seasons:
        train_mask = seasons < test_szn
        test_mask = seasons == test_szn

        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue

        X_train = X[train_mask]
        y_train = y_margin[train_mask]
        w_train = weights[train_mask]
        X_test = X[test_mask]
        y_test = y_margin[test_mask]
        spread_test = spread[test_mask]
        has_mkt_test = has_market[test_mask]
        win_test = y_win[test_mask]

        # Step 1: Lasso feature selection on training data
        survivors = lasso_select_features(X_train, y_train, w_train)
        n_feat = len(survivors)

        # Step 2: Train Lasso on selected features
        X_tr_slim = X_train[survivors]
        X_te_slim = X_test[survivors]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_slim)
        X_te_s = scaler.transform(X_te_slim)

        lasso = Lasso(alpha=0.1, max_iter=5000)
        lasso.fit(X_tr_s, y_train, sample_weight=w_train)
        pred_margin = lasso.predict(X_te_s)

        # Step 3: Calibrate win probability
        # Use training data for isotonic calibration
        train_pred = lasso.predict(X_tr_s)
        raw_prob = 1.0 / (1.0 + np.exp(-train_pred / 8.0))  # sigma
        train_win = (y_train > 0).astype(int)

        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_prob, train_win)
            test_raw_prob = 1.0 / (1.0 + np.exp(-pred_margin / 8.0))
            cal_prob = iso.predict(test_raw_prob)
        except:
            cal_prob = 1.0 / (1.0 + np.exp(-pred_margin / 8.0))

        # Step 4: Compute metrics
        mae = np.abs(y_test.values - pred_margin).mean()
        acc = ((pred_margin > 0) == (y_test.values > 0)).mean()
        brier = np.mean((cal_prob - win_test.values) ** 2)

        # ATS: model predicted margin vs market spread
        mkt_mask = has_mkt_test.values
        if mkt_mask.sum() > 0:
            sp = spread_test.values[mkt_mask]
            pm = pred_margin[mkt_mask]
            am = y_test.values[mkt_mask]

            # Model covers = model disagrees with spread AND model is right
            # actual_margin + spread > 0 means home covers
            actual_covers = (am + sp) > 0
            model_picks_home = (pm + sp) > 0  # model thinks home covers
            # Push handling
            push = (am + sp) == 0
            non_push = ~push

            if non_push.sum() > 0:
                ats_correct = (model_picks_home[non_push] == actual_covers[non_push]).mean()
                ats_n = non_push.sum()
            else:
                ats_correct = float("nan")
                ats_n = 0

            # Disagreement = |pred_margin - (-spread)|
            disagreement = np.abs(pm - (-sp))

            # Vegas MAE on same games
            vegas_mae = np.abs(am - (-sp)).mean()
            model_mae_mkt = np.abs(am - pm).mean()
        else:
            ats_correct = float("nan")
            ats_n = 0
            disagreement = np.array([])
            vegas_mae = float("nan")
            model_mae_mkt = float("nan")

        result = {
            "season": int(test_szn),
            "n_games": int(test_mask.sum()),
            "n_features": n_feat,
            "accuracy": round(acc, 4),
            "mae": round(mae, 3),
            "brier": round(brier, 4),
            "ats": round(ats_correct, 4) if not np.isnan(ats_correct) else None,
            "ats_n": int(ats_n),
            "vegas_mae": round(vegas_mae, 3) if not np.isnan(vegas_mae) else None,
            "model_mae_mkt": round(model_mae_mkt, 3) if not np.isnan(model_mae_mkt) else None,
        }

        # Store per-game results for ATS-by-disagreement
        if len(disagreement) > 0:
            result["_disagreement"] = disagreement
            result["_ats_correct"] = (model_picks_home[non_push] == actual_covers[non_push]) if non_push.sum() > 0 else np.array([])
            result["_disagree_vals"] = disagreement[non_push] if non_push.sum() > 0 else np.array([])

        results.append(result)

        vegas_str = f"  Vegas={vegas_mae:.2f}" if not np.isnan(vegas_mae) else ""
        ats_str = f"  ATS={ats_correct:.1%}({ats_n})" if not np.isnan(ats_correct) else ""
        print(f"  {test_szn}: Acc={acc:.1%}  MAE={mae:.2f}{ats_str}{vegas_str}  [{n_feat} feat]")

    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE RESULTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*70}")

    total_games = sum(r["n_games"] for r in results)
    total_mae = np.average([r["mae"] for r in results],
                           weights=[r["n_games"] for r in results])
    total_acc = np.average([r["accuracy"] for r in results],
                           weights=[r["n_games"] for r in results])
    total_brier = np.average([r["brier"] for r in results],
                              weights=[r["n_games"] for r in results])

    ats_results = [r for r in results if r["ats"] is not None and r["ats_n"] > 0]
    if ats_results:
        total_ats_n = sum(r["ats_n"] for r in ats_results)
        total_ats = np.average([r["ats"] for r in ats_results],
                               weights=[r["ats_n"] for r in ats_results])
    else:
        total_ats = float("nan")
        total_ats_n = 0

    # Vegas comparison
    vegas_results = [r for r in results if r["vegas_mae"] is not None]
    if vegas_results:
        total_vegas_mae = np.average([r["vegas_mae"] for r in vegas_results],
                                      weights=[r["ats_n"] for r in vegas_results])
        total_model_mae_mkt = np.average([r["model_mae_mkt"] for r in vegas_results],
                                          weights=[r["ats_n"] for r in vegas_results])
    else:
        total_vegas_mae = float("nan")
        total_model_mae_mkt = float("nan")

    print(f"\n  Games:     {total_games}")
    print(f"  Accuracy:  {total_acc:.1%}")
    print(f"  MAE:       {total_mae:.3f}")
    print(f"  Brier:     {total_brier:.4f}")
    print(f"  ATS:       {total_ats:.1%} ({total_ats_n} games)")

    if not np.isnan(total_vegas_mae):
        print(f"\n  Model MAE (mkt games): {total_model_mae_mkt:.3f}")
        print(f"  Vegas MAE (mkt games): {total_vegas_mae:.3f}")
        diff = total_vegas_mae - total_model_mae_mkt
        winner = "MODEL" if diff > 0 else "VEGAS"
        print(f"  Advantage: {diff:+.3f} ({winner} wins)")

    # ═══════════════════════════════════════════════════════════════
    # ATS BY DISAGREEMENT THRESHOLD
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  ATS BY DISAGREEMENT THRESHOLD")
    print(f"{'='*70}")

    # Collect all per-game ATS data
    all_disagree = []
    all_correct = []
    for r in results:
        if "_disagree_vals" in r and len(r["_disagree_vals"]) > 0:
            all_disagree.extend(r["_disagree_vals"])
            all_correct.extend(r["_ats_correct"])

    if all_disagree:
        all_disagree = np.array(all_disagree)
        all_correct = np.array(all_correct)

        print(f"\n  {'Threshold':>10s}  {'ATS%':>6s}  {'Games':>6s}  {'ROI':>7s}")
        print(f"  {'-'*35}")
        for thresh in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]:
            mask = all_disagree >= thresh
            if mask.sum() < 20:
                continue
            ats_pct = all_correct[mask].mean()
            n = mask.sum()
            roi = (ats_pct * 2.0 * 0.909 - 1.0) * 100  # -110 odds
            print(f"  {thresh:>8d}+    {ats_pct:>5.1%}  {n:>6d}  {roi:>+6.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # PER-SEASON DETAIL TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PER-SEASON BREAKDOWN")
    print(f"{'='*70}")
    print(f"\n  {'Season':>6s}  {'Acc':>6s}  {'ATS':>6s}  {'ATS_N':>6s}  {'MAE':>6s}  {'Brier':>6s}  {'vMAE':>6s}  {'mMAE':>6s}")
    print(f"  {'-'*60}")
    for r in results:
        ats_str = f"{r['ats']:.1%}" if r["ats"] is not None else "  n/a"
        vegas_str = f"{r['vegas_mae']:.2f}" if r["vegas_mae"] is not None else "  n/a"
        model_str = f"{r['model_mae_mkt']:.2f}" if r["model_mae_mkt"] is not None else "  n/a"
        print(f"  {r['season']:>6d}  {r['accuracy']:.1%}  {ats_str:>6s}  {r['ats_n']:>6d}  {r['mae']:>5.2f}  {r['brier']:.4f}  {vegas_str:>6s}  {model_str:>6s}")

    # ═══════════════════════════════════════════════════════════════
    # CALIBRATION CHECK
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  CALIBRATION (predicted prob vs actual win%)")
    print(f"{'='*70}")

    # Re-run to collect all calibrated probs and outcomes
    # (simplified: use pred_margin -> sigma -> bucket)
    all_pred_margins = []
    all_actual_wins = []
    for r in results:
        # We need to reconstruct — use stored results
        pass  # Would need to store per-game data; skip for now

    print(f"\n  (Run with --detailed for per-game calibration output)")

    # Save results
    save_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    summary = {
        "model": "Lasso_solo_70feat",
        "total_games": total_games,
        "accuracy": round(total_acc, 4),
        "mae": round(total_mae, 3),
        "brier": round(total_brier, 4),
        "ats": round(total_ats, 4) if not np.isnan(total_ats) else None,
        "ats_n": total_ats_n,
        "vegas_mae": round(total_vegas_mae, 3) if not np.isnan(total_vegas_mae) else None,
        "model_mae_mkt": round(total_model_mae_mkt, 3) if not np.isnan(total_model_mae_mkt) else None,
        "per_season": save_results,
    }

    with open("nba_lasso_walkforward_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: nba_lasso_walkforward_results.json")

    # Also save per-game CSV for deeper analysis
    if all_disagree:
        csv_data = pd.DataFrame({
            "disagreement": all_disagree,
            "ats_correct": all_correct.astype(int),
        })
        csv_data.to_csv("nba_lasso_ats_detail.csv", index=False)
        print(f"  Saved: nba_lasso_ats_detail.csv ({len(csv_data)} games)")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    walkforward()
    print(f"\n  Total time: {time.time()-t0:.0f}s")
