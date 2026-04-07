#!/usr/bin/env python3
"""
mlb_ou_v2_eval.py — Evaluate each triple-agreement component independently
Shows standalone performance of residual, classifier, and ATS models
so we can see what each brings to the table before combining.
"""
import sys, os
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor

SEED = 42
N_FOLDS = 30

from mlb_retrain import load_data
from mlb_ou_retrain import build_ou_features

# Feature sets (same as training script)
RES_FEATURE_COLS = [
    "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold", "temp_x_park",
    "fip_combined", "bullpen_combined", "k_bb_combined", "sp_ip_combined", "bp_exposure_combined",
    "woba_combined", "woba_diff", "ump_run_env",
    "scoring_entropy_combined", "first_inn_rate_combined", "rest_combined", "series_game_num", "lg_rpg",
]
CLS_FEATURE_COLS = [
    "market_total", "park_factor", "temp_f", "wind_out", "temp_x_park",
    "fip_combined", "bullpen_combined", "woba_combined", "ump_run_env",
    "scoring_entropy_combined", "is_warm", "is_cold",
    "sp_ip_combined", "bp_exposure_combined", "rest_combined", "k_bb_combined", "lg_rpg",
]
ATS_FEATURE_COLS = [
    "woba_diff", "fip_diff", "fip_combined", "k_bb_combined", "bullpen_combined",
    "sp_ip_combined", "bp_exposure_combined", "park_factor", "temp_f", "wind_out", "temp_x_park",
    "ump_run_env", "rest_combined", "series_game_num", "woba_combined",
    "scoring_entropy_combined", "first_inn_rate_combined", "market_total", "lg_rpg",
]

# Winning configs from the architecture search
RES_BUILDERS = [
    ("Lasso_0.1", lambda: Lasso(alpha=0.1, max_iter=5000, random_state=SEED)),
    ("LGBM_d3_200", lambda: LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
    ("CatBoost_d3_200", lambda: CatBoostRegressor(iterations=200, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)),
]
CLS_BUILDERS = [
    ("LogReg_C0.5", lambda: LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)),
    ("LogReg_C0.1", lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)),
]
ATS_BUILDERS = [
    ("Lasso_0.1", lambda: Lasso(alpha=0.1, max_iter=5000, random_state=SEED)),
    ("LGBM_d4", lambda: LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
    ("CatBoost_d4", lambda: CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0)),
]

MODERN_TO_RETRO = {
    "LAA": "ANA", "CWS": "CHA", "CHC": "CHN", "KC": "KCA",
    "LAD": "LAN", "NYY": "NYA", "NYM": "NYN", "SD": "SDN",
    "SF": "SFN", "STL": "SLN", "TB": "TBA", "WSH": "WAS",
}


def build_all_features(df):
    ou_df = build_ou_features(df)
    if "woba_diff" not in ou_df.columns:
        ou_df["woba_diff"] = pd.to_numeric(df.get("home_woba", 0.315), errors="coerce").fillna(0.315) - \
                              pd.to_numeric(df.get("away_woba", 0.315), errors="coerce").fillna(0.315)
    if "fip_diff" not in ou_df.columns:
        ou_df["fip_diff"] = pd.to_numeric(df.get("home_sp_fip", 4.25), errors="coerce").fillna(4.25) - \
                             pd.to_numeric(df.get("away_sp_fip", 4.25), errors="coerce").fillna(4.25)
    return ou_df


def main():
    print("=" * 70)
    print("  MLB O/U v2 — INDIVIDUAL COMPONENT EVALUATION")
    print("=" * 70)

    # ── Load + backfill (same as training script) ──
    df = load_data()
    odds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_odds_2014_2021.csv")
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path, usecols=["game_date", "home_team", "away_team", "market_ou_total",
                                                 "market_spread_home", "market_home_ml", "market_away_ml"])
        odds["home_team"] = odds["home_team"].replace(MODERN_TO_RETRO)
        odds["away_team"] = odds["away_team"].replace(MODERN_TO_RETRO)
        odds = odds.rename(columns={"market_ou_total": "_ou", "market_spread_home": "_sp",
                                     "market_home_ml": "_hml", "market_away_ml": "_aml"})
        df = df.merge(odds, on=["game_date", "home_team", "away_team"], how="left")
        for col, oc in [("market_ou_total", "_ou"), ("market_spread_home", "_sp"),
                         ("market_home_ml", "_hml"), ("market_away_ml", "_aml")]:
            if col not in df.columns: df[col] = df[oc]
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].where(df[col] > 0, other=None).fillna(df[oc])
        df = df.drop(columns=["_ou", "_sp", "_hml", "_aml"], errors="ignore")

    df["actual_total"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) + \
                         pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    df["market_total_raw"] = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)
    df = df[(df["market_total_raw"] > 0) & (df["actual_total"] > 0)].copy()
    print(f"  Games: {len(df)}")

    X_all = build_all_features(df)
    mkt = df["market_total_raw"].values; actual = df["actual_total"].values
    yr = actual - mkt; yu = (actual < mkt).astype(int)
    yh = pd.to_numeric(df["actual_home_runs"], errors="coerce").fillna(0).values
    ya = pd.to_numeric(df["actual_away_runs"], errors="coerce").fillna(0).values
    w = df["season_weight"].values if "season_weight" in df.columns else np.ones(len(df))

    res_f = [f for f in RES_FEATURE_COLS if f in X_all.columns]
    cls_f = [f for f in CLS_FEATURE_COLS if f in X_all.columns]
    ats_f = [f for f in ATS_FEATURE_COLS if f in X_all.columns]
    Xr = X_all[res_f].fillna(0).values; Xc = X_all[cls_f].fillna(0).values; Xa = X_all[ats_f].fillna(0).values

    n = len(df); fold_size = n // N_FOLDS

    # ═══════════════════════════════════════════════════════════
    # WALK-FORWARD — collect all OOF predictions
    # ═══════════════════════════════════════════════════════════
    oof_res = np.full((len(RES_BUILDERS), n), np.nan)
    oof_cls = np.full((len(CLS_BUILDERS), n), np.nan)
    oof_ats = np.full(n, np.nan)
    oof_ats_home = np.full((len(ATS_BUILDERS), n), np.nan)
    oof_ats_away = np.full((len(ATS_BUILDERS), n), np.nan)

    print(f"\n  Running {N_FOLDS}-fold walk-forward...")
    for fold in range(N_FOLDS):
        ts = fold * fold_size
        te = min((fold + 1) * fold_size, n) if fold < N_FOLDS - 1 else n
        if ts < 300: continue

        # Residual
        scl = StandardScaler().fit(Xr[:ts])
        Xtr, Xte = scl.transform(Xr[:ts]), scl.transform(Xr[ts:te])
        for i, (nm, b) in enumerate(RES_BUILDERS):
            m = b()
            try: m.fit(Xtr, yr[:ts], sample_weight=w[:ts])
            except TypeError: m.fit(Xtr, yr[:ts])
            oof_res[i, ts:te] = m.predict(Xte)

        # Classifier
        scl = StandardScaler().fit(Xc[:ts])
        Xtr, Xte = scl.transform(Xc[:ts]), scl.transform(Xc[ts:te])
        for i, (nm, b) in enumerate(CLS_BUILDERS):
            m = b()
            try: m.fit(Xtr, yu[:ts], sample_weight=w[:ts])
            except TypeError: m.fit(Xtr, yu[:ts])
            oof_cls[i, ts:te] = m.predict_proba(Xte)[:, 1]

        # ATS scores
        scl = StandardScaler().fit(Xa[:ts])
        Xtr, Xte = scl.transform(Xa[:ts]), scl.transform(Xa[ts:te])
        hp, ap = [], []
        for i, (nm, b) in enumerate(ATS_BUILDERS):
            mh = b()
            try: mh.fit(Xtr, yh[:ts], sample_weight=w[:ts])
            except TypeError: mh.fit(Xtr, yh[:ts])
            oof_ats_home[i, ts:te] = mh.predict(Xte)
            hp.append(mh.predict(Xte))
            ma = b()
            try: ma.fit(Xtr, ya[:ts], sample_weight=w[:ts])
            except TypeError: ma.fit(Xtr, ya[:ts])
            oof_ats_away[i, ts:te] = ma.predict(Xte)
            ap.append(ma.predict(Xte))
        oof_ats[ts:te] = np.mean(hp, axis=0) + np.mean(ap, axis=0)

        if (fold + 1) % 10 == 0: print(f"    Fold {fold+1}/{N_FOLDS}")

    valid = ~np.isnan(oof_res[0])
    res_avg = np.mean(oof_res, axis=0)
    cls_avg = np.mean(oof_cls, axis=0)
    ats_edge = oof_ats - mkt

    # ═══════════════════════════════════════════════════════════
    # EVAL 1: RESIDUAL MODELS (individually + ensemble)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  RESIDUAL MODELS — Predicting market error (actual - market)")
    print(f"{'='*70}")
    print(f"  Target: mean={yr[valid].mean():.3f} std={yr[valid].std():.3f}")
    print(f"  {'Model':<25s} {'MAE':>7s} {'Dir%':>7s} {'Corr':>7s}")
    print(f"  {'-'*48}")
    for i, (nm, _) in enumerate(RES_BUILDERS):
        v = ~np.isnan(oof_res[i])
        mae = mean_absolute_error(yr[v], oof_res[i][v])
        nz = yr[v] != 0
        dir_acc = ((oof_res[i][v][nz] > 0) == (yr[v][nz] > 0)).mean()
        corr = np.corrcoef(oof_res[i][v], yr[v])[0, 1]
        print(f"  {nm:<25s} {mae:>7.3f} {dir_acc:>6.1%} {corr:>7.3f}")
    mae_ens = mean_absolute_error(yr[valid], res_avg[valid])
    nz = yr[valid] != 0
    dir_ens = ((res_avg[valid][nz] > 0) == (yr[valid][nz] > 0)).mean()
    corr_ens = np.corrcoef(res_avg[valid], yr[valid])[0, 1]
    print(f"  {'ENSEMBLE':<25s} {mae_ens:>7.3f} {dir_ens:>6.1%} {corr_ens:>7.3f}")
    print(f"  {'Market (baseline)':<25s} {mean_absolute_error(actual[valid], mkt[valid]):>7.3f}")

    # Residual standalone thresholds
    print(f"\n  Residual-only O/U thresholds (no classifier/ATS required):")
    print(f"  {'Threshold':<12s} {'Side':<7s} {'Games':>6s} {'Acc':>7s} {'ROI':>7s}")
    print(f"  {'-'*42}")
    for th in [-0.3, -0.5, -0.8, -1.0, -1.5, 0.3, 0.5, 0.8, 1.0, 1.5]:
        side = "UNDER" if th < 0 else "OVER"
        if th < 0:
            mask = valid & (res_avg <= th)
        else:
            mask = valid & (res_avg >= th)
        n_g = mask.sum()
        if n_g < 10: continue
        pushes = (actual[mask] == mkt[mask]).sum()
        decided = n_g - pushes
        if decided < 10: continue
        if side == "UNDER":
            correct = ((yu[mask] == 1) & (actual[mask] != mkt[mask])).sum()
        else:
            correct = ((yu[mask] == 0) & (actual[mask] != mkt[mask])).sum()
        acc = correct / decided
        roi = (acc * 0.909 - (1 - acc)) * 100
        print(f"  res{'≤' if th<0 else '≥'}{abs(th):<5.1f}   {side:<7s} {decided:>6d} {acc:>6.1%} {roi:>+6.1f}%")

    # ═══════════════════════════════════════════════════════════
    # EVAL 2: CLASSIFIER MODELS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  CLASSIFIER MODELS — Predicting P(under)")
    print(f"{'='*70}")
    print(f"  Base under rate: {yu[valid].mean():.3f}")
    print(f"  {'Model':<25s} {'Brier':>8s} {'Acc@50':>8s} {'AvgP':>8s}")
    print(f"  {'-'*52}")
    for i, (nm, _) in enumerate(CLS_BUILDERS):
        v = ~np.isnan(oof_cls[i])
        brier = np.mean((oof_cls[i][v] - yu[v]) ** 2)
        acc = ((oof_cls[i][v] > 0.5) == yu[v].astype(bool)).mean()
        avgp = oof_cls[i][v].mean()
        print(f"  {nm:<25s} {brier:>8.4f} {acc:>7.1%} {avgp:>8.3f}")
    brier_ens = np.mean((cls_avg[valid] - yu[valid]) ** 2)
    acc_ens = ((cls_avg[valid] > 0.5) == yu[valid].astype(bool)).mean()
    print(f"  {'ENSEMBLE':<25s} {brier_ens:>8.4f} {acc_ens:>7.1%} {cls_avg[valid].mean():>8.3f}")

    # Classifier standalone thresholds
    print(f"\n  Classifier-only O/U thresholds:")
    print(f"  {'Threshold':<12s} {'Side':<7s} {'Games':>6s} {'Acc':>7s} {'ROI':>7s}")
    print(f"  {'-'*42}")
    for th in [0.52, 0.54, 0.56, 0.58, 0.60, 0.48, 0.46, 0.44, 0.42, 0.40]:
        if th > 0.5:
            side = "UNDER"; mask = valid & (cls_avg >= th)
        else:
            side = "OVER"; mask = valid & (cls_avg <= th)
        n_g = mask.sum()
        if n_g < 10: continue
        pushes = (actual[mask] == mkt[mask]).sum()
        decided = n_g - pushes
        if decided < 10: continue
        if side == "UNDER":
            correct = ((yu[mask] == 1) & (actual[mask] != mkt[mask])).sum()
        else:
            correct = ((yu[mask] == 0) & (actual[mask] != mkt[mask])).sum()
        acc = correct / decided
        roi = (acc * 0.909 - (1 - acc)) * 100
        print(f"  cls{'≥' if th>0.5 else '≤'}{th:<5.2f}   {side:<7s} {decided:>6d} {acc:>6.1%} {roi:>+6.1f}%")

    # ═══════════════════════════════════════════════════════════
    # EVAL 3: ATS SCORE MODELS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  ATS SCORE MODELS — Predicting home/away runs")
    print(f"{'='*70}")
    print(f"  {'Model':<25s} {'Home MAE':>9s} {'Away MAE':>9s} {'Total MAE':>10s}")
    print(f"  {'-'*56}")
    for i, (nm, _) in enumerate(ATS_BUILDERS):
        v = ~np.isnan(oof_ats_home[i])
        h_mae = mean_absolute_error(yh[v], oof_ats_home[i][v])
        a_mae = mean_absolute_error(ya[v], oof_ats_away[i][v])
        t_mae = mean_absolute_error(actual[v], oof_ats_home[i][v] + oof_ats_away[i][v])
        print(f"  {nm:<25s} {h_mae:>9.3f} {a_mae:>9.3f} {t_mae:>10.3f}")
    v = ~np.isnan(oof_ats)
    t_mae_ens = mean_absolute_error(actual[v], oof_ats[v])
    print(f"  {'ENSEMBLE':<25s} {'':>9s} {'':>9s} {t_mae_ens:>10.3f}")
    print(f"  {'Market (baseline)':<25s} {'':>9s} {'':>9s} {mean_absolute_error(actual[v], mkt[v]):>10.3f}")

    # ATS-implied edge thresholds
    print(f"\n  ATS edge-only O/U thresholds (implied total vs market):")
    print(f"  {'Threshold':<12s} {'Side':<7s} {'Games':>6s} {'Acc':>7s} {'ROI':>7s}")
    print(f"  {'-'*42}")
    for th in [-0.5, -1.0, -1.5, -2.0, 0.5, 1.0, 1.5, 2.0]:
        side = "UNDER" if th < 0 else "OVER"
        if th < 0:
            mask = valid & (ats_edge <= th)
        else:
            mask = valid & (ats_edge >= th)
        n_g = mask.sum()
        if n_g < 10: continue
        pushes = (actual[mask] == mkt[mask]).sum()
        decided = n_g - pushes
        if decided < 10: continue
        if side == "UNDER":
            correct = ((yu[mask] == 1) & (actual[mask] != mkt[mask])).sum()
        else:
            correct = ((yu[mask] == 0) & (actual[mask] != mkt[mask])).sum()
        acc = correct / decided
        roi = (acc * 0.909 - (1 - acc)) * 100
        print(f"  ats{'≤' if th<0 else '≥'}{abs(th):<5.1f}   {side:<7s} {decided:>6d} {acc:>6.1%} {roi:>+6.1f}%")

    # ═══════════════════════════════════════════════════════════
    # EVAL 4: TRIPLE AGREEMENT vs INDIVIDUAL
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TRIPLE AGREEMENT vs BEST INDIVIDUAL COMPONENT")
    print(f"{'='*70}")
    # Compare: triple agreement UNDER vs residual-only UNDER vs classifier-only UNDER
    for label, side, mask_fn in [
        ("Triple U-2u", "UNDER", lambda: valid & (res_avg <= -0.3) & (cls_avg >= 0.50) & (ats_edge <= -0.5)),
        ("Residual ≤-0.3", "UNDER", lambda: valid & (res_avg <= -0.3)),
        ("Classifier ≥0.52", "UNDER", lambda: valid & (cls_avg >= 0.52)),
        ("ATS edge ≤-0.5", "UNDER", lambda: valid & (ats_edge <= -0.5)),
        ("Triple O-2u", "OVER", lambda: valid & (res_avg >= 0.8) & (cls_avg <= 0.36) & (ats_edge >= 1.5)),
        ("Residual ≥0.8", "OVER", lambda: valid & (res_avg >= 0.8)),
        ("Classifier ≤0.46", "OVER", lambda: valid & (cls_avg <= 0.46)),
        ("ATS edge ≥1.5", "OVER", lambda: valid & (ats_edge >= 1.5)),
    ]:
        mask = mask_fn()
        n_g = mask.sum()
        if n_g < 10: continue
        pushes = (actual[mask] == mkt[mask]).sum()
        decided = n_g - pushes
        if decided < 10: continue
        if side == "UNDER":
            correct = ((yu[mask] == 1) & (actual[mask] != mkt[mask])).sum()
        else:
            correct = ((yu[mask] == 0) & (actual[mask] != mkt[mask])).sum()
        acc = correct / decided
        roi = (acc * 0.909 - (1 - acc)) * 100
        print(f"  {label:<22s} {side:<7s} {decided:>6d} games  {acc:>6.1%}  ROI {roi:>+6.1f}%")


if __name__ == "__main__":
    main()
