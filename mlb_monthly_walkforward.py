#!/usr/bin/env python3
"""
mlb_monthly_walkforward.py — Month-by-month walk-forward analysis
==================================================================
Runs both O/U v3 and ATS v9 walk-forward, then breaks down accuracy by month.
Shows: does the model struggle in April? When does it peak?

Usage:
    python3 mlb_monthly_walkforward.py
"""
import numpy as np
import pandas as pd
import warnings, os, sys
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression, Ridge
from catboost import CatBoostRegressor, CatBoostClassifier
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════
# ATS v9 MONTHLY
# ═══════════════════════════════════════════════════════════════
def run_ats_monthly():
    from mlb_v9_retrain import load_data, V8_FEATURES, LINEUP_FEATURES, walk_forward, SEED

    print("=" * 70)
    print("  ATS v9 — MONTHLY WALK-FORWARD BREAKDOWN")
    print("=" * 70)

    X, y, spreads, has_spread, w, seasons, df = load_data()

    v9_feats = [f for f in V8_FEATURES if f in X.columns] + \
               [f for f in LINEUP_FEATURES if f in X.columns]

    print(f"  Features: {len(v9_feats)}")
    print(f"  Running walk-forward...")

    _, _, _, oof = walk_forward(X, y, w, v9_feats)

    # Get months
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    months = dates.dt.month.values
    years = dates.dt.year.values

    valid = ~np.isnan(oof) & has_spread
    ats = y + spreads
    push = ats == 0
    edge = np.abs(oof - (-spreads))

    # Monthly breakdown
    print(f"\n  {'Month':>8} {'Games':>7} {'ATS 0+':>8} {'ATS 1+':>8} {'ATS 2+':>8} {'ML%':>7} {'Avg Edge':>9}")
    print(f"  {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*9}")

    month_names = {3: "March", 4: "April", 5: "May", 6: "June",
                   7: "July", 8: "August", 9: "Sept", 10: "Oct"}

    for m in [3, 4, 5, 6, 7, 8, 9, 10]:
        mask = valid & (months == m) & ~push
        if mask.sum() < 30:
            continue

        p = oof[mask]; yv = y[mask]; sv = spreads[mask]
        atsv = ats[mask]; ev = edge[mask]

        # ATS at various thresholds
        model_home = p > (-sv)
        actual_home = atsv > 0
        ats_0 = (model_home == actual_home).mean()

        mask_1 = ev >= 1.0
        ats_1 = (model_home[mask_1] == actual_home[mask_1]).mean() if mask_1.sum() >= 20 else 0

        mask_2 = ev >= 2.0
        ats_2 = (model_home[mask_2] == actual_home[mask_2]).mean() if mask_2.sum() >= 10 else 0

        ml_correct = ((p > 0) == (yv > 0)).mean()
        avg_edge = ev.mean()

        name = month_names.get(m, str(m))
        n1 = mask_1.sum() if mask_1.sum() >= 20 else 0
        n2 = mask_2.sum() if mask_2.sum() >= 10 else 0
        print(f"  {name:>8} {mask.sum():>7} {ats_0:>7.1%} {ats_1:>6.1%}/{n1:<3} {ats_2:>6.1%}/{n2:<3} {ml_correct:>6.1%} {avg_edge:>8.2f}")

    # Early vs late season
    print(f"\n  {'Period':>12} {'Games':>7} {'ATS 1+':>10} {'ATS 2+':>10} {'ML%':>7}")
    print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*10} {'─'*7}")

    for label, month_range in [("Mar-Apr", [3, 4]), ("May-Jun", [5, 6]),
                                ("Jul-Aug", [7, 8]), ("Sep-Oct", [9, 10])]:
        mask = valid & np.isin(months, month_range) & ~push
        if mask.sum() < 30:
            continue
        p = oof[mask]; yv = y[mask]; sv = spreads[mask]
        atsv = ats[mask]; ev = edge[mask]
        model_home = p > (-sv); actual_home = atsv > 0

        mask_1 = ev >= 1.0
        ats_1 = (model_home[mask_1] == actual_home[mask_1]).mean() if mask_1.sum() >= 20 else 0
        n1 = mask_1.sum()

        mask_2 = ev >= 2.0
        ats_2 = (model_home[mask_2] == actual_home[mask_2]).mean() if mask_2.sum() >= 10 else 0
        n2 = mask_2.sum()

        ml_correct = ((p > 0) == (yv > 0)).mean()
        print(f"  {label:>12} {mask.sum():>7} {ats_1:>7.1%}/{n1:<4} {ats_2:>7.1%}/{n2:<4} {ml_correct:>6.1%}")


# ═══════════════════════════════════════════════════════════════
# O/U v3 MONTHLY
# ═══════════════════════════════════════════════════════════════
def run_ou_monthly():
    from mlb_retrain import load_data
    from mlb_retrain_ou_v3 import build_all_features, RES_FEATURE_COLS, CLS_FEATURE_COLS, ATS_FEATURE_COLS

    print(f"\n\n{'=' * 70}")
    print(f"  O/U v3 — MONTHLY WALK-FORWARD BREAKDOWN")
    print(f"{'=' * 70}")

    df = load_data()
    df["actual_total"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) + \
                         pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)
    df = df[(df["market_total"] > 0) & (df["actual_total"] > 0)].copy()

    X_all = build_all_features(df)
    mkt = df["market_total"].values
    actual = df["actual_total"].values
    yr = actual - mkt  # residual target
    yu = (actual < mkt).astype(int)  # under target

    dates = pd.to_datetime(df["game_date"], errors="coerce")
    months = dates.dt.month.values

    # Walk-forward for residual model (tree config - best from retrain)
    print(f"  Running O/U walk-forward...")
    n = len(X_all)
    n_folds = 20
    fs = n // (n_folds + 1)
    mt = max(fs * 3, 1000)

    res_cols = [c for c in RES_FEATURE_COLS if c in X_all.columns]
    cls_cols = [c for c in CLS_FEATURE_COLS if c in X_all.columns]
    ats_cols = [c for c in ATS_FEATURE_COLS if c in X_all.columns]

    oof_res = np.full(n, np.nan)
    oof_cls = np.full(n, np.nan)
    oof_ats_home = np.full(n, np.nan)
    oof_ats_away = np.full(n, np.nan)

    for fold in range(n_folds):
        ts = mt + fold * fs
        te = min(ts + fs, n)
        if ts >= n:
            break

        # Residual (tree ensemble)
        sc_r = StandardScaler()
        Xr_tr = sc_r.fit_transform(X_all[res_cols].iloc[:ts])
        Xr_te = sc_r.transform(X_all[res_cols].iloc[ts:te])

        m1 = CatBoostRegressor(depth=3, iterations=200, random_seed=SEED, verbose=0)
        m1.fit(Xr_tr, yr[:ts])
        if HAS_LGBM:
            m2 = LGBMRegressor(max_depth=3, n_estimators=200, random_state=SEED, verbose=-1)
            m2.fit(Xr_tr, yr[:ts])
            m3 = CatBoostRegressor(depth=5, iterations=300, random_seed=SEED+1, verbose=0)
            m3.fit(Xr_tr, yr[:ts])
            oof_res[ts:te] = (m1.predict(Xr_te) + m2.predict(Xr_te) + m3.predict(Xr_te)) / 3
        else:
            oof_res[ts:te] = m1.predict(Xr_te)

        # Classifier (tree)
        sc_c = StandardScaler()
        Xc_tr = sc_c.fit_transform(X_all[cls_cols].iloc[:ts])
        Xc_te = sc_c.transform(X_all[cls_cols].iloc[ts:te])

        if HAS_LGBM:
            mc1 = LGBMClassifier(max_depth=3, n_estimators=100, random_state=SEED, verbose=-1)
            mc1.fit(Xc_tr, yu[:ts])
            mc2 = CatBoostClassifier(depth=3, iterations=100, random_seed=SEED, verbose=0)
            mc2.fit(Xc_tr, yu[:ts])
            oof_cls[ts:te] = (mc1.predict_proba(Xc_te)[:, 1] + mc2.predict_proba(Xc_te)[:, 1]) / 2
        else:
            mc1 = CatBoostClassifier(depth=3, iterations=100, random_seed=SEED, verbose=0)
            mc1.fit(Xc_tr, yu[:ts])
            oof_cls[ts:te] = mc1.predict_proba(Xc_te)[:, 1]

        # ATS score (mixed)
        sc_a = StandardScaler()
        Xa_tr = sc_a.fit_transform(X_all[ats_cols].iloc[:ts])
        Xa_te = sc_a.transform(X_all[ats_cols].iloc[ts:te])

        home_runs = df["actual_home_runs"].iloc[:ts].astype(float).values
        away_runs = df["actual_away_runs"].iloc[:ts].astype(float).values

        ma1 = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        ma1.fit(Xa_tr, home_runs)
        if HAS_LGBM:
            ma2 = LGBMRegressor(max_depth=4, n_estimators=100, random_state=SEED, verbose=-1)
            ma2.fit(Xa_tr, home_runs)
            ma3 = CatBoostRegressor(depth=4, iterations=100, random_seed=SEED, verbose=0)
            ma3.fit(Xa_tr, home_runs)
            oof_ats_home[ts:te] = (ma1.predict(Xa_te) + ma2.predict(Xa_te) + ma3.predict(Xa_te)) / 3
        else:
            oof_ats_home[ts:te] = ma1.predict(Xa_te)

        ma1a = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        ma1a.fit(Xa_tr, away_runs)
        if HAS_LGBM:
            ma2a = LGBMRegressor(max_depth=4, n_estimators=100, random_state=SEED, verbose=-1)
            ma2a.fit(Xa_tr, away_runs)
            ma3a = CatBoostRegressor(depth=4, iterations=100, random_seed=SEED, verbose=0)
            ma3a.fit(Xa_tr, away_runs)
            oof_ats_away[ts:te] = (ma1a.predict(Xa_te) + ma2a.predict(Xa_te) + ma3a.predict(Xa_te)) / 3
        else:
            oof_ats_away[ts:te] = ma1a.predict(Xa_te)

        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{n_folds}")

    oof_ats_total = oof_ats_home + oof_ats_away
    oof_ats_edge = oof_ats_total - mkt

    # Monthly breakdown
    valid = ~np.isnan(oof_res)

    print(f"\n  {'Month':>8} {'Games':>7} {'UNDER':>15} {'OVER':>15} {'Res MAE':>8}")
    print(f"  {'─'*8} {'─'*7} {'─'*15} {'─'*15} {'─'*8}")

    month_names = {3: "March", 4: "April", 5: "May", 6: "June",
                   7: "July", 8: "August", 9: "Sept", 10: "Oct"}

    for m in [3, 4, 5, 6, 7, 8, 9, 10]:
        mask = valid & (months == m)
        if mask.sum() < 30:
            continue

        res = oof_res[mask]
        cls = oof_cls[mask]
        ats_e = oof_ats_edge[mask]
        act = actual[mask]
        mk = mkt[mask]
        act_side = np.where(act > mk, "OVER", np.where(act < mk, "UNDER", "PUSH"))

        # UNDER picks: res <= -0.3 AND cls >= 0.52 AND ats_edge <= -0.5
        u_mask = (res <= -0.3) & (cls >= 0.52) & (ats_e <= -0.5)
        u_correct = (act_side[u_mask] == "UNDER").sum()
        u_total = (act_side[u_mask] != "PUSH").sum()
        u_pct = u_correct / u_total * 100 if u_total > 0 else 0

        # OVER picks: res >= 0.8 AND cls <= 0.48 AND ats_e >= 0.5
        o_mask = (res >= 0.8) & (cls <= 0.48) & (ats_e >= 0.5)
        o_correct = (act_side[o_mask] == "OVER").sum()
        o_total = (act_side[o_mask] != "PUSH").sum()
        o_pct = o_correct / o_total * 100 if o_total > 0 else 0

        mae = np.mean(np.abs(res - (act - mk)))

        name = month_names.get(m, str(m))
        u_str = f"{u_correct}/{u_total} ({u_pct:.0f}%)" if u_total > 0 else "—"
        o_str = f"{o_correct}/{o_total} ({o_pct:.0f}%)" if o_total > 0 else "—"
        print(f"  {name:>8} {mask.sum():>7} {u_str:>15} {o_str:>15} {mae:>7.3f}")

    # Early vs late
    print(f"\n  {'Period':>12} {'Games':>7} {'UNDER':>15} {'OVER':>15}")
    print(f"  {'─'*12} {'─'*7} {'─'*15} {'─'*15}")

    for label, month_range in [("Mar-Apr", [3, 4]), ("May-Jun", [5, 6]),
                                ("Jul-Aug", [7, 8]), ("Sep-Oct", [9, 10])]:
        mask = valid & np.isin(months, month_range)
        if mask.sum() < 30:
            continue

        res = oof_res[mask]; cls = oof_cls[mask]; ats_e = oof_ats_edge[mask]
        act = actual[mask]; mk = mkt[mask]
        act_side = np.where(act > mk, "OVER", np.where(act < mk, "UNDER", "PUSH"))

        u_mask = (res <= -0.3) & (cls >= 0.52) & (ats_e <= -0.5)
        u_correct = (act_side[u_mask] == "UNDER").sum()
        u_total = (act_side[u_mask] != "PUSH").sum()
        u_pct = u_correct / u_total * 100 if u_total > 0 else 0

        o_mask = (res >= 0.8) & (cls <= 0.48) & (ats_e >= 0.5)
        o_correct = (act_side[o_mask] == "OVER").sum()
        o_total = (act_side[o_mask] != "PUSH").sum()
        o_pct = o_correct / o_total * 100 if o_total > 0 else 0

        u_str = f"{u_correct}/{u_total} ({u_pct:.0f}%)" if u_total > 0 else "—"
        o_str = f"{o_correct}/{o_total} ({o_pct:.0f}%)" if o_total > 0 else "—"
        print(f"  {label:>12} {mask.sum():>7} {u_str:>15} {o_str:>15}")


if __name__ == "__main__":
    run_ats_monthly()
    run_ou_monthly()
    print(f"\n  Done.")
