#!/usr/bin/env python3
"""
mlb_retrain_ou_v3.py — MLB Over/Under v3: Triple Agreement + Lineup Features
═══════════════════════════════════════════════════════════════════════════════
v3 adds lineup_delta_sum (r=-0.164), lineup_total_top3 (r=-0.111), and ump_career_rpg (r=+0.126).
These capture lineup CHANGE signal (rested starters) that the market misses.

Usage:
    python3 mlb_retrain_ou_v3.py                # Evaluate all configs
    python3 mlb_retrain_ou_v3.py --upload       # Train best + upload
    python3 mlb_retrain_ou_v3.py --refresh      # Re-pull data first
"""
import sys, os, time, warnings, pickle, io, base64
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

SEED = 42
N_FOLDS = 30

# ═══════════════════════════════════════════════════════════
# FEATURE SETS — MLB-specific, purpose-built per component
# ═══════════════════════════════════════════════════════════

# Residual: detect Vegas O/U mispricing. Environment + bullpen + ump + lineup change.
RES_FEATURE_COLS = [
    "park_factor", "temp_f", "wind_mph", "wind_out",
    "is_warm", "is_cold", "temp_x_park",
    "fip_combined", "bullpen_combined", "k_bb_combined",
    "sp_ip_combined", "bp_exposure_combined",
    "woba_combined", "woba_diff",
    "ump_run_env",
    "scoring_entropy_combined", "first_inn_rate_combined",
    "rest_combined", "series_game_num", "lg_rpg",
    # v3: Lineup features (market misses lineup CHANGE)
    "lineup_delta_sum", "lineup_total_top3", "lineup_total_woba",
    "ump_career_rpg", "ump_career_bb",
]

# Classifier: predict direction. market_total is the strongest anchor.
CLS_FEATURE_COLS = [
    "market_total",
    "park_factor", "temp_f", "wind_out", "temp_x_park",
    "fip_combined", "bullpen_combined",
    "woba_combined", "ump_run_env",
    "scoring_entropy_combined",
    "is_warm", "is_cold",
    "sp_ip_combined", "bp_exposure_combined",
    "rest_combined", "k_bb_combined", "lg_rpg",
    # v3: Lineup features
    "lineup_delta_sum", "lineup_total_top3", "lineup_total_woba",
    "ump_career_rpg", "ump_career_bb",
]

# ATS scores: predict actual runs per team. Pitcher-matchup heavy.
ATS_FEATURE_COLS = [
    "woba_diff", "fip_diff", "fip_combined",
    "k_bb_combined", "bullpen_combined",
    "sp_ip_combined", "bp_exposure_combined",
    "park_factor", "temp_f", "wind_out", "temp_x_park",
    "ump_run_env", "rest_combined", "series_game_num",
    "woba_combined",
    "scoring_entropy_combined", "first_inn_rate_combined",
    "market_total", "lg_rpg",
    # v3: Lineup features
    "lineup_delta_sum", "lineup_total_top3", "lineup_total_woba",
    "ump_career_rpg", "ump_career_bb",
]

# ═══════════════════════════════════════════════════════════
# MODEL CONFIGS — 3 architectures per component
# ═══════════════════════════════════════════════════════════

RES_CONFIGS = {
    "linear": [
        ("Lasso_0.05", lambda: Lasso(alpha=0.05, max_iter=5000, random_state=SEED)),
        ("ElasticNet_0.1", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=SEED)),
        ("Ridge_1.0", lambda: Ridge(alpha=1.0, random_state=SEED)),
    ],
    "tree": [
        ("CatBoost_d3_200", lambda: CatBoostRegressor(iterations=200, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)),
        ("LGBM_d3_200", lambda: LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
        ("CatBoost_d5_300", lambda: CatBoostRegressor(iterations=300, depth=5, learning_rate=0.03, random_seed=SEED, verbose=0)),
    ],
    "mixed": [
        ("Lasso_0.1", lambda: Lasso(alpha=0.1, max_iter=5000, random_state=SEED)),
        ("LGBM_d3_200", lambda: LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
        ("CatBoost_d3_200", lambda: CatBoostRegressor(iterations=200, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)),
    ],
}

CLS_CONFIGS = {
    "linear": [
        ("LogReg_C0.5", lambda: LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)),
        ("LogReg_C0.1", lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)),
    ],
    "tree": [
        ("LGBM_d3", lambda: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
        ("CatBoost_d3", lambda: CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)),
    ],
    "mixed": [
        ("LogReg_C0.5", lambda: LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)),
        ("LGBM_d3", lambda: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
    ],
}

ATS_CONFIGS = {
    "linear": [
        ("Lasso_0.1", lambda: Lasso(alpha=0.1, max_iter=5000, random_state=SEED)),
        ("ElasticNet_0.1", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=SEED)),
    ],
    "tree": [
        ("CatBoost_d4", lambda: CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0)),
        ("LGBM_d4", lambda: LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
    ],
    "mixed": [
        ("Lasso_0.1", lambda: Lasso(alpha=0.1, max_iter=5000, random_state=SEED)),
        ("LGBM_d4", lambda: LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, verbose=-1, random_state=SEED)),
        ("CatBoost_d4", lambda: CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0)),
    ],
}

# Imports
from mlb_retrain import load_data
from mlb_ou_retrain import build_ou_features


def build_all_features(df):
    ou_df = build_ou_features(df)
    if "woba_diff" not in ou_df.columns:
        ou_df["woba_diff"] = pd.to_numeric(df.get("home_woba", 0.315), errors="coerce").fillna(0.315) - \
                              pd.to_numeric(df.get("away_woba", 0.315), errors="coerce").fillna(0.315)
    if "fip_diff" not in ou_df.columns:
        ou_df["fip_diff"] = pd.to_numeric(df.get("home_sp_fip", 4.25), errors="coerce").fillna(4.25) - \
                             pd.to_numeric(df.get("away_sp_fip", 4.25), errors="coerce").fillna(4.25)

    # ── v3: Merge lineup features from parquet ──
    lineup_path = "mlb_lineup_features_advanced.parquet"
    if os.path.exists(lineup_path) and "lineup_delta_sum" not in ou_df.columns:
        lf = pd.read_parquet(lineup_path)
        lf["_key"] = lf["game_date"].astype(str).str[:10] + "|" + lf["home_abbr"]
        ou_df["_key"] = (df["game_date"].astype(str).str[:10] + "|" + df["home_team"]).values
        lineup_cols = ["lineup_delta_sum", "lineup_total_top3", "lineup_total_woba"]
        available = [c for c in lineup_cols if c in lf.columns]
        if available:
            lf_dedup = lf.drop_duplicates(subset="_key", keep="last")[["_key"] + available]
            n_before = len(ou_df)
            ou_df = ou_df.merge(lf_dedup, on="_key", how="left")
            assert len(ou_df) == n_before, f"Merge changed row count: {n_before} -> {len(ou_df)}"
            for c in available:
                ou_df[c] = ou_df[c].fillna(0)
            matched = (ou_df["lineup_delta_sum"] != 0).sum()
            print(f"  [v3] Merged lineup features: {matched}/{len(ou_df)} games ({matched/len(ou_df)*100:.0f}%)")
        ou_df.drop(columns=["_key"], inplace=True, errors="ignore")
    else:
        for c in ["lineup_delta_sum", "lineup_total_top3", "lineup_total_woba"]:
            if c not in ou_df.columns:
                ou_df[c] = 0

    # ── v3: Merge umpire career RPG from parquet ──
    ump_path = "mlb_ump_game_data_2022_2025.parquet"
    if os.path.exists(ump_path) and "ump_career_rpg" not in ou_df.columns:
        ump = pd.read_parquet(ump_path)
        # Compute career avg RPG per umpire (30+ game minimum)
        career = ump.groupby("hp_ump").agg(games=("game_pk", "count"), avg_rpg=("total", "mean"), avg_bb=("total_bb", "mean")).query("games >= 30")
        # Training data uses retrosheet codes, ump parquet uses modern
        RETRO_TO_MODERN = {
            "ANA": "LAA", "CHA": "CWS", "CHN": "CHC", "KCA": "KC",
            "LAN": "LAD", "NYA": "NYY", "NYN": "NYM", "SDN": "SD",
            "SFN": "SF", "SLN": "STL", "TBA": "TB", "WAS": "WSH",
        }
        home_modern = df["home_team"].replace(RETRO_TO_MODERN)
        ump["_key"] = ump["game_date"].astype(str).str[:10] + "|" + ump["home"]
        ou_df["_key"] = (df["game_date"].astype(str).str[:10] + "|" + home_modern).values
        ump_game = ump[["_key", "hp_ump"]].drop_duplicates("_key", keep="last")
        n_before = len(ou_df)
        ou_df = ou_df.merge(ump_game, on="_key", how="left")
        assert len(ou_df) == n_before, f"Ump merge changed row count: {n_before} -> {len(ou_df)}"
        ou_df["ump_career_rpg"] = ou_df["hp_ump"].map(career["avg_rpg"]).fillna(8.5)
        ou_df["ump_career_bb"] = ou_df["hp_ump"].map(career["avg_bb"]).fillna(6.5)
        matched = (ou_df["ump_career_rpg"] != 8.5).sum()
        print(f"  [v3] Merged ump career RPG: {matched}/{len(ou_df)} games ({matched/len(ou_df)*100:.0f}%)")
        ou_df.drop(columns=["_key", "hp_ump"], inplace=True, errors="ignore")
    else:
        if "ump_career_rpg" not in ou_df.columns:
            ou_df["ump_career_rpg"] = 8.5
        if "ump_career_bb" not in ou_df.columns:
            ou_df["ump_career_bb"] = 6.5

    return ou_df


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD ENGINES
# ═══════════════════════════════════════════════════════════

def wf_regressor(X, y, w, builders, min_train=300):
    n = len(X)
    fold_size = n // N_FOLDS
    oof = np.full((len(builders), n), np.nan)
    for fold in range(N_FOLDS):
        ts = fold * fold_size
        te = min((fold + 1) * fold_size, n) if fold < N_FOLDS - 1 else n
        if ts < min_train: continue
        scl = StandardScaler().fit(X[:ts])
        Xtr, Xte = scl.transform(X[:ts]), scl.transform(X[ts:te])
        for i, (_, b) in enumerate(builders):
            m = b()
            try: m.fit(Xtr, y[:ts], sample_weight=w[:ts])
            except TypeError: m.fit(Xtr, y[:ts])
            oof[i, ts:te] = m.predict(Xte)
    return oof


def wf_classifier(X, y, w, builders, min_train=300):
    n = len(X)
    fold_size = n // N_FOLDS
    oof = np.full((len(builders), n), np.nan)
    for fold in range(N_FOLDS):
        ts = fold * fold_size
        te = min((fold + 1) * fold_size, n) if fold < N_FOLDS - 1 else n
        if ts < min_train: continue
        scl = StandardScaler().fit(X[:ts])
        Xtr, Xte = scl.transform(X[:ts]), scl.transform(X[ts:te])
        for i, (_, b) in enumerate(builders):
            m = b()
            try: m.fit(Xtr, y[:ts], sample_weight=w[:ts])
            except TypeError: m.fit(Xtr, y[:ts])
            oof[i, ts:te] = m.predict_proba(Xte)[:, 1]
    return oof


def wf_ats_scores(X, yh, ya, w, builders, min_train=300):
    n = len(X)
    fold_size = n // N_FOLDS
    oof = np.full(n, np.nan)
    for fold in range(N_FOLDS):
        ts = fold * fold_size
        te = min((fold + 1) * fold_size, n) if fold < N_FOLDS - 1 else n
        if ts < min_train: continue
        scl = StandardScaler().fit(X[:ts])
        Xtr, Xte = scl.transform(X[:ts]), scl.transform(X[ts:te])
        hp, ap = [], []
        for _, b in builders:
            mh = b()
            try: mh.fit(Xtr, yh[:ts], sample_weight=w[:ts])
            except TypeError: mh.fit(Xtr, yh[:ts])
            hp.append(mh.predict(Xte))
            ma = b()
            try: ma.fit(Xtr, ya[:ts], sample_weight=w[:ts])
            except TypeError: ma.fit(Xtr, ya[:ts])
            ap.append(ma.predict(Xte))
        oof[ts:te] = np.mean(hp, axis=0) + np.mean(ap, axis=0)
    return oof


# ═══════════════════════════════════════════════════════════
# THRESHOLD SEARCH
# ═══════════════════════════════════════════════════════════

def search_thresholds(res_avg, cls_avg, ats_edge, y_under, mkt, actual, label=""):
    print(f"\n  {'='*65}")
    print(f"  {label}")
    print(f"  {'='*65}")
    valid = ~np.isnan(res_avg) & (mkt > 0)
    best_u, best_o = {}, {}

    for res_th in np.arange(-0.2, -2.1, -0.1):
        for cls_th in np.arange(0.50, 0.65, 0.02):
            for ats_th in np.arange(-0.3, -2.5, -0.2):
                mask = valid & (res_avg <= res_th) & (cls_avg >= cls_th) & (ats_edge <= ats_th)
                n = mask.sum()
                if n < 25: continue
                decided = n - (actual[mask] == mkt[mask]).sum()
                if decided < 20: continue
                correct = ((y_under[mask] == 1) & (actual[mask] != mkt[mask])).sum()
                acc = correct / decided
                roi = (acc * 0.909 - (1 - acc)) * 100
                tier = 3 if acc >= 0.63 else (2 if acc >= 0.59 else (1 if acc >= 0.55 else 0))
                if tier > 0:
                    prev = best_u.get(tier)
                    if prev is None or acc > prev["acc"] or (acc == prev["acc"] and decided > prev["n"]):
                        best_u[tier] = {"res_avg": round(float(res_th), 2), "cls_avg": round(float(cls_th), 2),
                                        "ats_edge": round(float(ats_th), 2), "acc": round(float(acc), 4),
                                        "n": int(decided), "roi": round(float(roi), 1)}

    for res_th in np.arange(0.2, 2.1, 0.1):
        for cls_th in np.arange(0.50, 0.35, -0.02):
            for ats_th in np.arange(0.3, 2.5, 0.2):
                mask = valid & (res_avg >= res_th) & (cls_avg <= cls_th) & (ats_edge >= ats_th)
                n = mask.sum()
                if n < 15: continue
                decided = n - (actual[mask] == mkt[mask]).sum()
                if decided < 12: continue
                correct = ((y_under[mask] == 0) & (actual[mask] != mkt[mask])).sum()
                acc = correct / decided
                roi = (acc * 0.909 - (1 - acc)) * 100
                tier = 2 if acc >= 0.61 else (1 if acc >= 0.56 else 0)
                if tier > 0:
                    prev = best_o.get(tier)
                    if prev is None or acc > prev["acc"] or (acc == prev["acc"] and decided > prev["n"]):
                        best_o[tier] = {"res_avg": round(float(res_th), 2), "cls_avg": round(float(cls_th), 2),
                                        "ats_edge": round(float(ats_th), 2), "acc": round(float(acc), 4),
                                        "n": int(decided), "roi": round(float(roi), 1)}

    print(f"  {'Tier':<8} {'Res':<8} {'Cls':<8} {'ATS':<8} {'Games':>6} {'Acc':>7} {'ROI':>7}")
    print(f"  {'-'*55}")
    for t in sorted(best_u): v=best_u[t]; print(f"  U-{t}u   {v['res_avg']:>+5.1f}   {v['cls_avg']:>5.2f}   {v['ats_edge']:>+5.1f}   {v['n']:>6d} {v['acc']*100:>6.1f}% {v['roi']:>+6.1f}%")
    for t in sorted(best_o): v=best_o[t]; print(f"  O-{t}u   {v['res_avg']:>+5.1f}   {v['cls_avg']:>5.2f}   {v['ats_edge']:>+5.1f}   {v['n']:>6d} {v['acc']*100:>6.1f}% {v['roi']:>+6.1f}%")
    if not best_u and not best_o: print("  ⚠ No profitable tiers found")
    return best_u, best_o


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    upload = "--upload" in sys.argv
    print("=" * 70)
    print("  MLB O/U v2 — TRIPLE AGREEMENT ARCHITECTURE SEARCH")
    print("=" * 70)

    df = load_data(refresh="--refresh" in sys.argv)
    print(f"  Raw: {len(df)}")

    # ── Backfill market O/U from historical odds files (2014-2021) ──
    odds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_odds_2014_2021.csv")
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path, usecols=["game_date", "home_team", "away_team", "market_ou_total",
                                                 "market_spread_home", "market_home_ml", "market_away_ml"])
        # Training data uses retrosheet abbreviations — map odds to match
        MODERN_TO_RETRO = {
            "LAA": "ANA", "CWS": "CHA", "CHC": "CHN", "KC": "KCA",
            "LAD": "LAN", "NYY": "NYA", "NYM": "NYN", "SD": "SDN",
            "SF": "SFN", "STL": "SLN", "TB": "TBA", "WSH": "WAS",
        }
        odds["home_team"] = odds["home_team"].replace(MODERN_TO_RETRO)
        odds["away_team"] = odds["away_team"].replace(MODERN_TO_RETRO)
        odds = odds.rename(columns={"market_ou_total": "_odds_ou", "market_spread_home": "_odds_spread",
                                     "market_home_ml": "_odds_hml", "market_away_ml": "_odds_aml"})
        before_ou = (pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0) > 0).sum()
        df = df.merge(odds, on=["game_date", "home_team", "away_team"], how="left")
        # Fill missing market data from odds file
        for col, odds_col in [("market_ou_total", "_odds_ou"), ("market_spread_home", "_odds_spread"),
                               ("market_home_ml", "_odds_hml"), ("market_away_ml", "_odds_aml")]:
            if col not in df.columns:
                df[col] = df[odds_col]
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].where(df[col] > 0, other=None).fillna(df[odds_col])
        df = df.drop(columns=["_odds_ou", "_odds_spread", "_odds_hml", "_odds_aml"], errors="ignore")
        after_ou = (pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0) > 0).sum()
        print(f"  O/U backfill: {before_ou} → {after_ou} games (+{after_ou - before_ou})")
    else:
        print(f"  ⚠ No odds file found at {odds_path} — using Supabase data only")

    df["actual_total"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) + \
                         pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    df["market_total_raw"] = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)
    df = df[(df["market_total_raw"] > 0) & (df["actual_total"] > 0)].copy()
    print(f"  With market O/U + scores: {len(df)}")
    if len(df) < 500: print("  ❌ Need 500+"); return

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

    print(f"  Features: res={len(res_f)} cls={len(cls_f)} ats={len(ats_f)}")
    print(f"  Residual target: mean={yr.mean():.3f} std={yr.std():.3f}")
    print(f"  Under rate: {yu.mean():.3f}")
    print(f"  Market MAE: {mean_absolute_error(actual, mkt):.3f}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1-3: ARCHITECTURE COMPARISON
    # ═══════════════════════════════════════════════════════════
    res_oof_all, cls_oof_all, ats_oof_all = {}, {}, {}

    print(f"\n{'='*70}\n  RESIDUAL MODELS\n{'='*70}")
    for name, builders in RES_CONFIGS.items():
        print(f"  >>> {name.upper()}: {[n for n,_ in builders]}")
        oof = wf_regressor(Xr, yr, w, builders)
        v = ~np.isnan(oof[0]); avg = np.mean(oof, axis=0)
        mae = mean_absolute_error(yr[v], avg[v])
        dir_ok = ((avg[v] > 0) == (yr[v] > 0)); dir_acc = dir_ok[yr[v] != 0].mean()
        print(f"    MAE: {mae:.3f} | Dir acc: {dir_acc:.3f}")
        res_oof_all[name] = {"mae": mae, "dir": dir_acc, "oof": oof}
    best_r = min(res_oof_all, key=lambda k: res_oof_all[k]["mae"])
    print(f"  ★ Best: {best_r} (MAE={res_oof_all[best_r]['mae']:.3f})")

    print(f"\n{'='*70}\n  CLASSIFIERS\n{'='*70}")
    for name, builders in CLS_CONFIGS.items():
        print(f"  >>> {name.upper()}: {[n for n,_ in builders]}")
        oof = wf_classifier(Xc, yu, w, builders)
        v = ~np.isnan(oof[0]); avg = np.mean(oof, axis=0)
        brier = np.mean((avg[v] - yu[v]) ** 2)
        clf_acc = ((avg[v] > 0.5) == yu[v].astype(bool)).mean()
        print(f"    Brier: {brier:.4f} | Acc: {clf_acc:.3f}")
        cls_oof_all[name] = {"brier": brier, "acc": clf_acc, "oof": oof}
    best_c = min(cls_oof_all, key=lambda k: cls_oof_all[k]["brier"])
    print(f"  ★ Best: {best_c} (Brier={cls_oof_all[best_c]['brier']:.4f})")

    print(f"\n{'='*70}\n  ATS SCORE MODELS\n{'='*70}")
    for name, builders in ATS_CONFIGS.items():
        print(f"  >>> {name.upper()}: {[n for n,_ in builders]}")
        oof = wf_ats_scores(Xa, yh, ya, w, builders)
        v = ~np.isnan(oof); mae = mean_absolute_error(actual[v], oof[v])
        print(f"    Total MAE: {mae:.3f} (mkt: {mean_absolute_error(actual[v], mkt[v]):.3f})")
        ats_oof_all[name] = {"mae": mae, "oof": oof}
    best_a = min(ats_oof_all, key=lambda k: ats_oof_all[k]["mae"])
    print(f"  ★ Best: {best_a} (MAE={ats_oof_all[best_a]['mae']:.3f})")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: TRIPLE AGREEMENT — best per-component
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  TRIPLE AGREEMENT: {best_r}/{best_c}/{best_a}\n{'='*70}")
    r_avg = np.mean(res_oof_all[best_r]["oof"], axis=0)
    c_avg = np.mean(cls_oof_all[best_c]["oof"], axis=0)
    a_edge = ats_oof_all[best_a]["oof"] - mkt
    best_under, best_over = search_thresholds(r_avg, c_avg, a_edge, yu, mkt, actual, "BEST PER-COMPONENT")

    # ── Also test all 27 combos ──
    print(f"\n{'='*70}\n  CROSS-CONFIG SEARCH (all combos)\n{'='*70}")
    combo_scores = []
    for rn in res_oof_all:
        for cn in cls_oof_all:
            for an in ats_oof_all:
                ra = np.mean(res_oof_all[rn]["oof"], axis=0)
                ca = np.mean(cls_oof_all[cn]["oof"], axis=0)
                ae = ats_oof_all[an]["oof"] - mkt
                v = ~np.isnan(ra) & ~np.isnan(ca) & ~np.isnan(ae) & (mkt > 0)
                # Quick loose UNDER check
                m = v & (ra <= -0.3) & (ca >= 0.52) & (ae <= -0.5)
                n = m.sum()
                if n < 20: continue
                dec = n - (actual[m] == mkt[m]).sum()
                if dec < 15: continue
                cor = ((yu[m] == 1) & (actual[m] != mkt[m])).sum()
                acc = cor / dec
                combo_scores.append((rn, cn, an, acc, dec))
    combo_scores.sort(key=lambda x: -x[3])
    print(f"  Top 5 combos (loose UNDER):")
    for rn, cn, an, acc, n in combo_scores[:5]:
        marker = " ★" if (rn, cn, an) == (best_r, best_c, best_a) else ""
        print(f"    {rn:>6s}/{cn:>6s}/{an:>6s}: {acc:.1%} on {n}{marker}")

    # If top combo differs and is better, search its thresholds too
    if combo_scores and combo_scores[0][:3] != (best_r, best_c, best_a) and combo_scores[0][3] > 0.55:
        top = combo_scores[0]
        print(f"\n  Testing cross-config winner: {top[0]}/{top[1]}/{top[2]}...")
        ra2 = np.mean(res_oof_all[top[0]]["oof"], axis=0)
        ca2 = np.mean(cls_oof_all[top[1]]["oof"], axis=0)
        ae2 = ats_oof_all[top[2]]["oof"] - mkt
        bu2, bo2 = search_thresholds(ra2, ca2, ae2, yu, mkt, actual, f"CROSS: {top[0]}/{top[1]}/{top[2]}")
        # Compare total weighted accuracy
        score_orig = sum(v["acc"] * v["n"] for v in best_under.values()) + sum(v["acc"] * v["n"] for v in best_over.values())
        score_cross = sum(v["acc"] * v["n"] for v in bu2.values()) + sum(v["acc"] * v["n"] for v in bo2.values())
        if score_cross > score_orig:
            print(f"  ★ Cross-config wins! Using {top[0]}/{top[1]}/{top[2]}")
            best_r, best_c, best_a = top[0], top[1], top[2]
            best_under, best_over = bu2, bo2

    if not best_under and not best_over:
        print("\n  ⚠ No profitable tiers. Try different features or more data."); return

    # ═══════════════════════════════════════════════════════════
    # SUMMARY / PRODUCTION TRAINING
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  FINAL ARCHITECTURE\n{'='*70}")
    print(f"  Residual: {best_r} ({[n for n,_ in RES_CONFIGS[best_r]]})")
    print(f"  Classifier: {best_c} ({[n for n,_ in CLS_CONFIGS[best_c]]})")
    print(f"  ATS scores: {best_a} ({[n for n,_ in ATS_CONFIGS[best_a]]})")
    print(f"  Under tiers: {list(best_under.keys())}")
    print(f"  Over tiers: {list(best_over.keys())}")
    for t in sorted(best_under): v=best_under[t]; print(f"    U-{t}u: {v['acc']*100:.1f}% on {v['n']} games, ROI {v['roi']:+.1f}%")
    for t in sorted(best_over): v=best_over[t]; print(f"    O-{t}u: {v['acc']*100:.1f}% on {v['n']} games, ROI {v['roi']:+.1f}%")

    if not upload:
        print(f"\n  Run with --upload to train + save to Supabase"); return

    print(f"\n  Training production models...")
    rs = StandardScaler().fit(Xr); cs = StandardScaler().fit(Xc); als = StandardScaler().fit(Xa)
    Xr_s, Xc_s, Xa_s = rs.transform(Xr), cs.transform(Xc), als.transform(Xa)

    res_models = []
    for _, b in RES_CONFIGS[best_r]:
        m = b()
        try: m.fit(Xr_s, yr, sample_weight=w)
        except TypeError: m.fit(Xr_s, yr)
        res_models.append(m)

    cls_models = []
    for _, b in CLS_CONFIGS[best_c]:
        m = b()
        try: m.fit(Xc_s, yu, sample_weight=w)
        except TypeError: m.fit(Xc_s, yu)
        cls_models.append(m)

    ats_h, ats_a = [], []
    for _, b in ATS_CONFIGS[best_a]:
        mh = b()
        try: mh.fit(Xa_s, yh, sample_weight=w)
        except TypeError: mh.fit(Xa_s, yh)
        ats_h.append(mh)
        ma = b()
        try: ma.fit(Xa_s, ya, sample_weight=w)
        except TypeError: ma.fit(Xa_s, ya)
        ats_a.append(ma)

    bundle = {
        "res_scaler": rs, "cls_scaler": cs, "ats_scaler": als,
        "res_models": res_models, "cls_models": cls_models,
        "ats_home_models": ats_h, "ats_away_models": ats_a,
        "res_feature_cols": res_f, "cls_feature_cols": cls_f, "ats_feature_cols": ats_f,
        "under_tiers": best_under, "over_tiers": best_over,
        "model_type": "mlb_ou_v3_triple_lineup",
        "architecture": {"residual": best_r, "classifier": best_c, "ats_scores": best_a,
                         "res_models": [n for n,_ in RES_CONFIGS[best_r]],
                         "cls_models": [n for n,_ in CLS_CONFIGS[best_c]],
                         "ats_models": [n for n,_ in ATS_CONFIGS[best_a]]},
        "n_train": len(df), "mae_cv": round(res_oof_all[best_r]["mae"], 3),
        "ats_mae": round(ats_oof_all[best_a]["mae"], 3),
        "market_mae": round(mean_absolute_error(actual, mkt), 3),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "_v3_lineup": True,
    }

    buf = io.BytesIO(); pickle.dump(bundle, buf)
    blob = base64.b64encode(buf.getvalue()).decode()
    print(f"  Bundle: {len(buf.getvalue())/1024:.0f} KB")

    import requests
    SB_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
    SB_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not SB_KEY: print("  ❌ No SUPABASE_KEY"); return
    h = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}",
         "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
    resp = requests.post(f"{SB_URL}/rest/v1/model_store", headers=h, json={
        "name": "mlb_ou", "data": blob,
        "metadata": {"model_type": bundle["model_type"], "architecture": bundle["architecture"],
                     "n_train": len(df), "mae_cv": bundle["mae_cv"], "trained_at": bundle["trained_at"],
                     "under_tiers": {str(k): v for k, v in best_under.items()},
                     "over_tiers": {str(k): v for k, v in best_over.items()}}
    }, timeout=60)
    print(f"  {'✅ Uploaded' if resp.ok else '❌ Failed: ' + resp.text[:200]}")


if __name__ == "__main__":
    main()
