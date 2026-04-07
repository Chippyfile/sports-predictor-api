#!/usr/bin/env python3
"""
mlb_ou_v2_production.py — MLB O/U v2 FINAL
============================================
Residual: Lasso α=0.2 on [market_total, sp_form_combined, real_temp_f, real_wind_out]
ATS:      Lasso α=0.1 on same features (home + away score models)
Training: 2015-2019 + 2022-2025 (2020 COVID + 2021 default-feature excluded)

Tiers (walk-forward validated on 16K+ games):
  1u: residual ≤ -0.3                      (~66%, ~1/day)
  2u: residual ≤ -0.3 + ATS edge ≤ -0.8   (~71%, ~1/2-3 days)
  3u: residual ≤ -0.8                      (~75%, ~1/5 days)

Usage:
    python3 mlb_ou_v2_production.py               # Validate
    python3 mlb_ou_v2_production.py --upload       # Train + upload to Supabase
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

SEED = 42
N_FOLDS = 30
RES_ALPHA = 0.2    # Tuned: higher regularization → sharper signal at tails
ATS_ALPHA = 0.1    # Tuned: best combo accuracy at tight ATS gate
DROP_SEASONS = [2020, 2021]  # 2020=COVID 60-game, 2021=default features only
FEATURE_COLS = ["market_total", "sp_form_combined", "real_temp_f", "real_wind_out"]
MODERN_TO_RETRO = {"LAA":"ANA","CWS":"CHA","CHC":"CHN","KC":"KCA","LAD":"LAN","NYY":"NYA","NYM":"NYN","SD":"SDN","SF":"SFN","STL":"SLN","TB":"TBA","WSH":"WAS"}


def ensure_sp_form(df):
    if "sp_form_combined" in df.columns and df["sp_form_combined"].notna().sum() > len(df) * 0.5:
        return df
    print("  Computing sp_form_combined...")
    df = df.sort_values("game_date").reset_index(drop=True)
    sc = next((c for c in ["home_starter_name", "home_starter"] if c in df.columns), None)
    sa = next((c for c in ["away_starter_name", "away_starter"] if c in df.columns), None)
    if sc:
        ph = defaultdict(list)
        df["home_sp_recent_era"] = np.nan
        df["away_sp_recent_era"] = np.nan
        for idx, row in df.iterrows():
            for side, col in [("home", sc), ("away", sa)]:
                opp = "actual_away_runs" if side == "home" else "actual_home_runs"
                p = str(row.get(col, "")).strip()
                if not p or p == "nan": continue
                h = ph.get(p, [])
                if h: df.at[idx, f"{side}_sp_recent_era"] = np.mean([r for _, r in h[-3:]])
                ra = row.get(opp)
                if pd.notna(ra): ph[p].append((row["game_date"], float(ra)))
        for side in ["home", "away"]:
            fip = pd.to_numeric(df.get(f"{side}_sp_fip", 4.25), errors="coerce").fillna(4.25)
            df[f"{side}_sp_form_delta"] = df[f"{side}_sp_recent_era"].fillna(fip) - fip
        df["sp_form_combined"] = df["home_sp_form_delta"].fillna(0) + df["away_sp_form_delta"].fillna(0)
    else:
        df["sp_form_combined"] = 0.0
    return df


def load_and_prepare():
    from mlb_retrain import load_data
    df = load_data()
    print(f"  Loaded: {len(df)} games")

    # Drop 2021 (default features hurt model)
    before = len(df)
    df = df[~df["season"].isin(DROP_SEASONS)].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} games from seasons {DROP_SEASONS}")

    # Backfill O/U
    odds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_odds_2014_2021.csv")
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path, usecols=["game_date", "home_team", "away_team", "market_ou_total"])
        odds["home_team"] = odds["home_team"].replace(MODERN_TO_RETRO)
        odds["away_team"] = odds["away_team"].replace(MODERN_TO_RETRO)
        odds = odds.rename(columns={"market_ou_total": "_ou"})
        df = df.merge(odds, on=["game_date", "home_team", "away_team"], how="left")
        if "market_ou_total" not in df.columns:
            df["market_ou_total"] = df["_ou"]
        else:
            df["market_ou_total"] = pd.to_numeric(df["market_ou_total"], errors="coerce")
            df["market_ou_total"] = df["market_ou_total"].where(df["market_ou_total"] > 0).fillna(df["_ou"])
        df = df.drop(columns=["_ou"], errors="ignore")
        ou_n = (pd.to_numeric(df["market_ou_total"], errors="coerce").fillna(0) > 0).sum()
        print(f"  O/U available: {ou_n} games")

    # Targets
    df["actual_total"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) + \
                         pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)
    df = df[(df["market_total"] > 0) & (df["actual_total"] > 0)].copy()

    # Features
    df = ensure_sp_form(df)
    for col, default in [("real_temp_f", 72.0), ("real_wind_out", 0)]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df["sp_form_combined"] = df["sp_form_combined"].fillna(0)
    print(f"  Final training set: {len(df)} games")
    return df


def eval_mask(mask, yu, actual, mkt, side="UNDER"):
    n = mask.sum()
    if n < 10: return None
    pushes = (actual[mask] == mkt[mask]).sum()
    dec = n - pushes
    if dec < 10: return None
    if side == "UNDER":
        cor = ((yu[mask] == 1) & (actual[mask] != mkt[mask])).sum()
    else:
        cor = ((yu[mask] == 0) & (actual[mask] != mkt[mask])).sum()
    acc = cor / dec
    roi = (acc * 0.909 - (1 - acc)) * 100
    per_day = dec / len(set(df["season"].unique())) / 180
    return {"acc": acc, "n": dec, "roi": roi, "per_day": per_day}


def main():
    global df  # for eval_mask per_day calc
    upload = "--upload" in sys.argv

    print("=" * 70)
    print(f"  MLB O/U v2 FINAL — Lasso res α={RES_ALPHA}, ATS α={ATS_ALPHA}")
    print(f"  Drop: {DROP_SEASONS}")
    print("=" * 70)

    df = load_and_prepare()

    mkt = df["market_total"].values
    actual = df["actual_total"].values
    yr = actual - mkt
    yu = (actual < mkt).astype(int)
    yh = pd.to_numeric(df["actual_home_runs"], errors="coerce").fillna(0).values
    ya = pd.to_numeric(df["actual_away_runs"], errors="coerce").fillna(0).values
    X = df[FEATURE_COLS].fillna(0).values
    n = len(X)

    print(f"\n  Features: {FEATURE_COLS}")
    print(f"  Residual α={RES_ALPHA}, ATS α={ATS_ALPHA}")
    print(f"  Games: {n}")
    print(f"  Market MAE: {mean_absolute_error(actual, mkt):.3f}")

    # ═══════════════════════════════════════════════════════════
    # WALK-FORWARD
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD ({N_FOLDS} folds)")
    print(f"{'='*70}")

    fs = n // (N_FOLDS + 2)
    mt = fs * 2
    oof_res = np.full(n, np.nan)
    oof_ats = np.full(n, np.nan)

    for fold in range(N_FOLDS):
        ts = mt + fold * fs
        te = min(ts + fs, n)
        if ts >= n: break
        sc = StandardScaler().fit(X[:ts])
        Xtr, Xte = sc.transform(X[:ts]), sc.transform(X[ts:te])

        m = Lasso(alpha=RES_ALPHA, max_iter=5000, random_state=SEED)
        m.fit(Xtr, yr[:ts])
        oof_res[ts:te] = m.predict(Xte)

        mh = Lasso(alpha=ATS_ALPHA, max_iter=5000, random_state=SEED)
        mh.fit(Xtr, yh[:ts])
        ma = Lasso(alpha=ATS_ALPHA, max_iter=5000, random_state=SEED)
        ma.fit(Xtr, ya[:ts])
        oof_ats[ts:te] = mh.predict(Xte) + ma.predict(Xte)

        if (fold + 1) % 10 == 0:
            print(f"    Fold {fold+1}/{N_FOLDS}")

    valid = ~np.isnan(oof_res)
    ats_edge = oof_ats - mkt
    res_mae = mean_absolute_error(yr[valid], oof_res[valid])
    mkt_mae = mean_absolute_error(actual[valid], mkt[valid])

    print(f"\n  Residual MAE: {res_mae:.3f} (market: {mkt_mae:.3f})")
    print(f"  Beats market: {'✅ YES' if res_mae < mkt_mae else '❌ NO'} by {mkt_mae - res_mae:.3f}")

    # ── Thresholds ──
    print(f"\n  UNDER — Residual only (Lasso α={RES_ALPHA}):")
    print(f"  {'Threshold':<14s} {'Games':>7s} {'Acc':>6s} {'ROI':>7s} {'Freq':>10s}")
    print(f"  {'-'*46}")
    for th in [-0.2, -0.3, -0.4, -0.5, -0.6, -0.8, -1.0]:
        r = eval_mask(valid & (oof_res <= th), yu, actual, mkt)
        if r: print(f"  res ≤ {th:>+4.1f}    {r['n']:>7d} {r['acc']:>5.1%} {r['roi']:>+6.1f}% {r['per_day']:>7.1f}/day")

    print(f"\n  UNDER — Residual + ATS (α={ATS_ALPHA}):")
    print(f"  {'Config':<30s} {'Games':>7s} {'Acc':>6s} {'ROI':>7s} {'Freq':>10s}")
    print(f"  {'-'*62}")
    for rth, ath in [(-0.3, -0.5), (-0.3, -0.8), (-0.3, -1.0), (-0.5, -0.5), (-0.5, -0.8), (-0.8, -0.5)]:
        r = eval_mask(valid & (oof_res <= rth) & (ats_edge <= ath), yu, actual, mkt)
        if r: print(f"  res≤{rth:+.1f} ats≤{ath:+.1f}      {r['n']:>7d} {r['acc']:>5.1%} {r['roi']:>+6.1f}% {r['per_day']:>7.1f}/day")

    print(f"\n  OVER — Residual only:")
    print(f"  {'Threshold':<14s} {'Games':>7s} {'Acc':>6s} {'ROI':>7s} {'Freq':>10s}")
    print(f"  {'-'*46}")
    for th in [0.3, 0.5, 0.8, 1.0]:
        r = eval_mask(valid & (oof_res >= th), yu, actual, mkt, side="OVER")
        if r: print(f"  res ≥ {th:>+4.1f}    {r['n']:>7d} {r['acc']:>5.1%} {r['roi']:>+6.1f}% {r['per_day']:>7.1f}/day")

    # Per-season
    if "season" in df.columns:
        print(f"\n  Per-season UNDER (res ≤ -0.3):")
        for s in sorted(df["season"].unique()):
            r = eval_mask(valid & (df["season"].values == s) & (oof_res <= -0.3), yu, actual, mkt)
            if r and r["n"] >= 5: print(f"    {s}: {r['acc']:.1%} on {r['n']} games")

    # ═══════════════════════════════════════════════════════════
    # PRODUCTION TRAINING
    # ═══════════════════════════════════════════════════════════
    if not upload:
        print(f"\n  Run with --upload to train + save to Supabase")
        return

    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING")
    print(f"{'='*70}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model_res = Lasso(alpha=RES_ALPHA, max_iter=5000, random_state=SEED)
    model_res.fit(Xs, yr)
    print(f"  Residual coefficients:")
    for f, c in zip(FEATURE_COLS, model_res.coef_):
        print(f"    {f:25s} {c:+.4f}")
    print(f"    {'intercept':25s} {model_res.intercept_:+.4f}")

    model_h = Lasso(alpha=ATS_ALPHA, max_iter=5000, random_state=SEED)
    model_h.fit(Xs, yh)
    model_a = Lasso(alpha=ATS_ALPHA, max_iter=5000, random_state=SEED)
    model_a.fit(Xs, ya)
    print(f"  ATS total MAE: {mean_absolute_error(actual, model_h.predict(Xs) + model_a.predict(Xs)):.3f}")

    bias = float(np.mean(model_res.predict(Xs) - yr))
    print(f"  Bias: {bias:+.4f}")

    bundle = {
        "model_res": model_res,
        "model_ats_home": model_h,
        "model_ats_away": model_a,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "under_thresholds": {1: -0.3, 2: -0.3, 3: -0.8},  # residual thresholds per tier
        "ats_threshold": -0.8,  # ATS gate for 2u (requires BOTH res≤-0.3 AND ats≤-0.8)
        "over_thresholds": {1: 0.5},
        "model_type": "mlb_ou_v2_sp_form",
        "architecture": f"Residual(Lasso α={RES_ALPHA}) + ATS(Lasso α={ATS_ALPHA}×2)",
        "n_train": len(X),
        "mae_cv": round(res_mae, 4),
        "market_mae": round(mkt_mae, 4),
        "bias_correction": round(bias, 4),
        "res_alpha": RES_ALPHA,
        "ats_alpha": ATS_ALPHA,
        "drop_seasons": DROP_SEASONS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "_v2_sp_form": True,
    }

    import joblib
    lp = "mlb_ou_v2.pkl"
    joblib.dump(bundle, lp)
    print(f"\n  Saved: {lp} ({os.path.getsize(lp) / 1024:.0f} KB)")

    try:
        from db import save_model
        save_model("mlb_ou", bundle)
        print(f"  ✅ Uploaded as 'mlb_ou'")
        from db import load_model
        ck = load_model("mlb_ou")
        if ck and ck.get("_v2_sp_form"):
            print(f"  ✅ Verified: {ck.get('model_type')}, α_res={ck.get('res_alpha')}, α_ats={ck.get('ats_alpha')}")
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")

    print(f"\n{'='*70}")
    print(f"  COMPLETE — MLB O/U v2")
    print(f"{'='*70}")
    print(f"  Residual: Lasso α={RES_ALPHA}")
    print(f"  ATS:      Lasso α={ATS_ALPHA}")
    print(f"  Seasons:  {[s for s in sorted(df['season'].unique())]}")
    print(f"  UNDER 1u: res ≤ -0.3          (~66%, ~1/day)")
    print(f"  UNDER 2u: res ≤ -0.3 + ATS ≤ -0.8  (~71%, ~1/2-3 days)")
    print(f"  UNDER 3u: res ≤ -0.8          (~75%, ~1/5 days)")


if __name__ == "__main__":
    main()
