#!/usr/bin/env python3
"""
nba_ou_train_v2.py — NBA Over/Under total prediction model
===========================================================
Dedicated O/U feature set: COMBINED features (sums/averages, not diffs).
Totals don't care which team scores more — they care about total scoring environment.

Architecture: CatBoost + Ridge blend
Target: actual_home_score + actual_away_score

Usage:
    python3 nba_ou_train_v2.py              # Train + evaluate
    python3 nba_ou_train_v2.py --upload     # Train + upload to Supabase as 'nba_ou'
"""
import sys, os, time, warnings, argparse
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    from sklearn.ensemble import GradientBoostingRegressor

from nba_build_features_v27 import load_training_data


def _sc(df, col, default=0):
    return pd.to_numeric(df.get(col, default), errors="coerce").fillna(default)


def build_ou_features(df):
    """Build O/U features: COMBINED (sums/averages) not diffs."""
    f = pd.DataFrame(index=df.index)

    # ── MARKET ──
    f["market_total"] = _sc(df, "market_ou_total", 220)
    f["overround"] = _sc(df, "overround", 0.04)

    # ── SCORING ENVIRONMENT (combined) ──
    h_ppg = _sc(df, "home_ppg", 112); a_ppg = _sc(df, "away_ppg", 112)
    f["ppg_combined"] = h_ppg + a_ppg
    f["opp_ppg_combined"] = _sc(df, "home_opp_ppg", 112) + _sc(df, "away_opp_ppg", 112)
    f["net_rtg_sum"] = _sc(df, "home_net_rtg", 0) + _sc(df, "away_net_rtg", 0)
    f["ou_gap"] = f["ppg_combined"] - f["market_total"]

    # ── PACE ──
    h_tempo = _sc(df, "home_tempo", 100); a_tempo = _sc(df, "away_tempo", 100)
    f["tempo_combined"] = h_tempo + a_tempo
    f["pace_min"] = np.minimum(h_tempo, a_tempo)
    f["ppg_x_pace"] = (f["ppg_combined"] * f["tempo_combined"]) / 20000

    # ── SHOOTING (averages) ──
    h_fg = _sc(df, "home_fgpct", 0.46); a_fg = _sc(df, "away_fgpct", 0.46)
    h_3p = _sc(df, "home_threepct", 0.36); a_3p = _sc(df, "away_threepct", 0.36)
    h_ft = _sc(df, "home_ftpct", 0.77); a_ft = _sc(df, "away_ftpct", 0.77)
    f["fgpct_avg"] = (h_fg + a_fg) / 2
    f["threepct_avg"] = (h_3p + a_3p) / 2
    f["ftpct_avg"] = (h_ft + a_ft) / 2
    f["efg_avg"] = ((h_fg + 0.2 * h_3p) + (a_fg + 0.2 * a_3p)) / 2

    # ── POSSESSIONS ──
    f["turnovers_combined"] = _sc(df, "home_turnovers", 14) + _sc(df, "away_turnovers", 14)
    f["steals_combined"] = _sc(df, "home_steals", 7.5) + _sc(df, "away_steals", 7.5)
    f["orb_pct_avg"] = (_sc(df, "home_orb_pct", 0.25) + _sc(df, "away_orb_pct", 0.25)) / 2
    f["fta_rate_avg"] = (_sc(df, "home_fta_rate", 0.28) + _sc(df, "away_fta_rate", 0.28)) / 2

    # ── DEFENSE (combined) ──
    f["opp_fgpct_avg"] = (_sc(df, "home_opp_fgpct", 0.46) + _sc(df, "away_opp_fgpct", 0.46)) / 2
    f["opp_threepct_avg"] = (_sc(df, "home_opp_threepct", 0.36) + _sc(df, "away_opp_threepct", 0.36)) / 2
    f["opp_suppression_sum"] = _sc(df, "home_opp_suppression", 0) + _sc(df, "away_opp_suppression", 0)

    # ── ROLLING PBP (combined) ──
    f["roll_bench_combined"] = _sc(df, "home_roll_bench_pts", 0) + _sc(df, "away_roll_bench_pts", 0)
    f["roll_paint_combined"] = _sc(df, "home_roll_paint_pts", 0) + _sc(df, "away_roll_paint_pts", 0)
    f["roll_fastbreak_combined"] = _sc(df, "home_roll_fast_break_pts", 0) + _sc(df, "away_roll_fast_break_pts", 0)
    f["roll_game_pf_combined"] = _sc(df, "home_roll_game_pf", 0) + _sc(df, "away_roll_game_pf", 0)
    f["roll_max_run_avg"] = (_sc(df, "home_roll_max_run", 0) + _sc(df, "away_roll_max_run", 0)) / 2
    f["roll_ft_trip_combined"] = _sc(df, "home_roll_ft_trip_rate", 0) + _sc(df, "away_roll_ft_trip_rate", 0)

    # ── ENRICHMENT (combined) ──
    f["ceiling_combined"] = _sc(df, "home_ceiling", 0) + _sc(df, "away_ceiling", 0)
    f["floor_combined"] = _sc(df, "home_floor", 0) + _sc(df, "away_floor", 0)
    f["scoring_var_combined"] = _sc(df, "home_scoring_var", 0) + _sc(df, "away_scoring_var", 0)

    # ── CONTEXT ──
    h_rest = _sc(df, "home_days_rest", 2); a_rest = _sc(df, "away_days_rest", 2)
    f["rest_combined"] = h_rest + a_rest
    f["b2b_either"] = ((h_rest == 0) | (a_rest == 0)).astype(int)
    f["altitude_factor"] = 0
    if "home_team" in df.columns:
        f["altitude_factor"] = (df["home_team"].astype(str).str.upper() == "DEN").astype(int)

    # ── REFEREE ──
    f["ref_ou_bias"] = _sc(df, "ref_ou_bias", 0)
    f["ref_pace_impact"] = _sc(df, "ref_pace_impact", 0)

    # ── DIFFS that matter for totals ──
    f["efg_diff"] = (h_fg + 0.2 * h_3p) - (a_fg + 0.2 * a_3p)
    hw = _sc(df, "home_wins", 20); hl = _sc(df, "home_losses", 20)
    aw = _sc(df, "away_wins", 20); al = _sc(df, "away_losses", 20)
    f["blowout_risk"] = np.abs(hw / np.maximum(hw + hl, 1) - aw / np.maximum(aw + al, 1))

    f = f.select_dtypes(include=[np.number]).fillna(0)
    return f, list(f.columns)


def build_ou_features_live(row, feature_cols):
    """Build O/U features from live row dict. Used by nba_full_predict.py."""
    def g(key, default=0):
        v = row.get(key, default)
        if v is None: return default
        try:
            v = float(v)
            return default if np.isnan(v) else v
        except: return default

    f = {}
    f["market_total"] = g("market_ou_total", 220)
    f["overround"] = g("overround", 0.04)
    h_ppg = g("home_ppg", 112); a_ppg = g("away_ppg", 112)
    f["ppg_combined"] = h_ppg + a_ppg
    f["opp_ppg_combined"] = g("home_opp_ppg", 112) + g("away_opp_ppg", 112)
    f["net_rtg_sum"] = g("home_net_rtg", 0) + g("away_net_rtg", 0)
    f["ou_gap"] = f["ppg_combined"] - f["market_total"]
    h_tempo = g("home_tempo", 100); a_tempo = g("away_tempo", 100)
    f["tempo_combined"] = h_tempo + a_tempo
    f["pace_min"] = min(h_tempo, a_tempo)
    f["ppg_x_pace"] = (f["ppg_combined"] * f["tempo_combined"]) / 20000
    h_fg = g("home_fgpct", 0.46); a_fg = g("away_fgpct", 0.46)
    h_3p = g("home_threepct", 0.36); a_3p = g("away_threepct", 0.36)
    f["fgpct_avg"] = (h_fg + a_fg) / 2
    f["threepct_avg"] = (h_3p + a_3p) / 2
    f["ftpct_avg"] = (g("home_ftpct", 0.77) + g("away_ftpct", 0.77)) / 2
    f["efg_avg"] = ((h_fg + 0.2 * h_3p) + (a_fg + 0.2 * a_3p)) / 2
    f["turnovers_combined"] = g("home_turnovers", 14) + g("away_turnovers", 14)
    f["steals_combined"] = g("home_steals", 7.5) + g("away_steals", 7.5)
    f["orb_pct_avg"] = (g("home_orb_pct", 0.25) + g("away_orb_pct", 0.25)) / 2
    f["fta_rate_avg"] = (g("home_fta_rate", 0.28) + g("away_fta_rate", 0.28)) / 2
    f["opp_fgpct_avg"] = (g("home_opp_fgpct", 0.46) + g("away_opp_fgpct", 0.46)) / 2
    f["opp_threepct_avg"] = (g("home_opp_threepct", 0.36) + g("away_opp_threepct", 0.36)) / 2
    f["opp_suppression_sum"] = g("home_opp_suppression", 0) + g("away_opp_suppression", 0)
    f["roll_bench_combined"] = g("home_roll_bench_pts", 0) + g("away_roll_bench_pts", 0)
    f["roll_paint_combined"] = g("home_roll_paint_pts", 0) + g("away_roll_paint_pts", 0)
    f["roll_fastbreak_combined"] = g("home_roll_fast_break_pts", 0) + g("away_roll_fast_break_pts", 0)
    f["roll_game_pf_combined"] = g("home_roll_game_pf", 0) + g("away_roll_game_pf", 0)
    f["roll_max_run_avg"] = (g("home_roll_max_run", 0) + g("away_roll_max_run", 0)) / 2
    f["roll_ft_trip_combined"] = g("home_roll_ft_trip_rate", 0) + g("away_roll_ft_trip_rate", 0)
    f["ceiling_combined"] = g("home_ceiling", 0) + g("away_ceiling", 0)
    f["floor_combined"] = g("home_floor", 0) + g("away_floor", 0)
    f["scoring_var_combined"] = g("home_scoring_var", 0) + g("away_scoring_var", 0)
    h_rest = g("home_days_rest", 2); a_rest = g("away_days_rest", 2)
    f["rest_combined"] = h_rest + a_rest
    f["b2b_either"] = 1 if (h_rest == 0 or a_rest == 0) else 0
    f["altitude_factor"] = 1 if str(row.get("home_team", "")).upper() == "DEN" else 0
    f["ref_ou_bias"] = g("ref_ou_bias", 0)
    f["ref_pace_impact"] = g("ref_pace_impact", 0)
    f["efg_diff"] = (h_fg + 0.2 * h_3p) - (a_fg + 0.2 * a_3p)
    hw = g("home_wins", 20); hl = g("home_losses", 20)
    aw = g("away_wins", 20); al = g("away_losses", 20)
    f["blowout_risk"] = abs(hw / max(hw + hl, 1) - aw / max(aw + al, 1))
    for col in feature_cols:
        if col not in f: f[col] = 0.0
    return pd.DataFrame([{k: f[k] for k in feature_cols}])


class OUBlendModel:
    """Blended CatBoost + Ridge for O/U. Pickle-compatible (module-level class)."""
    def __init__(self, primary, secondary, w_primary=0.7):
        self.primary = primary
        self.secondary = secondary
        self.w_primary = w_primary
    def predict(self, X):
        return self.w_primary * self.primary.predict(X) + (1 - self.w_primary) * self.secondary.predict(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA O/U MODEL v2 — COMBINED Features for Total Points")
    print("=" * 70)

    df = load_training_data("nba_training_data.parquet")
    df["target_total"] = df["actual_home_score"].astype(float) + df["actual_away_score"].astype(float)
    y = df["target_total"].values

    X_df, feature_names = build_ou_features(df)
    feature_cols = feature_names
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Games: {len(X_df)}")
    print(f"  Target mean: {y.mean():.1f}, std: {y.std():.1f}")

    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X_df = X_df.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]; df = df.iloc[sort_idx].reset_index(drop=True)
    X = X_df.values
    market_total = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0).values

    # Walk-forward
    n_folds = 30; fold_size = len(X) // (n_folds + 3); min_train = fold_size * 3
    all_preds = np.full(len(X), np.nan); t0 = time.time()
    print(f"\n  Walk-forward: {n_folds} folds, fold={fold_size}, min_train={min_train}")

    for fold in range(n_folds):
        ts = min_train + fold * fold_size; te = min(ts + fold_size, len(X))
        if ts >= len(X): break
        sc = StandardScaler(); X_tr = sc.fit_transform(X[:ts]); X_te = sc.transform(X[ts:te])
        if HAS_CAT:
            m = CatBoostRegressor(depth=4, iterations=600, learning_rate=0.03, l2_leaf_reg=5, random_seed=42, verbose=0)
        else:
            m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
        m.fit(X_tr, y[:ts]); r = Ridge(alpha=1.0); r.fit(X_tr, y[:ts])
        all_preds[ts:te] = 0.7 * m.predict(X_te) + 0.3 * r.predict(X_te)
        if (fold + 1) % 10 == 0: print(f"    Fold {fold+1}/{n_folds} ({time.time()-t0:.0f}s)")

    print(f"  Done in {time.time()-t0:.0f}s")
    valid = ~np.isnan(all_preds); has_mkt = valid & (market_total > 0)
    mae = np.mean(np.abs(all_preds[valid] - y[valid]))
    print(f"\n  Walk-forward MAE: {mae:.2f}")

    ou_results = {}
    if has_mkt.sum() > 100:
        mkt_mae = np.mean(np.abs(market_total[has_mkt] - y[has_mkt]))
        mdl_mae = np.mean(np.abs(all_preds[has_mkt] - y[has_mkt]))
        print(f"  Model MAE (mkt games): {mdl_mae:.2f}")
        print(f"  Market MAE:            {mkt_mae:.2f}")
        print(f"  Model vs market:       {mdl_mae - mkt_mae:+.2f}")
        mt = all_preds[has_mkt]; mk = market_total[has_mkt]; ac = y[has_mkt]
        edge = mt - mk; co = (edge > 0) & (ac > mk); cu = (edge < 0) & (ac < mk); dec = ac != mk
        print(f"\n  {'Edge':>6s} {'Games':>7s} {'Acc%':>6s} {'ROI':>7s}")
        print("  " + "-" * 30)
        for t in [0, 2, 3, 4, 5, 7, 10]:
            mask = (np.abs(edge) >= t) & dec; n = mask.sum()
            if n < 20: continue
            nc = (co[mask] | cu[mask]).sum(); acc = nc / n; roi = (acc * 1.909 - 1) * 100
            tag = "YES" if acc > 0.524 else "no"
            print(f"  {t:>5d}+ {n:>7d} {acc:>5.1%} {roi:>+6.1f}%  {tag}")
            ou_results[t] = {"n": int(n), "acc": round(acc, 4), "roi": round(roi, 1)}

    # Production training
    print(f"\n  Training production model...")
    scaler = StandardScaler(); X_s = scaler.fit_transform(X)
    if HAS_CAT:
        cat = CatBoostRegressor(depth=4, iterations=600, learning_rate=0.03, l2_leaf_reg=5, random_seed=42, verbose=0)
    else:
        cat = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    cat.fit(X_s, y); ridge = Ridge(alpha=1.0); ridge.fit(X_s, y)
    blend = OUBlendModel(cat, ridge, 0.7)
    print(f"  Blend MAE: {np.mean(np.abs(blend.predict(X_s) - y)):.2f}")
    bias = float(np.mean(y - blend.predict(X_s)))
    print(f"  Bias: {bias:+.2f}")

    from datetime import datetime, timezone
    bundle = {"reg": blend, "scaler": scaler, "ou_feature_cols": feature_cols,
              "bias_correction": bias, "model_type": "ou_catboost_ridge_v2",
              "n_games": len(X), "n_features": len(feature_cols),
              "cv_mae": round(mae, 2), "trained_at": datetime.now(timezone.utc).isoformat()}

    import joblib
    joblib.dump(bundle, "nba_ou_v2.pkl", compress=3)
    print(f"  Saved: nba_ou_v2.pkl ({os.path.getsize('nba_ou_v2.pkl')/1024:.0f} KB)")

    if args.upload:
        try:
            from db import save_model
            save_model("nba_ou", bundle)
            print(f"  ✅ Uploaded to Supabase as 'nba_ou'")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
    else:
        print(f"  To upload: python3 nba_ou_train_v2.py --upload")

    print(f"\n  DONE: {len(feature_cols)} features, MAE={mae:.2f}")


if __name__ == "__main__":
    main()
