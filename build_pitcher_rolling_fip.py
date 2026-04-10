#!/usr/bin/env python3
"""
build_pitcher_rolling_fip.py — Compute rolling FIP from Retrosheet pitching.csv
================================================================================
Uses 51K local starter appearances (2015-2025). No API calls needed.
Sweeps 30/45/60/72/90/120 IP windows to find optimal for O/U prediction.

Usage:
    python3 build_pitcher_rolling_fip.py          # Full sweep
    python3 build_pitcher_rolling_fip.py --save   # Save best window to parquet
"""
import sys, time
import pandas as pd
import numpy as np

CFIP = 3.10
WINDOWS = [30, 45, 60, 72, 90, 120]
ROLLING_FIP_OUTPUT = "mlb_pitcher_rolling_fip.parquet"


def load_starters():
    print("  Loading pitching.csv...")
    p = pd.read_csv("pitching.csv", low_memory=False)
    starters = p[(p["p_gs"] == 1) & (p["stattype"] == "value") & (p["date"] >= 20150101)].copy()

    starters["ip"] = pd.to_numeric(starters["p_ipouts"], errors="coerce").fillna(0) / 3.0
    starters["k"] = pd.to_numeric(starters["p_k"], errors="coerce").fillna(0)
    starters["bb"] = pd.to_numeric(starters["p_w"], errors="coerce").fillna(0)
    starters["hr"] = pd.to_numeric(starters["p_hr"], errors="coerce").fillna(0)
    starters["game_date"] = pd.to_datetime(starters["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    starters["season"] = starters["date"].astype(str).str[:4].astype(int)
    starters["is_home"] = starters["vishome"] == "h"

    starters["start_fip"] = np.where(
        starters["ip"] > 0,
        (13 * starters["hr"] + 3 * starters["bb"] - 2 * starters["k"]) / starters["ip"] + CFIP,
        6.0
    )
    starters["start_fip"] = starters["start_fip"].clip(0.0, 12.0)
    starters = starters.sort_values(["id", "game_date"]).reset_index(drop=True)

    print(f"  {len(starters)} starter appearances, {starters['id'].nunique()} pitchers")
    print(f"  Seasons: {sorted(starters['season'].unique())}")
    print(f"  Mean single-start FIP: {starters['start_fip'].mean():.2f}")
    return starters


def compute_rolling_fip(starters, windows=WINDOWS):
    print(f"\n  Computing rolling FIP at windows: {windows}")
    start_time = time.time()

    results = []
    for pid, group in starters.groupby("id"):
        group = group.sort_values("game_date").reset_index(drop=True)
        ips = group["ip"].values
        hrs = group["hr"].values
        bbs = group["bb"].values
        ks = group["k"].values

        for i in range(len(group)):
            row = group.iloc[i]
            result = {
                "pitcher_id": pid,
                "game_date": row["game_date"],
                "season": row["season"],
                "team": row["team"],
                "is_home": row["is_home"],
                "game_ip": row["ip"],
                "start_fip": row["start_fip"],
            }

            for w in windows:
                total_ip = 0
                total_hr = 0
                total_bb = 0
                total_k = 0
                n_starts = 0

                for j in range(i - 1, -1, -1):
                    total_ip += ips[j]
                    total_hr += hrs[j]
                    total_bb += bbs[j]
                    total_k += ks[j]
                    n_starts += 1
                    if total_ip >= w:
                        break

                if total_ip >= 10:
                    fip = (13 * total_hr + 3 * total_bb - 2 * total_k) / total_ip + CFIP
                    fip = max(1.0, min(8.0, fip))
                else:
                    fip = None

                result[f"rolling_fip_{w}"] = fip
                result[f"rolling_ip_{w}"] = round(total_ip, 1)
                result[f"rolling_n_{w}"] = n_starts

            results.append(result)

    df = pd.DataFrame(results)
    elapsed = time.time() - start_time
    print(f"  Done: {len(df)} rows in {elapsed:.1f}s")

    for w in windows:
        col = f"rolling_fip_{w}"
        valid = df[col].notna().sum()
        mean = df[col].mean()
        std = df[col].std()
        print(f"    Window {w:3d} IP: {valid:,}/{len(df):,} have rolling FIP (mean={mean:.2f}, std={std:.2f})")

    return df


def sweep(rolling_df, windows=WINDOWS):
    from mlb_retrain import load_data

    df = load_data()
    df["actual_total"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) + \
                         pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)
    df = df[(df["market_total"] > 0) & (df["actual_total"] > 0)].copy()
    residual = df["actual_total"] - df["market_total"]

    print(f"\n{'='*70}")
    print(f"  ROLLING FIP WINDOW SWEEP")
    print(f"{'='*70}")
    print(f"  Training games with O/U: {len(df)}")

    # Baseline: season FIP (may not exist in training data)
    if "home_starter_fip" in df.columns:
        season_home = pd.to_numeric(df["home_starter_fip"], errors="coerce").fillna(4.0)
        season_away = pd.to_numeric(df["away_starter_fip"], errors="coerce").fillna(4.0)
    else:
        season_home = pd.Series(4.0, index=df.index)
        season_away = pd.Series(4.0, index=df.index)
    season_combined = season_home + season_away
    valid_season = (season_home != 4.0) & residual.notna()
    if valid_season.sum() > 100:
        r_season = season_combined[valid_season].corr(residual[valid_season])
        r_total = season_combined[valid_season].corr(df["actual_total"][valid_season])
        print(f"\n  Baseline (season FIP combined):")
        print(f"    r(residual) = {r_season:+.4f}, r(total) = {r_total:+.4f}, n={valid_season.sum()}")
    else:
        print(f"\n  Baseline: no season FIP in training data — rolling FIP IS the baseline")

    # Build merge keys — retrosheet codes match directly
    home_starters = rolling_df[rolling_df["is_home"] == True].copy()
    away_starters = rolling_df[rolling_df["is_home"] == False].copy()
    home_starters["_key"] = home_starters["game_date"].astype(str).str[:10] + "|" + home_starters["team"]
    away_starters["_key"] = away_starters["game_date"].astype(str).str[:10] + "|" + away_starters["team"]
    df["_home_key"] = df["game_date"].astype(str).str[:10] + "|" + df["home_team"]
    df["_away_key"] = df["game_date"].astype(str).str[:10] + "|" + df["away_team"]

    print(f"\n  Window | r(res) | r(total) | MAE_mkt | MAE_mdl | improve | n_games")
    print(f"  -------|--------|----------|---------|---------|---------|--------")

    best_r = 0
    best_w = 60

    for w in windows:
        col = f"rolling_fip_{w}"
        home_map = home_starters.drop_duplicates("_key", keep="last").set_index("_key")[col]
        away_map = away_starters.drop_duplicates("_key", keep="last").set_index("_key")[col]
        home_rfip = df["_home_key"].map(home_map)
        away_rfip = df["_away_key"].map(away_map)
        rfip_combined = home_rfip + away_rfip

        valid = rfip_combined.notna() & residual.notna()
        if valid.sum() < 100:
            print(f"  {w:3d} IP  | — skip ({valid.sum()} matches)")
            continue

        r_res = rfip_combined[valid].corr(residual[valid])
        r_tot = rfip_combined[valid].corr(df["actual_total"][valid])

        x = rfip_combined[valid].values
        y = residual[valid].values
        mask = np.isfinite(x) & np.isfinite(y)
        coefs = np.polyfit(x[mask], y[mask], 1)
        pred_res = np.polyval(coefs, x[mask])
        mae_model = np.mean(np.abs(y[mask] - pred_res))
        mae_market = np.mean(np.abs(y[mask]))
        delta = mae_market - mae_model

        print(f"  {w:3d} IP  | {r_res:+.4f} | {r_tot:+.4f}   | {mae_market:.3f}   | {mae_model:.3f}   | {delta:+.4f} | {valid.sum():,}")

        if abs(r_res) > abs(best_r):
            best_r = r_res
            best_w = w

    print(f"\n  ★ Best window: {best_w} IP (r={best_r:+.4f})")

    # FIP change: rolling - season
    if valid_season.sum() > 100:
        print(f"\n  --- FIP CHANGE SIGNAL (rolling - season) ---")
        print(f"  Window | r(delta, res) | n")
        print(f"  -------|---------------|--------")
        for w in windows:
            col = f"rolling_fip_{w}"
            home_map = home_starters.drop_duplicates("_key", keep="last").set_index("_key")[col]
            away_map = away_starters.drop_duplicates("_key", keep="last").set_index("_key")[col]
            home_rfip = df["_home_key"].map(home_map)
            away_rfip = df["_away_key"].map(away_map)
            rfip_combined = home_rfip + away_rfip
            delta_fip = rfip_combined - season_combined
            valid = delta_fip.notna() & residual.notna() & (season_home != 4.0)
            if valid.sum() > 100:
                r_delta = delta_fip[valid].corr(residual[valid])
                print(f"  {w:3d} IP  | {r_delta:+.4f}        | {valid.sum():,}")
    else:
        print(f"\n  FIP change signal: skipped (no season FIP in training data)")

    # Reference: sp_form_combined
    if "sp_form_combined" in df.columns:
        sp_form = pd.to_numeric(df["sp_form_combined"], errors="coerce")
        valid = sp_form.notna() & residual.notna() & (sp_form != 0)
        if valid.sum() > 100:
            r_spf = sp_form[valid].corr(residual[valid])
            print(f"\n  Reference: sp_form_combined r(residual) = {r_spf:+.4f} (n={valid.sum()})")

    return best_w


def main():
    starters = load_starters()
    rolling = compute_rolling_fip(starters, WINDOWS)
    best_w = sweep(rolling, WINDOWS)

    if "--save" in sys.argv:
        rolling.to_parquet(ROLLING_FIP_OUTPUT, index=False)
        print(f"\n  Saved: {ROLLING_FIP_OUTPUT} ({len(rolling)} rows)")
        print(f"  Use: fetch_rolling_fip(target_ip={best_w})")


if __name__ == "__main__":
    main()
