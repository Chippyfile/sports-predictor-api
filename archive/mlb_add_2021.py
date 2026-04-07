#!/usr/bin/env python3
"""
mlb_add_2021.py — Add 2021 season to mlb_training_data.parquet
================================================================
Parses Retrosheet game logs + odds data → appends to training parquet.
Missing features (wOBA, FIP, etc.) get league-average defaults.
The heuristic backfill in load_data() will compute predictions from these.

Usage:
    python3 mlb_add_2021.py              # Preview what will be added
    python3 mlb_add_2021.py --write      # Actually append to parquet
"""
import sys, os, csv
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════
# STATIC LOOKUPS
# ═══════════════════════════════════════════════════════════

# Retrosheet park_id → park_factor (matches PARK_FACTORS in mlb_full_predict.py)
PARK_FACTORS = {
    "ANA01": 1.02, "PHO01": 1.03, "ATL03": 1.00, "BAL12": 0.95,
    "BOS07": 1.04, "CHI11": 1.04, "CIN09": 1.00, "CLE08": 0.97,
    "DEN02": 1.16, "DET05": 0.98, "HOU03": 0.99, "KAN06": 1.01,
    "LOS03": 1.00, "MIA02": 0.97, "MIL06": 1.02, "MIN04": 0.98,
    "NYC21": 1.03, "NYC20": 1.03, "OAK01": 0.99, "PHI13": 1.05,
    "PIT08": 0.96, "SAN02": 0.95, "SEA03": 0.94, "SFO03": 0.96,
    "STL10": 0.98, "STP01": 0.96, "ARL02": 1.01, "TOR02": 1.01,
    "WAS11": 1.01, "DUN01": 1.01,  # TD Ballpark (Jays temp 2021)
    "BUF05": 1.00,  # Sahlen Field (Jays temp 2021)
}

# Modern → retrosheet team mapping (for odds merge)
MODERN_TO_RETRO = {
    "LAA": "ANA", "CWS": "CHA", "CHC": "CHN", "KC": "KCA",
    "LAD": "LAN", "NYY": "NYA", "NYM": "NYN", "SD": "SDN",
    "SF": "SFN", "STL": "SLN", "TB": "TBA", "WSH": "WAS",
}

# 2021 league averages (from FanGraphs)
LG_2021 = {
    "lg_woba": 0.313, "lg_fip": 4.26, "lg_rpg": 4.53,
    "lg_era": 4.26, "lg_k9": 8.7, "lg_bb9": 3.2,
    "lg_bullpen_era": 4.10, "lg_sp_ip": 5.3,
}

# Dome parks (weather doesn't matter)
DOME_PARKS = {"HOU03", "MIA02", "MIL06", "MIN04", "STP01", "ARL02", "PHO01"}


def parse_game_logs(path):
    """Parse Retrosheet GL file into game rows."""
    games = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 94:
                continue
            try:
                date = r[0].strip().strip('"')
                game_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                away = r[3].strip().strip('"')
                home = r[6].strip().strip('"')
                away_score = int(r[9])
                home_score = int(r[10])
                day_night = r[12].strip().strip('"')
                park_id = r[16].strip().strip('"')
                
                # Umpire (HP ump name at field 79)
                ump_name = r[79].strip().strip('"') if len(r) > 79 else ""
                
                # Starting pitchers (names at fields 89 and 93)
                away_sp = r[89].strip().strip('"') if len(r) > 89 else ""
                home_sp = r[93].strip().strip('"') if len(r) > 93 else ""
                
                games.append({
                    "game_date": game_date,
                    "season": 2021,
                    "home_team": home,
                    "away_team": away,
                    "actual_home_runs": home_score,
                    "actual_away_runs": away_score,
                    "home_win": 1 if home_score > away_score else 0,
                    "park_id": park_id,
                    "park_factor": PARK_FACTORS.get(park_id, 1.0),
                    "day_night": day_night,
                    "ump_name_raw": ump_name,
                    "home_starter_name": home_sp,
                    "away_starter_name": away_sp,
                })
            except (ValueError, IndexError):
                continue
    return games


def main():
    write = "--write" in sys.argv
    
    print("=" * 70)
    print("  ADD 2021 SEASON TO MLB TRAINING DATA")
    print("=" * 70)
    
    # ── Parse game logs ──
    gl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gl2021.txt")
    if not os.path.exists(gl_path):
        print(f"  ❌ Not found: {gl_path}")
        return
    
    games = parse_game_logs(gl_path)
    print(f"  Parsed: {len(games)} games from Retrosheet")
    
    gl_df = pd.DataFrame(games)
    
    # Drop early season (before April 15, matching existing pipeline)
    gl_df = gl_df[gl_df["game_date"] >= "2021-04-15"].copy()
    print(f"  After April 15 filter: {len(gl_df)} games")
    
    # ── Merge odds data ──
    odds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_odds_2014_2021.csv")
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path)
        odds = odds[odds["season"] == 2021].copy()
        odds["home_team"] = odds["home_team"].replace(MODERN_TO_RETRO)
        odds["away_team"] = odds["away_team"].replace(MODERN_TO_RETRO)
        
        before = len(gl_df)
        gl_df = gl_df.merge(
            odds[["game_date", "home_team", "away_team", "market_ou_total",
                  "market_spread_home", "market_home_ml", "market_away_ml"]],
            on=["game_date", "home_team", "away_team"],
            how="left"
        )
        matched = gl_df["market_ou_total"].notna().sum()
        print(f"  Odds merged: {matched}/{len(gl_df)} games have market O/U")
    else:
        print(f"  ⚠ No odds file — market data will be missing")
    
    # ── Fill feature defaults (2021 league averages) ──
    defaults = {
        "home_woba": LG_2021["lg_woba"],
        "away_woba": LG_2021["lg_woba"],
        "home_sp_fip": LG_2021["lg_fip"],
        "away_sp_fip": LG_2021["lg_fip"],
        "home_fip": LG_2021["lg_fip"],
        "away_fip": LG_2021["lg_fip"],
        "home_bullpen_era": LG_2021["lg_bullpen_era"],
        "away_bullpen_era": LG_2021["lg_bullpen_era"],
        "home_k9": LG_2021["lg_k9"],
        "away_k9": LG_2021["lg_k9"],
        "home_bb9": LG_2021["lg_bb9"],
        "away_bb9": LG_2021["lg_bb9"],
        "home_sp_ip": LG_2021["lg_sp_ip"],
        "away_sp_ip": LG_2021["lg_sp_ip"],
        "temp_f": 72.0,  # average game temp
        "wind_mph": 6.0,
        "wind_out_flag": 0,
        "home_rest_days": 1.0,
        "away_rest_days": 1.0,
        "home_travel": 0,
        "away_travel": 0,
        "home_lineup_confirmed": 0,
        "away_lineup_confirmed": 0,
        "home_platoon_delta": 0,
        "away_platoon_delta": 0,
        "is_outlier_season": 0,
        "season_weight": 0.75,  # discount since features are defaults
        "win_pct_home": 0.5,
        "pred_home_runs": 4.5,
        "pred_away_runs": 4.5,
    }
    for col, val in defaults.items():
        if col not in gl_df.columns:
            gl_df[col] = val
    
    # ── Umpire run environment (compute from 2021 game logs) ──
    # Average total runs for each umpire across all their 2021 games
    gl_df["total_runs"] = gl_df["actual_home_runs"] + gl_df["actual_away_runs"]
    ump_avg = gl_df.groupby("ump_name_raw")["total_runs"].mean()
    gl_df["ump_run_env"] = gl_df["ump_name_raw"].map(ump_avg).fillna(LG_2021["lg_rpg"] * 2)
    
    # ── Advanced rolling features (defaults — no history to compute from) ──
    adv_defaults = {
        "pyth_residual_diff": 0.0,
        "babip_luck_diff": 0.0,
        "scoring_entropy_diff": 0.0,
        "first_inn_rate_diff": 0.0,
        "clutch_divergence_diff": 0.0,
        "opp_adj_form_diff": 0.0,
        "scoring_entropy_combined": 5.0,
        "first_inn_rate_combined": 0.8,
        "series_game_num": 1.0,
        "sp_relative_fip_diff": 0.0,
    }
    for col, val in adv_defaults.items():
        if col not in gl_df.columns:
            gl_df[col] = val
    
    # ── Load existing parquet and check for overlap ──
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_training_data.parquet")
    if not os.path.exists(parquet_path):
        print(f"  ❌ Not found: {parquet_path}")
        return
    
    existing = pd.read_parquet(parquet_path)
    print(f"\n  Existing parquet: {len(existing)} games, seasons {sorted(existing['season'].unique())}")
    
    if 2021 in existing["season"].unique():
        n_2021 = (existing["season"] == 2021).sum()
        print(f"  ⚠ 2021 already in parquet ({n_2021} games). Skipping to avoid duplicates.")
        print(f"  To replace, first remove 2021: existing = existing[existing['season'] != 2021]")
        return
    
    # ── Align columns ──
    # Add any missing columns from existing parquet as NaN
    for col in existing.columns:
        if col not in gl_df.columns:
            gl_df[col] = np.nan
    
    # Only keep columns that exist in the parquet
    gl_df = gl_df[[c for c in existing.columns if c in gl_df.columns]]
    
    print(f"\n  2021 data prepared: {len(gl_df)} games")
    print(f"  With market O/U: {gl_df['market_ou_total'].notna().sum()}")
    print(f"  Columns matched: {len(gl_df.columns)}/{len(existing.columns)}")
    
    missing_cols = set(existing.columns) - set(gl_df.columns)
    if missing_cols:
        print(f"  Missing columns (will be NaN): {sorted(missing_cols)}")
    
    # Preview
    print(f"\n  Sample row:")
    row = gl_df.iloc[0]
    for col in ["game_date", "home_team", "away_team", "actual_home_runs", "actual_away_runs",
                 "market_ou_total", "market_spread_home", "park_factor", "season_weight"]:
        if col in gl_df.columns:
            print(f"    {col}: {row[col]}")
    
    if not write:
        print(f"\n  Run with --write to append to parquet")
        return
    
    # ── Append and save ──
    combined = pd.concat([existing, gl_df], ignore_index=True)
    combined.to_parquet(parquet_path, index=False)
    print(f"\n  ✅ Saved: {len(combined)} games ({len(existing)} + {len(gl_df)} new)")
    print(f"  Seasons: {sorted(combined['season'].unique())}")


if __name__ == "__main__":
    main()
