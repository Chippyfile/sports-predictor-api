#!/usr/bin/env python3
"""
mlb_savant_parse.py — Parse Baseball Savant CSVs into training features
=======================================================================
Reads from: mlb stats/20XX/ folders (2015-2025)
Each folder has ~5 CSVs from Savant leaderboards.
Auto-detects: team batting, team pitching, team EV, player pitching.

Creates per-team per-season features:
  - woba_luck_gap:     wOBA - xwOBA (positive = overperforming)
  - xwoba:             expected wOBA (true contact quality)
  - barrel_rate:       barrel% (team batting)
  - barrel_rate_against: barrel% allowed (team pitching)
  - hard_hit_pct:      hard-hit% (team batting)
  - ev_avg:            avg exit velocity

Merges into mlb_training_data.parquet and tests correlations.

Usage:
    python3 mlb_savant_parse.py              # Parse + test
    python3 mlb_savant_parse.py --write      # Parse + save to parquet
"""
import sys, os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Savant uses modern abbreviations; our parquet uses retrosheet
SAVANT_TO_RETRO = {
    "LAA": "ANA", "CWS": "CHA", "CHC": "CHN", "KC": "KCA",
    "LAD": "LAN", "NYY": "NYA", "NYM": "NYN", "SD": "SDN",
    "SF": "SFN", "STL": "SLN", "TB": "TBA", "WSH": "WAS",
    "AZ": "ARI",  # Savant uses AZ, retrosheet uses ARI
    # Teams that match already
    "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CIN": "CIN",
    "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "TEX": "TEX",
    "TOR": "TOR",
}


def detect_csv_type(filepath):
    """Auto-detect what type of Savant CSV this is."""
    try:
        df = pd.read_csv(filepath, nrows=3)
        cols = [c.lower().strip() for c in df.columns]
        
        has_team_id = any("team_id" in c for c in cols)
        has_player_id = any("player_id" in c for c in cols)
        has_est_woba = any("est_woba" in c for c in cols)
        has_barrel = any("brl_percent" in c or "barrel" in c for c in cols)
        has_ev = any("avg_hit_speed" in c for c in cols)
        has_k_percent = any("k_percent" in c for c in cols)
        
        if has_player_id and has_est_woba:
            return "player_expected"
        elif has_player_id and has_k_percent:
            return "player_pitching_stats"
        elif has_team_id and has_est_woba:
            # Need to distinguish batting vs pitching
            # Pitching teams tend to have higher wOBA (runs allowed)
            # Check if it looks like batting or pitching based on wOBA values
            full = pd.read_csv(filepath)
            avg_woba = pd.to_numeric(full.get("woba", full.get("est_woba", 0)), errors="coerce").mean()
            # Team batting expected stats — could be either. Check the filename
            fname = os.path.basename(filepath).lower()
            if "(1)" in fname or "__1_" in fname or "_pitch" in fname:
                return "team_pitching_expected"
            # If avg wOBA > 0.340, likely pitching (runs allowed tend to be higher)
            # But this is unreliable. Let's check if there are duplicate team_ids
            # between this file and exit_velocity in same folder
            return "team_expected"  # default — we'll split later
        elif has_team_id and has_ev:
            fname = os.path.basename(filepath).lower()
            if "(1)" in fname or "__1_" in fname or "_pitch" in fname:
                return "team_pitching_ev"
            return "team_ev"
        
        return "unknown"
    except Exception as e:
        return f"error: {e}"


def parse_all_folders():
    """Scan mlb stats/20XX/ folders and parse all CSVs."""
    base_dir = os.path.join(SCRIPT_DIR, "mlb stats")
    
    if not os.path.exists(base_dir):
        # Try alternate locations
        for alt in ["mlb_stats", "mlb-stats", "savant"]:
            alt_path = os.path.join(SCRIPT_DIR, alt)
            if os.path.exists(alt_path):
                base_dir = alt_path
                break
    
    if not os.path.exists(base_dir):
        print(f"  ❌ Folder not found: {base_dir}")
        print(f"     Expected: sports-predictor-api/mlb stats/20XX/")
        return None, None
    
    print(f"  Scanning: {base_dir}")
    
    team_batting = []    # team expected stats (batting side)
    team_pitching = []   # team expected stats (pitching side)
    team_bat_ev = []     # team exit velocity (batting)
    team_pitch_ev = []   # team exit velocity (pitching)
    player_pitch = []    # player-level pitching expected stats
    
    year_folders = sorted(glob.glob(os.path.join(base_dir, "20*")))
    if not year_folders:
        # Maybe CSVs are directly in the folder
        year_folders = [base_dir]
    
    for folder in year_folders:
        folder_name = os.path.basename(folder)
        try:
            year = int(folder_name)
        except ValueError:
            year = None
        
        csvs = glob.glob(os.path.join(folder, "*.csv"))
        if not csvs:
            continue
        
        print(f"\n  {folder_name}/ ({len(csvs)} files)")
        
        # Sort CSVs and detect types
        # Heuristic: expected_stats.csv = batting, expected_stats(1).csv = pitching
        # exit_velocity.csv = batting EV, exit_velocity(1).csv = pitching EV (if exists)
        expected_files = sorted([f for f in csvs if "expected" in os.path.basename(f).lower()])
        ev_files = sorted([f for f in csvs if "exit" in os.path.basename(f).lower() or "velocity" in os.path.basename(f).lower()])
        stats_files = sorted([f for f in csvs if "stats" in os.path.basename(f).lower() and "expected" not in os.path.basename(f).lower()])
        
        for csv_path in csvs:
            fname = os.path.basename(csv_path)
            dtype = detect_csv_type(csv_path)
            
            try:
                df = pd.read_csv(csv_path)
                if year:
                    df["season"] = year
                elif "year" in df.columns:
                    df["season"] = pd.to_numeric(df["year"], errors="coerce")
                
                # Map team IDs to retrosheet codes
                if "team_id" in df.columns:
                    df["team_retro"] = df["team_id"].map(SAVANT_TO_RETRO).fillna(df["team_id"])
                
                is_team = "team_id" in df.columns
                is_player = "player_id" in df.columns
                has_expected = "est_woba" in df.columns
                has_ev = "avg_hit_speed" in df.columns
                
                # Determine batting vs pitching for team files
                # If multiple expected_stats files exist, first = batting, second = pitching
                if is_team and has_expected:
                    idx_in_expected = expected_files.index(csv_path) if csv_path in expected_files else -1
                    if idx_in_expected == 0:
                        team_batting.append(df)
                        print(f"    {fname}: TEAM BATTING expected ({len(df)} teams)")
                    elif idx_in_expected >= 1:
                        team_pitching.append(df)
                        print(f"    {fname}: TEAM PITCHING expected ({len(df)} teams)")
                    else:
                        team_batting.append(df)
                        print(f"    {fname}: TEAM expected (assumed batting, {len(df)} teams)")
                        
                elif is_team and has_ev:
                    idx_in_ev = ev_files.index(csv_path) if csv_path in ev_files else -1
                    if idx_in_ev == 0:
                        team_bat_ev.append(df)
                        print(f"    {fname}: TEAM BATTING EV ({len(df)} teams)")
                    elif idx_in_ev >= 1:
                        team_pitch_ev.append(df)
                        print(f"    {fname}: TEAM PITCHING EV ({len(df)} teams)")
                    else:
                        team_bat_ev.append(df)
                        print(f"    {fname}: TEAM EV (assumed batting, {len(df)} teams)")
                        
                elif is_player and has_expected:
                    player_pitch.append(df)
                    print(f"    {fname}: PLAYER expected ({len(df)} players)")
                    
                elif is_player:
                    player_pitch.append(df)
                    print(f"    {fname}: PLAYER stats ({len(df)} players)")
                    
                else:
                    print(f"    {fname}: ⚠ UNKNOWN format — skipping")
                    
            except Exception as e:
                print(f"    {fname}: ERROR — {e}")
    
    # ── Combine all team data ──
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    
    bat_exp = pd.concat(team_batting, ignore_index=True) if team_batting else pd.DataFrame()
    pitch_exp = pd.concat(team_pitching, ignore_index=True) if team_pitching else pd.DataFrame()
    bat_ev = pd.concat(team_bat_ev, ignore_index=True) if team_bat_ev else pd.DataFrame()
    pitch_ev = pd.concat(team_pitch_ev, ignore_index=True) if team_pitch_ev else pd.DataFrame()
    
    print(f"  Team batting expected: {len(bat_exp)} rows ({bat_exp['season'].nunique() if len(bat_exp) else 0} seasons)")
    print(f"  Team pitching expected: {len(pitch_exp)} rows ({pitch_exp['season'].nunique() if len(pitch_exp) else 0} seasons)")
    print(f"  Team batting EV: {len(bat_ev)} rows ({bat_ev['season'].nunique() if len(bat_ev) else 0} seasons)")
    print(f"  Team pitching EV: {len(pitch_ev)} rows ({pitch_ev['season'].nunique() if len(pitch_ev) else 0} seasons)")
    
    # ── Build team features ──
    # Batting: wOBA luck gap + xwOBA
    team_features = {}
    
    if len(bat_exp) > 0:
        for _, row in bat_exp.iterrows():
            key = (int(row.get("season", 0)), row.get("team_retro", row.get("team_id", "")))
            woba = pd.to_numeric(row.get("woba"), errors="coerce")
            xwoba = pd.to_numeric(row.get("est_woba"), errors="coerce")
            if key not in team_features:
                team_features[key] = {}
            if pd.notna(woba): team_features[key]["bat_woba"] = woba
            if pd.notna(xwoba): team_features[key]["bat_xwoba"] = xwoba
            if pd.notna(woba) and pd.notna(xwoba):
                team_features[key]["bat_luck_gap"] = round(woba - xwoba, 4)
    
    if len(pitch_exp) > 0:
        for _, row in pitch_exp.iterrows():
            key = (int(row.get("season", 0)), row.get("team_retro", row.get("team_id", "")))
            woba = pd.to_numeric(row.get("woba"), errors="coerce")
            xwoba = pd.to_numeric(row.get("est_woba"), errors="coerce")
            if key not in team_features:
                team_features[key] = {}
            if pd.notna(woba): team_features[key]["pitch_woba_against"] = woba
            if pd.notna(xwoba): team_features[key]["pitch_xwoba_against"] = xwoba
            if pd.notna(woba) and pd.notna(xwoba):
                team_features[key]["pitch_luck_gap"] = round(woba - xwoba, 4)
    
    if len(bat_ev) > 0:
        for _, row in bat_ev.iterrows():
            key = (int(row.get("season", 0)), row.get("team_retro", row.get("team_id", "")))
            if key not in team_features:
                team_features[key] = {}
            for col, feat in [("brl_percent", "bat_barrel_rate"), ("ev95percent", "bat_hard_hit_pct"),
                              ("avg_hit_speed", "bat_avg_ev")]:
                val = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(val):
                    team_features[key][feat] = float(val)
    
    if len(pitch_ev) > 0:
        for _, row in pitch_ev.iterrows():
            key = (int(row.get("season", 0)), row.get("team_retro", row.get("team_id", "")))
            if key not in team_features:
                team_features[key] = {}
            for col, feat in [("brl_percent", "pitch_barrel_rate_against"), ("ev95percent", "pitch_hard_hit_against"),
                              ("avg_hit_speed", "pitch_avg_ev_against")]:
                val = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(val):
                    team_features[key][feat] = float(val)
    
    print(f"\n  Team features built: {len(team_features)} team-seasons")
    if team_features:
        sample_key = list(team_features.keys())[0]
        print(f"  Sample ({sample_key}): {team_features[sample_key]}")
    
    return team_features, pd.concat(player_pitch, ignore_index=True) if player_pitch else pd.DataFrame()


def merge_into_training(team_features, parquet_path):
    """Merge Savant team features into training parquet."""
    df = pd.read_parquet(parquet_path)
    df = df[df["season"] != 2020].copy()
    print(f"\n  Training data: {len(df)} games")
    
    if not team_features:
        print("  ⚠ No team features to merge")
        return df
    
    # New feature columns
    new_cols = [
        "home_bat_xwoba", "away_bat_xwoba",
        "home_bat_luck_gap", "away_bat_luck_gap",
        "home_pitch_xwoba_against", "away_pitch_xwoba_against",
        "home_pitch_luck_gap", "away_pitch_luck_gap",
        "home_bat_barrel_rate", "away_bat_barrel_rate",
        "home_bat_hard_hit_pct", "away_bat_hard_hit_pct",
        "home_pitch_barrel_rate_against", "away_pitch_barrel_rate_against",
    ]
    for col in new_cols:
        df[col] = np.nan
    
    matched = 0
    for idx, row in df.iterrows():
        season = int(row.get("season", 0))
        ht = row.get("home_team", "")
        at = row.get("away_team", "")
        
        h_feat = team_features.get((season, ht), {})
        a_feat = team_features.get((season, at), {})
        
        if h_feat or a_feat:
            matched += 1
        
        # Batting features
        for prefix, feat_dict in [("home", h_feat), ("away", a_feat)]:
            if feat_dict.get("bat_xwoba"):
                df.at[idx, f"{prefix}_bat_xwoba"] = feat_dict["bat_xwoba"]
            if feat_dict.get("bat_luck_gap") is not None:
                df.at[idx, f"{prefix}_bat_luck_gap"] = feat_dict["bat_luck_gap"]
            if feat_dict.get("bat_barrel_rate"):
                df.at[idx, f"{prefix}_bat_barrel_rate"] = feat_dict["bat_barrel_rate"]
            if feat_dict.get("bat_hard_hit_pct"):
                df.at[idx, f"{prefix}_bat_hard_hit_pct"] = feat_dict["bat_hard_hit_pct"]
            if feat_dict.get("pitch_xwoba_against"):
                df.at[idx, f"{prefix}_pitch_xwoba_against"] = feat_dict["pitch_xwoba_against"]
            if feat_dict.get("pitch_luck_gap") is not None:
                df.at[idx, f"{prefix}_pitch_luck_gap"] = feat_dict["pitch_luck_gap"]
            if feat_dict.get("pitch_barrel_rate_against"):
                df.at[idx, f"{prefix}_pitch_barrel_rate_against"] = feat_dict["pitch_barrel_rate_against"]
    
    # ── Compute diff features ──
    df["xwoba_luck_gap_diff"] = df["home_bat_luck_gap"].fillna(0) - df["away_bat_luck_gap"].fillna(0)
    df["xwoba_luck_gap_combined"] = df["home_bat_luck_gap"].fillna(0) + df["away_bat_luck_gap"].fillna(0)
    df["bat_xwoba_diff"] = df["home_bat_xwoba"].fillna(0.315) - df["away_bat_xwoba"].fillna(0.315)
    df["barrel_rate_diff"] = df["home_bat_barrel_rate"].fillna(7.0) - df["away_bat_barrel_rate"].fillna(7.0)
    df["hard_hit_diff"] = df["home_bat_hard_hit_pct"].fillna(38.0) - df["away_bat_hard_hit_pct"].fillna(38.0)
    df["pitch_luck_gap_diff"] = df["home_pitch_luck_gap"].fillna(0) - df["away_pitch_luck_gap"].fillna(0)
    df["pitch_luck_gap_combined"] = df["home_pitch_luck_gap"].fillna(0) + df["away_pitch_luck_gap"].fillna(0)
    df["barrel_rate_against_diff"] = df["home_pitch_barrel_rate_against"].fillna(7.0) - df["away_pitch_barrel_rate_against"].fillna(7.0)
    
    # Combined luck: batting + pitching luck tells us if BOTH sides are regressing
    df["total_luck_combined"] = df["xwoba_luck_gap_combined"] + df["pitch_luck_gap_combined"]
    
    print(f"  Matched: {matched}/{len(df)} games ({matched/len(df):.1%})")
    print(f"  Savant features added: {len(new_cols) + 9} columns")
    
    # ── Correlation test ──
    print(f"\n{'='*75}")
    print(f"  SAVANT FEATURE CORRELATIONS")
    print(f"{'='*75}")
    
    df["actual_margin"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) - \
                          pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    df["actual_total"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) + \
                         pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    mkt_ou = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)
    has_ou = mkt_ou > 0
    df["ou_residual"] = df["actual_total"] - mkt_ou
    mkt_spread = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    has_spread = np.abs(mkt_spread) > 0.1
    df["ats_residual"] = df["actual_margin"] - (-mkt_spread)
    
    savant_features = [
        "xwoba_luck_gap_diff", "xwoba_luck_gap_combined", "bat_xwoba_diff",
        "barrel_rate_diff", "hard_hit_diff",
        "pitch_luck_gap_diff", "pitch_luck_gap_combined",
        "barrel_rate_against_diff", "total_luck_combined",
        # Comparison
        "sp_form_combined",
    ]
    
    print(f"\n  {'Feature':<30s} {'ATS r':>8s} {'n':>7s}  {'O/U r':>8s} {'n':>7s}  {'Margin r':>9s}")
    print(f"  {'-'*75}")
    
    for feat in savant_features:
        if feat not in df.columns:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce")
        
        ats_mask = has_spread & vals.notna() & (vals != 0)
        ats_r = vals[ats_mask].corr(df.loc[ats_mask, "ats_residual"]) if ats_mask.sum() > 50 else np.nan
        
        ou_mask = has_ou & vals.notna() & (vals != 0)
        ou_r = vals[ou_mask].corr(df.loc[ou_mask, "ou_residual"]) if ou_mask.sum() > 50 else np.nan
        
        mar_mask = vals.notna() & (vals != 0)
        mar_r = vals[mar_mask].corr(df.loc[mar_mask, "actual_margin"]) if mar_mask.sum() > 50 else np.nan
        
        flag = "★" if pd.notna(ou_r) and abs(ou_r) >= 0.03 else ""
        print(f"  {feat:<30s} {ats_r:>+7.4f} {ats_mask.sum():>7d}  {ou_r:>+7.4f} {ou_mask.sum():>7d}  {mar_r:>+8.4f} {flag}")
    
    return df


def main():
    write = "--write" in sys.argv
    
    print("=" * 70)
    print("  MLB SAVANT DATA PARSER")
    print("=" * 70)
    
    team_features, player_df = parse_all_folders()
    
    if team_features:
        parquet_path = os.path.join(SCRIPT_DIR, "mlb_training_data.parquet")
        df = merge_into_training(team_features, parquet_path)
        
        if write:
            df.to_parquet(parquet_path, index=False)
            print(f"\n  ✅ Saved: {len(df)} games, {len(df.columns)} columns")
        else:
            print(f"\n  Run with --write to save")
    else:
        print("\n  No data parsed. Check folder structure:")
        print("  sports-predictor-api/mlb stats/2015/expected_stats.csv")
        print("  sports-predictor-api/mlb stats/2015/exit_velocity.csv")
        print("  etc.")


if __name__ == "__main__":
    main()
