#!/usr/bin/env python3
"""
compute_advanced_features.py — Build advanced features from historical data
============================================================================
Features computed:

1. HEAD-TO-HEAD MATCHUP HISTORY
   - h2h_margin_avg: average margin in previous meetings (last 3 seasons)
   - h2h_games: number of previous meetings
   - h2h_home_win_rate: how often home team wins this matchup

2. CONFERENCE STRENGTH
   - conf_strength_diff: aggregate conference adj_em difference
   - cross_conf_flag: 1 if teams are from different conferences
   
3. PACE-ADJUSTED STATS  
   - pace_adj_ppg_diff: PPG normalized to 70 possessions
   - pace_adj_opp_ppg_diff: opponent PPG normalized to 70 possessions
   
4. RECENCY FEATURES
   - recent_form_5g: win% in last 5 games (more responsive than season form)
   - scoring_trend_5g: ppg trend over last 5 games (improving/declining)

Run:
    python3 -u compute_advanced_features.py --dry-run
    python3 -u compute_advanced_features.py
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

DRY_RUN = "--dry-run" in sys.argv
PARQUET = "ncaa_training_data.parquet"

print("=" * 60)
print("  ADVANCED FEATURE COMPUTATION")
print("=" * 60)

df = pd.read_parquet(PARQUET)
df = df.sort_values(["season", "game_date"]).reset_index(drop=True)
print(f"  Loaded {len(df)} games")

# ═══════════════════════════════════════════════════════════════
# 1. HEAD-TO-HEAD MATCHUP HISTORY
# ═══════════════════════════════════════════════════════════════

print("\n  Computing head-to-head features...")
t0 = time.time()

# Track all completed games between team pairs
# Key: frozenset(team_a_id, team_b_id) → list of (date, season, home_id, margin)
h2h_history = defaultdict(list)

df["h2h_margin_avg"] = np.nan
df["h2h_games"] = 0
df["h2h_home_win_rate"] = np.nan

for idx, row in df.iterrows():
    h_id = str(row.get("home_team_id", ""))
    a_id = str(row.get("away_team_id", ""))
    season = row.get("season", 0)
    date = row.get("game_date", "")
    
    if not h_id or not a_id or h_id == a_id:
        continue
    
    pair_key = frozenset([h_id, a_id])
    
    # Look up history BEFORE this game (no leakage)
    # Only use games from last 3 seasons
    history = [g for g in h2h_history[pair_key] 
               if g["season"] >= season - 3 and g["date"] < date]
    
    if history:
        n_games = len(history)
        # Calculate margin from home team's perspective in THIS game
        margins = []
        home_wins = 0
        for g in history:
            if g["home_id"] == h_id:
                margins.append(g["margin"])
                if g["margin"] > 0: home_wins += 1
            else:
                margins.append(-g["margin"])  # flip perspective
                if g["margin"] < 0: home_wins += 1
        
        df.at[idx, "h2h_margin_avg"] = np.mean(margins)
        df.at[idx, "h2h_games"] = n_games
        df.at[idx, "h2h_home_win_rate"] = home_wins / n_games if n_games > 0 else 0.5
    
    # Record this game's result for future lookups
    margin = row.get("actual_home_score", np.nan)
    away_score = row.get("actual_away_score", np.nan)
    if pd.notna(margin) and pd.notna(away_score):
        actual_margin = margin - away_score
        h2h_history[pair_key].append({
            "date": date, "season": season, "home_id": h_id,
            "margin": actual_margin,
        })
    
    if idx > 0 and idx % 10000 == 0:
        print(f"    {idx}/{len(df)} processed...")

h2h_filled = df["h2h_games"].gt(0).sum()
print(f"  H2H: {h2h_filled} games have matchup history ({h2h_filled/len(df)*100:.1f}%)")
print(f"  Time: {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# 2. CONFERENCE STRENGTH
# ═══════════════════════════════════════════════════════════════

print("\n  Computing conference strength features...")

# Build conference average adj_em per season (from games BEFORE each date)
# This is a rolling calculation to avoid leakage

df["conf_strength_diff"] = 0.0
df["cross_conf_flag"] = 0

# Pre-compute: for each season, build a mapping of team → conference
# and conference → list of team adj_ems
# We use the team's adj_em as a proxy for their contribution to conference strength

season_conf_teams = defaultdict(lambda: defaultdict(list))

for idx, row in df.iterrows():
    h_conf = str(row.get("home_conference", ""))
    a_conf = str(row.get("away_conference", ""))
    h_em = row.get("home_adj_em")
    a_em = row.get("away_adj_em")
    season = row.get("season", 0)
    
    if h_conf and a_conf and h_conf != "" and a_conf != "":
        df.at[idx, "cross_conf_flag"] = int(h_conf != a_conf)
    
    # Track conference strength
    if h_conf and pd.notna(h_em):
        season_conf_teams[season][h_conf].append(float(h_em))
    if a_conf and pd.notna(a_em):
        season_conf_teams[season][a_conf].append(float(a_em))

# Now compute conference strength diff
for idx, row in df.iterrows():
    h_conf = str(row.get("home_conference", ""))
    a_conf = str(row.get("away_conference", ""))
    season = row.get("season", 0)
    
    if not h_conf or not a_conf:
        continue
    
    h_ems = season_conf_teams.get(season, {}).get(h_conf, [])
    a_ems = season_conf_teams.get(season, {}).get(a_conf, [])
    
    if h_ems and a_ems:
        df.at[idx, "conf_strength_diff"] = np.mean(h_ems) - np.mean(a_ems)

conf_filled = (df["conf_strength_diff"] != 0).sum()
print(f"  Conference strength: {conf_filled} games ({conf_filled/len(df)*100:.1f}%)")
cross = df["cross_conf_flag"].sum()
print(f"  Cross-conference games: {cross} ({cross/len(df)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# 3. PACE-ADJUSTED STATS
# ═══════════════════════════════════════════════════════════════

print("\n  Computing pace-adjusted features...")

STANDARD_PACE = 70.0  # normalize to 70 possessions

h_tempo = pd.to_numeric(df.get("home_tempo", 70), errors="coerce").fillna(70)
a_tempo = pd.to_numeric(df.get("away_tempo", 70), errors="coerce").fillna(70)
h_ppg = pd.to_numeric(df.get("home_ppg", 0), errors="coerce").fillna(0)
a_ppg = pd.to_numeric(df.get("away_ppg", 0), errors="coerce").fillna(0)
h_opp = pd.to_numeric(df.get("home_opp_ppg", 0), errors="coerce").fillna(0)
a_opp = pd.to_numeric(df.get("away_opp_ppg", 0), errors="coerce").fillna(0)

# Pace-adjust: multiply by (standard_pace / team_pace)
h_ppg_adj = np.where(h_tempo > 0, h_ppg * (STANDARD_PACE / h_tempo), h_ppg)
a_ppg_adj = np.where(a_tempo > 0, a_ppg * (STANDARD_PACE / a_tempo), a_ppg)
h_opp_adj = np.where(h_tempo > 0, h_opp * (STANDARD_PACE / h_tempo), h_opp)
a_opp_adj = np.where(a_tempo > 0, a_opp * (STANDARD_PACE / a_tempo), a_opp)

df["pace_adj_ppg_diff"] = h_ppg_adj - a_ppg_adj
df["pace_adj_opp_ppg_diff"] = h_opp_adj - a_opp_adj

print(f"  Pace-adjusted PPG diff: mean={df['pace_adj_ppg_diff'].mean():.2f}, std={df['pace_adj_ppg_diff'].std():.2f}")

# ═══════════════════════════════════════════════════════════════
# 4. RECENCY FEATURES (last 5 games)
# ═══════════════════════════════════════════════════════════════

print("\n  Computing recency features...")
t0 = time.time()

# Track each team's recent results
team_recent = defaultdict(list)  # team_id → list of (date, won, ppg)

df["home_recent_form_5g"] = np.nan
df["away_recent_form_5g"] = np.nan
df["home_scoring_trend_5g"] = np.nan
df["away_scoring_trend_5g"] = np.nan
df["recent_form_diff"] = np.nan
df["scoring_trend_diff"] = np.nan

for idx, row in df.iterrows():
    season = row.get("season", 0)
    date = row.get("game_date", "")
    
    for side in ["home", "away"]:
        tid = str(row.get(f"{side}_team_id", ""))
        if not tid:
            continue
        
        key = f"{tid}_{season}"
        recent = team_recent[key]
        
        if len(recent) >= 3:
            last5 = recent[-5:] if len(recent) >= 5 else recent
            # Win rate
            win_rate = np.mean([g["won"] for g in last5])
            df.at[idx, f"{side}_recent_form_5g"] = win_rate
            
            # Scoring trend (linear slope of PPG over last 5)
            if len(last5) >= 3:
                scores = [g["score"] for g in last5]
                x = np.arange(len(scores))
                if np.std(scores) > 0:
                    slope = np.polyfit(x, scores, 1)[0]
                    df.at[idx, f"{side}_scoring_trend_5g"] = slope
    
    # Diffs
    h_form = df.at[idx, "home_recent_form_5g"]
    a_form = df.at[idx, "away_recent_form_5g"]
    if pd.notna(h_form) and pd.notna(a_form):
        df.at[idx, "recent_form_diff"] = h_form - a_form
    
    h_trend = df.at[idx, "home_scoring_trend_5g"]
    a_trend = df.at[idx, "away_scoring_trend_5g"]
    if pd.notna(h_trend) and pd.notna(a_trend):
        df.at[idx, "scoring_trend_diff"] = h_trend - a_trend
    
    # Record this game for future lookups
    h_score = row.get("actual_home_score")
    a_score = row.get("actual_away_score")
    if pd.notna(h_score) and pd.notna(a_score):
        h_id = str(row.get("home_team_id", ""))
        a_id = str(row.get("away_team_id", ""))
        h_key = f"{h_id}_{season}"
        a_key = f"{a_id}_{season}"
        team_recent[h_key].append({"date": date, "won": int(h_score > a_score), "score": float(h_score)})
        team_recent[a_key].append({"date": date, "won": int(a_score > h_score), "score": float(a_score)})
    
    if idx > 0 and idx % 10000 == 0:
        print(f"    {idx}/{len(df)} processed...")

form_filled = df["recent_form_diff"].notna().sum()
print(f"  Recent form: {form_filled} games ({form_filled/len(df)*100:.1f}%)")
print(f"  Time: {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# CORRELATION CHECK
# ═══════════════════════════════════════════════════════════════

print(f"\n  === CORRELATION WITH MARGIN ===")
y = df["actual_home_score"] - df["actual_away_score"]

new_features = [
    "h2h_margin_avg", "h2h_games", "h2h_home_win_rate",
    "conf_strength_diff", "cross_conf_flag",
    "pace_adj_ppg_diff", "pace_adj_opp_ppg_diff",
    "recent_form_diff", "scoring_trend_diff",
    "player_rating_diff",  # if computed by player impact script
]

for feat in new_features:
    if feat in df.columns:
        valid = df[feat].notna() & y.notna() & (df[feat] != 0)
        if valid.sum() > 100:
            corr = np.corrcoef(df.loc[valid, feat], y[valid])[0, 1]
            print(f"  {feat:30s} r={corr:.4f}  n={valid.sum()}")

# Compare with existing strong features
print(f"\n  === COMPARISON WITH EXISTING FEATURES ===")
existing = {
    "adj_em_diff": lambda: df["home_adj_em"] - df["away_adj_em"],
    "elo_diff": lambda: df.get("home_elo", 0) - df.get("away_elo", 0),
    "ppg_diff": lambda: df.get("home_ppg", 0) - df.get("away_ppg", 0),
}
for name, fn in existing.items():
    try:
        vals = pd.to_numeric(fn(), errors="coerce")
        valid = vals.notna() & y.notna() & (vals != 0)
        if valid.sum() > 100:
            corr = np.corrcoef(vals[valid], y[valid])[0, 1]
            print(f"  {name:30s} r={corr:.4f}  (existing)")
    except:
        pass

# ═══════════════════════════════════════════════════════════════
# CHECK REDUNDANCY WITH EXISTING FEATURES
# ═══════════════════════════════════════════════════════════════

print(f"\n  === REDUNDANCY CHECK (correlation with existing features) ===")
pairs_to_check = [
    ("h2h_margin_avg", "is_revenge_game"),
    ("recent_form_diff", "form_diff"),
    ("conf_strength_diff", "sos_diff"),
    ("pace_adj_ppg_diff", "ppg_diff"),
    ("scoring_trend_diff", "margin_trend_diff"),
]

for new_f, existing_f in pairs_to_check:
    if new_f in df.columns and existing_f in df.columns:
        n_vals = pd.to_numeric(df[new_f], errors="coerce")
        e_vals = pd.to_numeric(df[existing_f], errors="coerce")
        valid = n_vals.notna() & e_vals.notna() & (n_vals != 0)
        if valid.sum() > 100:
            corr = np.corrcoef(n_vals[valid], e_vals[valid])[0, 1]
            flag = " ← HIGH REDUNDANCY" if abs(corr) > 0.7 else ""
            print(f"  {new_f:25s} × {existing_f:25s} r={corr:.4f}{flag}")

# ═══════════════════════════════════════════════════════════════
# STATS SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n  === FEATURE SUMMARY ===")
all_new = [
    "h2h_margin_avg", "h2h_games", "h2h_home_win_rate",
    "conf_strength_diff", "cross_conf_flag",
    "pace_adj_ppg_diff", "pace_adj_opp_ppg_diff",
    "recent_form_diff", "scoring_trend_diff",
]

for feat in all_new:
    if feat in df.columns:
        vals = df[feat].dropna()
        nonzero = (vals != 0).sum()
        print(f"  {feat:30s} filled={len(vals):>6} nonzero={nonzero:>6} mean={vals.mean():.3f} std={vals.std():.3f}")

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

if DRY_RUN:
    print(f"\n  DRY RUN — not saving.")
else:
    save_path = "ncaa_training_data.parquet"
    df.to_parquet(save_path, index=False)
    print(f"\n  Saved enriched data to {save_path}")

print(f"\n  Done.")
