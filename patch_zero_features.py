#!/usr/bin/env python3
"""
patch_zero_features.py — Compute h2h_avg_margin, conference_game, is_revenge_home
for the NBA training data and update the v27 features parquet.

These 3 features were all-zero because:
  - h2h_avg_margin: stored with _ prefix, excluded from row assembly
  - conference_game: never computed
  - is_revenge_home: never computed

Run from ~/Desktop/sports-predictor-api/:
    python3 patch_zero_features.py
"""
import sys, os, time
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from db import sb_get

NBA_CONFERENCES = {
    "ATL":"East","BOS":"East","BKN":"East","CHA":"East","CHI":"East","CLE":"East",
    "DET":"East","IND":"East","MIA":"East","MIL":"East","NYK":"East","ORL":"East",
    "PHI":"East","TOR":"East","WAS":"East",
    "DAL":"West","DEN":"West","GSW":"West","HOU":"West","LAC":"West","LAL":"West",
    "MEM":"West","MIN":"West","NOP":"West","OKC":"West","PHX":"West","POR":"West",
    "SAC":"West","SAS":"West","UTA":"West",
}

print("=" * 70)
print("  PATCH: h2h_avg_margin, conference_game, is_revenge_home")
print("=" * 70)

# ── 1. Load training parquet ──
parquet_path = "nba_v27_features.parquet"
if not os.path.exists(parquet_path):
    print(f"ERROR: {parquet_path} not found")
    sys.exit(1)

feat_df = pd.read_parquet(parquet_path)
print(f"\n  Loaded {len(feat_df)} rows from {parquet_path}")

# ── 2. Load raw training data (need team abbrs + H2H info) ──
print("  Loading raw training data from nba_historical...")
rows = sb_get(
    "nba_historical",
    "actual_home_score=not.is.null"
    "&select=game_date,season,home_team,away_team,"
    "actual_home_score,actual_away_score,market_spread_home"
    "&order=game_date.asc&limit=50000"
)
if not rows:
    # Try alternate column names
    print("  First query failed, trying select=*...")
    rows = sb_get(
        "nba_historical",
        "actual_home_score=not.is.null&order=game_date.asc&limit=50000"
    )
if not rows:
    print("  ERROR: No rows from nba_historical")
    sys.exit(1)

raw_df = pd.DataFrame(rows)
print(f"  Raw columns: {sorted(raw_df.columns.tolist())[:20]}...")

# Normalize column names — find team abbr columns
team_h_col = next((c for c in raw_df.columns if c in ["home_team", "home_team_abbr"]), None)
team_a_col = next((c for c in raw_df.columns if c in ["away_team", "away_team_abbr"]), None)
if not team_h_col or not team_a_col:
    print(f"  ERROR: Cannot find team columns. Available: {[c for c in raw_df.columns if 'team' in c.lower()]}")
    sys.exit(1)

# Standardize to home_team_abbr / away_team_abbr
raw_df["home_team_abbr"] = raw_df[team_h_col]
raw_df["away_team_abbr"] = raw_df[team_a_col]

for col in ["actual_home_score", "actual_away_score", "market_spread_home", "season"]:
    if col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# Filter to games with market spread (matching feature parquet)
raw_df = raw_df[raw_df["market_spread_home"].notna() & (raw_df["market_spread_home"] != 0)].copy()
raw_df = raw_df.sort_values("game_date").reset_index(drop=True)

print(f"  Raw data: {len(raw_df)} games with spreads")

# ── 3. Compute conference_game ──
print("\n  Computing conference_game...")
raw_df["conference_game"] = raw_df.apply(
    lambda r: 1 if NBA_CONFERENCES.get(r.get("home_team_abbr","")) == NBA_CONFERENCES.get(r.get("away_team_abbr","")) else 0,
    axis=1
)
n_conf = (raw_df["conference_game"] == 1).sum()
print(f"  Conference games: {n_conf}/{len(raw_df)} ({n_conf/len(raw_df)*100:.0f}%)")

# ── 4. Compute H2H avg margin and is_revenge_home ──
print("  Computing h2h_avg_margin and is_revenge_home...")

raw_df["actual_margin"] = raw_df["actual_home_score"] - raw_df["actual_away_score"]
raw_df["h2h_avg_margin"] = 0.0
raw_df["is_revenge_home"] = 0

# Create matchup key (alphabetical pair + season) for grouping
raw_df["_team_pair"] = raw_df.apply(
    lambda r: tuple(sorted([r["home_team_abbr"], r["away_team_abbr"]])), axis=1
)
raw_df["_matchup_key"] = raw_df["_team_pair"].astype(str) + "_" + raw_df["season"].astype(str)

t_h2h = time.time()
processed = 0
for key, group in raw_df.groupby("_matchup_key"):
    if len(group) < 2:
        continue  # first meeting — no prior H2H data

    indices = group.index.tolist()
    for i in range(1, len(indices)):
        idx = indices[i]
        h = raw_df.at[idx, "home_team_abbr"]
        # Compute margins from THIS home team's perspective for all prior meetings
        prior_indices = indices[:i]
        margins = []
        for pi in prior_indices:
            if raw_df.at[pi, "home_team_abbr"] == h:
                margins.append(raw_df.at[pi, "actual_margin"])
            else:
                margins.append(-raw_df.at[pi, "actual_margin"])

        raw_df.at[idx, "h2h_avg_margin"] = round(np.mean(margins), 1)
        raw_df.at[idx, "is_revenge_home"] = 1 if margins[-1] < 0 else 0

    processed += len(group) - 1

print(f"  Computed H2H for {processed} rematch games in {time.time()-t_h2h:.0f}s")

n_h2h = (raw_df["h2h_avg_margin"] != 0).sum()
n_revenge = (raw_df["is_revenge_home"] == 1).sum()
print(f"  H2H avg margin populated: {n_h2h}/{len(raw_df)} ({n_h2h/len(raw_df)*100:.0f}%)")
print(f"  Revenge games: {n_revenge}/{len(raw_df)} ({n_revenge/len(raw_df)*100:.0f}%)")

# ── 5. Align with feature parquet and patch ──
print(f"\n  Aligning {len(raw_df)} raw rows with {len(feat_df)} feature rows...")

# Drop temp columns
raw_df.drop(columns=["_team_pair", "_matchup_key", "actual_margin"], errors="ignore", inplace=True)

# The feature parquet was built in game_date order matching the raw data
# Both should have the same length after the market filter
if len(raw_df) == len(feat_df):
    feat_df["h2h_avg_margin"] = raw_df["h2h_avg_margin"].values
    feat_df["conference_game"] = raw_df["conference_game"].values
    feat_df["is_revenge_home"] = raw_df["is_revenge_home"].values
    print("  Direct alignment (same length)")
elif len(raw_df) > len(feat_df):
    # Parquet might be a subset — use first N rows
    feat_df["h2h_avg_margin"] = raw_df["h2h_avg_margin"].values[:len(feat_df)]
    feat_df["conference_game"] = raw_df["conference_game"].values[:len(feat_df)]
    feat_df["is_revenge_home"] = raw_df["is_revenge_home"].values[:len(feat_df)]
    print(f"  Used first {len(feat_df)} of {len(raw_df)} raw rows")
else:
    print(f"  WARNING: Length mismatch — raw={len(raw_df)}, parquet={len(feat_df)}")
    print(f"  Padding with zeros for extra rows")
    h2h = np.zeros(len(feat_df))
    conf = np.zeros(len(feat_df))
    rev = np.zeros(len(feat_df))
    h2h[:len(raw_df)] = raw_df["h2h_avg_margin"].values
    conf[:len(raw_df)] = raw_df["conference_game"].values
    rev[:len(raw_df)] = raw_df["is_revenge_home"].values
    feat_df["h2h_avg_margin"] = h2h
    feat_df["conference_game"] = conf
    feat_df["is_revenge_home"] = rev

# ── 6. Verify and save ──
print(f"\n  Verification:")
for col in ["h2h_avg_margin", "conference_game", "is_revenge_home"]:
    nz = (feat_df[col] != 0).sum()
    print(f"  {col:30s} nonzero={nz}/{len(feat_df)} ({nz/len(feat_df)*100:.0f}%)")

# Save
feat_df.to_parquet(parquet_path, index=False)
print(f"\n  Saved updated parquet: {parquet_path}")
print(f"  {len(feat_df)} rows × {len(feat_df.columns)} columns (was 102, now {len(feat_df.columns)})")
print(f"\n  Done. Re-run nba_v27_backward_elim.py to include these features.")
