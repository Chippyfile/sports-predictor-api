#!/usr/bin/env python3
"""
Wires dynamic constants into sports/mlb.py and sports/nba.py.
Run from sports-predictor-api directory.
"""

# ── Part 1: MLB — add dynamic derivation to train_mlb ──
with open("sports/mlb.py") as f:
    mlb = f.read()

# Add import at top (after existing imports)
mlb_import = "from dynamic_constants import compute_mlb_season_constants, MLB_DEFAULT_CONSTANTS\n"
if "compute_mlb_season_constants" not in mlb:
    # Insert after the last import line
    marker = "from ml_utils import"
    if marker in mlb:
        mlb = mlb.replace(marker, mlb_import + marker, 1)
        print("  sports/mlb.py: Added dynamic_constants import")

# Replace the static SEASON_CONSTANTS usage in mlb_build_features
# Instead of always using the hardcoded dict, try dynamic first
old_lg_rpg = '''        df["lg_rpg"] = df["season"].map(
            lambda s: SEASON_CONSTANTS.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
            if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
        )'''
new_lg_rpg = '''        # Dynamic league averages (derived from historical data at training time)
        _dyn = getattr(mlb_build_features, "_dynamic_constants", None)
        if _dyn:
            df["lg_rpg"] = df["season"].map(
                lambda s: _dyn.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
                if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
            )
        else:
            df["lg_rpg"] = df["season"].map(
                lambda s: SEASON_CONSTANTS.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
                if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
            )'''

if old_lg_rpg in mlb:
    mlb = mlb.replace(old_lg_rpg, new_lg_rpg, 1)
    print("  sports/mlb.py: Feature builder now checks dynamic constants first")
else:
    print("  WARNING: Could not find lg_rpg pattern in mlb.py — manual edit needed")

# Add dynamic constant computation in train_mlb
# Insert right before the feature build call
old_train_feature = "        X  = mlb_build_features(df)"
new_train_feature = '''        # Derive league constants from historical data
        try:
            _dyn_constants = compute_mlb_season_constants()
            if _dyn_constants:
                mlb_build_features._dynamic_constants = _dyn_constants
                print(f"  Using dynamic MLB constants ({len(_dyn_constants)} seasons)")
        except Exception as e:
            print(f"  Dynamic constants failed ({e}), using static")

        X  = mlb_build_features(df)'''

if old_train_feature in mlb:
    mlb = mlb.replace(old_train_feature, new_train_feature, 1)
    print("  sports/mlb.py: train_mlb computes dynamic constants before feature build")

with open("sports/mlb.py", "w") as f:
    f.write(mlb)


# ── Part 2: NBA — add dynamic league averages to feature builder ──
with open("sports/nba.py") as f:
    nba = f.read()

nba_import = "from dynamic_constants import compute_nba_league_averages, NBA_DEFAULT_AVERAGES\n"
if "compute_nba_league_averages" not in nba:
    marker = "from ml_utils import"
    if marker in nba:
        nba = nba.replace(marker, nba_import + marker, 1)
        print("  sports/nba.py: Added dynamic_constants import")
    elif "from db import" in nba:
        nba = nba.replace("from db import", nba_import + "from db import", 1)
        print("  sports/nba.py: Added dynamic_constants import (before db)")

# Replace hardcoded defaults in nba_build_features
old_defaults = '''    for col_base, default in [
        ("ppg", 110), ("opp_ppg", 110), ("fgpct", 0.46), ("threepct", 0.36),
        ("ftpct", 0.77), ("assists", 25), ("turnovers", 14), ("tempo", 100),
        ("orb_pct", 0.25), ("fta_rate", 0.28), ("ato_ratio", 1.7),
        ("opp_fgpct", 0.46), ("opp_threepct", 0.35),
        ("steals", 7.5), ("blocks", 5.0),
    ]:'''

new_defaults = '''    # Dynamic league averages (derived from nba_historical, falls back to static)
    _nba_avgs = getattr(nba_build_features, "_league_averages", NBA_DEFAULT_AVERAGES)
    for col_base, default in [
        ("ppg", _nba_avgs.get("ppg", 110)),
        ("opp_ppg", _nba_avgs.get("opp_ppg", 110)),
        ("fgpct", _nba_avgs.get("fgpct", 0.46)),
        ("threepct", _nba_avgs.get("threepct", 0.36)),
        ("ftpct", _nba_avgs.get("ftpct", 0.77)),
        ("assists", _nba_avgs.get("assists", 25)),
        ("turnovers", _nba_avgs.get("turnovers", 14)),
        ("tempo", _nba_avgs.get("tempo", 100)),
        ("orb_pct", _nba_avgs.get("orb_pct", 0.25)),
        ("fta_rate", _nba_avgs.get("fta_rate", 0.28)),
        ("ato_ratio", _nba_avgs.get("ato_ratio", 1.7)),
        ("opp_fgpct", _nba_avgs.get("fgpct", 0.46)),
        ("opp_threepct", _nba_avgs.get("threepct", 0.35)),
        ("steals", _nba_avgs.get("steals", 7.5)),
        ("blocks", _nba_avgs.get("blocks", 5.0)),
    ]:'''

if old_defaults in nba:
    nba = nba.replace(old_defaults, new_defaults, 1)
    print("  sports/nba.py: Feature builder uses dynamic league averages")
else:
    print("  WARNING: Could not find NBA defaults pattern — manual edit needed")

# Add dynamic computation in train_nba before feature build
old_nba_feat = "        X  = nba_build_features(df)"
new_nba_feat = '''        # Derive league averages from historical data
        try:
            _nba_lg = compute_nba_league_averages()
            if _nba_lg:
                nba_build_features._league_averages = _nba_lg
                print(f"  Using dynamic NBA averages ({len(_nba_lg)} stats)")
        except Exception as e:
            print(f"  Dynamic NBA averages failed ({e}), using static")

        X  = nba_build_features(df)'''

if old_nba_feat in nba:
    nba = nba.replace(old_nba_feat, new_nba_feat, 1)
    print("  sports/nba.py: train_nba computes dynamic averages before feature build")

with open("sports/nba.py", "w") as f:
    f.write(nba)


print("\nDone. Deploy files:")
print("  1. Copy dynamic_constants.py to your repo root")
print("  2. git add . && git commit -m 'Dynamic league averages from historical data' && git push")
print("  3. Retrain: curl -s -X POST .../train/nba && curl -s -X POST .../train/mlb")
