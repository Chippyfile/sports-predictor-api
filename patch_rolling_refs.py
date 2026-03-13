#!/usr/bin/env python3
"""One-pass patch: add rolling player features + referee features to ncaa.py"""
import sys

with open("sports/ncaa.py") as f:
    txt = f.read()

changes = 0

# 1. Add rolling raw columns after importance_multiplier (line 102)
old1 = '        "importance_multiplier": 1.0,\n    }'
new1 = '''        "importance_multiplier": 1.0,
        # Rolling player features (from prior-game averages)
        "home_roll_star1_share": 0.25, "away_roll_star1_share": 0.25,
        "home_roll_top3_share": 0.65, "away_roll_top3_share": 0.65,
        "home_roll_bench_share": 0.20, "away_roll_bench_share": 0.20,
        "home_roll_bench_pts": 15.0, "away_roll_bench_pts": 15.0,
    }'''
if "roll_star1_share" not in txt:
    txt = txt.replace(old1, new1, 1)
    changes += 1
    print("  1. Added rolling raw columns")

# 2. Add rolling + ref differential computations before feature_cols = [
old2 = '    feature_cols = ['
new2 = '''    # Rolling player features (legitimate pre-game signal)
    df["roll_star1_share_diff"] = df["home_roll_star1_share"] - df["away_roll_star1_share"]
    df["roll_top3_share_diff"] = df["home_roll_top3_share"] - df["away_roll_top3_share"]
    df["roll_bench_share_diff"] = df["home_roll_bench_share"] - df["away_roll_bench_share"]
    df["roll_bench_pts_diff"] = df["home_roll_bench_pts"] - df["away_roll_bench_pts"]

    # Referee crew features (from prebuilt profiles)
    _ref_profiles = getattr(ncaa_build_features, "_ref_profiles", {})
    _ref_ou = []
    _ref_hw = []
    _ref_fr = []
    _ref_pace = []
    for _, _row in df.iterrows():
        _ou, _hw, _fr, _pa = [], [], [], []
        for _rc in ["referee_1", "referee_2", "referee_3"]:
            _name = str(_row.get(_rc, "")).strip()
            if _name and _name in _ref_profiles:
                _p = _ref_profiles[_name]
                _ou.append(_p.get("ou_bias", 0))
                _hw.append(_p.get("home_whistle", 0))
                _fr.append(_p.get("foul_rate", 0))
                _pa.append(_p.get("pace_impact", 145))
        _ref_ou.append(float(np.mean(_ou)) if _ou else 0.0)
        _ref_hw.append(float(np.mean(_hw)) if _hw else 0.0)
        _ref_fr.append(float(np.mean(_fr)) if _fr else 0.0)
        _ref_pace.append(float(np.mean(_pa)) - 145.0 if _pa else 0.0)
    df["ref_ou_bias"] = _ref_ou
    df["ref_home_whistle"] = _ref_hw
    df["ref_foul_rate"] = _ref_fr
    df["ref_pace_impact"] = _ref_pace
    df["has_ref_data"] = (pd.Series(_ref_ou) != 0.0).astype(int).values

    feature_cols = ['''
if "roll_star1_share_diff" not in txt:
    txt = txt.replace(old2, new2, 1)
    changes += 1
    print("  2. Added rolling + ref computations")

# 3. Add feature names to feature_cols list after espn_predictor_edge
old3 = '        "espn_wp_edge", "espn_predictor_edge",'
new3 = '''        "espn_wp_edge", "espn_predictor_edge",
        # Rolling player features
        "roll_star1_share_diff", "roll_top3_share_diff",
        "roll_bench_share_diff", "roll_bench_pts_diff",
        # Referee crew features
        "ref_ou_bias", "ref_home_whistle", "ref_foul_rate",
        "ref_pace_impact", "has_ref_data",'''
if "roll_star1_share_diff" not in txt or "ref_ou_bias" not in txt:
    # Only replace in the feature_cols section
    txt = txt.replace(old3, new3, 1)
    changes += 1
    print("  3. Added features to feature_cols list")

# 4. Add ref profile loading before X = ncaa_build_features in train_ncaa
old4 = '        X  = ncaa_build_features(df)'
new4 = '''        # Load referee profiles for feature builder
        try:
            import json as _json
            with open("referee_profiles.json") as _rf:
                ncaa_build_features._ref_profiles = _json.load(_rf)
            print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
        except FileNotFoundError:
            print("  referee_profiles.json not found - ref features zero")
            ncaa_build_features._ref_profiles = {}

        X  = ncaa_build_features(df)'''
if "_ref_profiles" not in txt:
    txt = txt.replace(old4, new4, 1)
    changes += 1
    print("  4. Added ref profile loading in train_ncaa")

print(f"\n  Total: {changes} changes")

# Verify syntax
try:
    compile(txt, "sports/ncaa.py", "exec")
    print("  Syntax: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    sys.exit(1)

with open("sports/ncaa.py", "w") as f:
    f.write(txt)
print("  Written to sports/ncaa.py")
