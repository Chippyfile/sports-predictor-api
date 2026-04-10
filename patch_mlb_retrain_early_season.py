#!/usr/bin/env python3
"""
patch_mlb_retrain_early_season.py — Apply early-season fixes to mlb_retrain.py
===============================================================================
1. Removes the April 15 game filter (include all regular season games)
2. Adds input clamps in load_data() so ALL downstream feature builders see them
3. Ensures train/serve parity for early-season predictions

Usage:
    cd ~/Desktop/sports-predictor-api
    python3 patch_mlb_retrain_early_season.py
    # Then retrain:
    python3 mlb_retrain_ou_v3.py --upload
    python3 mlb_v9_retrain.py --upload
"""

FILE = "mlb_retrain.py"

with open(FILE, "r") as f:
    content = f.read()

changes = 0

# ═══════════════════════════════════════════════════════════════
# CHANGE 1: Remove April 15 filter in load_data()
# ═══════════════════════════════════════════════════════════════
old_filter = '''    # ── Filter: skip games before April 15 of any season ──
    # Early-season stats are unreliable (small sample, cold weather, roster flux)
    n_before = len(df)
    apr15_mask = ~((df["game_date_dt"].dt.month < 4) |
                   ((df["game_date_dt"].dt.month == 4) & (df["game_date_dt"].dt.day < 15)))
    df = df[apr15_mask].copy()
    n_early = n_before - len(df)
    if n_early > 0:
        print(f"  Dropped {n_early} games before April 15 (early-season noise)")

    df = df.drop(columns=["game_date_dt"])'''

new_filter = '''    # ── Early-season games INCLUDED (clamps handle OOD inputs for train/serve parity) ──
    # Previously filtered Apr 1-14; now kept so model learns "uncertain = small edge".

    df = df.drop(columns=["game_date_dt"])

    # ── Clamp early-season stats to training-realistic ranges ──
    # Same clamps applied at serve time in mlb_full_predict.py step 4f.
    # This ensures the model trains on identical feature distributions.
    _CLAMPS = {
        "home_sp_fip": (2.5, 6.5), "away_sp_fip": (2.5, 6.5),
        "home_fip": (2.5, 6.5), "away_fip": (2.5, 6.5),
        "home_bullpen_era": (2.5, 6.5), "away_bullpen_era": (2.5, 6.5),
        "home_woba": (0.260, 0.370), "away_woba": (0.260, 0.370),
        "home_k9": (5.0, 13.0), "away_k9": (5.0, 13.0),
        "home_bb9": (1.5, 5.5), "away_bb9": (1.5, 5.5),
    }
    clamped_count = 0
    for col, (lo, hi) in _CLAMPS.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            before = ((vals < lo) | (vals > hi)).sum()
            df[col] = vals.clip(lo, hi)
            clamped_count += before
    if clamped_count > 0:
        print(f"  Clamped {clamped_count} out-of-range values (early-season + outliers)")'''

if old_filter in content:
    content = content.replace(old_filter, new_filter)
    changes += 1
    print(f"  ✅ Change 1: Removed April 15 filter, added clamps in load_data()")
else:
    # Try alternative text matching
    if "april 15" in content.lower() or "apr15" in content.lower():
        print(f"  ⚠ Change 1: Found April 15 reference but exact text didn't match.")
        print(f"    Manual edit needed — search for 'April 15' in {FILE}")
    else:
        print(f"  ⚠ Change 1: April 15 filter not found (may already be removed)")

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
if changes > 0:
    with open(FILE, "w") as f:
        f.write(content)
    print(f"\n  ✅ Saved {FILE} with {changes} changes")
    print(f"\n  Next steps:")
    print(f"    1. python3 mlb_retrain_ou_v3.py          # Evaluate O/U v3 with early-season data")
    print(f"    2. python3 mlb_v9_retrain.py             # Evaluate ATS v9 with early-season data")
    print(f"    3. If results hold: add --upload to both")
else:
    print(f"\n  No changes applied — check {FILE} manually")
