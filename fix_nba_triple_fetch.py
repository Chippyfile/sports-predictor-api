#!/usr/bin/env python3
"""
Fix: compute_nba_league_averages was being called 3x during NBA training.
The apply_dynamic.py script inserted the call before EVERY 'X  = nba_build_features(df)'
line. We only need it once — before the first call.

Also adds in-memory caching to dynamic_constants.py so repeated calls are free.
"""

# Fix 1: Add caching to dynamic_constants.py
with open("dynamic_constants.py") as f:
    dc = f.read()

old_nba_fn = "def compute_nba_league_averages():"
new_nba_fn = """_nba_cache = None
def compute_nba_league_averages():
    global _nba_cache
    if _nba_cache is not None:
        return _nba_cache"""

if old_nba_fn in dc and "_nba_cache" not in dc:
    dc = dc.replace(old_nba_fn, new_nba_fn, 1)
    # Add cache store before the return
    dc = dc.replace(
        '    print(f"  NBA dynamic averages from {len(df)} historical games:")',
        '    _nba_cache = averages  # Cache for subsequent calls\n'
        '    print(f"  NBA dynamic averages from {len(df)} historical games:")'
    )
    print("  dynamic_constants.py: Added caching to compute_nba_league_averages")

# Same for MLB
old_mlb_fn = "def compute_mlb_season_constants():"
new_mlb_fn = """_mlb_cache = None
def compute_mlb_season_constants():
    global _mlb_cache
    if _mlb_cache is not None:
        return _mlb_cache"""

if old_mlb_fn in dc and "_mlb_cache" not in dc:
    dc = dc.replace(old_mlb_fn, new_mlb_fn, 1)
    dc = dc.replace(
        '    print(f"  MLB dynamic constants: {n_seasons} seasons derived from {len(df)} games")',
        '    _mlb_cache = constants  # Cache for subsequent calls\n'
        '    print(f"  MLB dynamic constants: {n_seasons} seasons derived from {len(df)} games")'
    )
    print("  dynamic_constants.py: Added caching to compute_mlb_season_constants")

with open("dynamic_constants.py", "w") as f:
    f.write(dc)

# Fix 2: Remove duplicate dynamic average blocks from sports/nba.py
with open("sports/nba.py") as f:
    nba = f.read()

# Count how many times the dynamic block appears
dynamic_block = """        # Derive league averages from historical data
        try:
            _nba_lg = compute_nba_league_averages()
            if _nba_lg:
                nba_build_features._league_averages = _nba_lg
                print(f"  Using dynamic NBA averages ({len(_nba_lg)} stats)")
        except Exception as e:
            print(f"  Dynamic NBA averages failed ({e}), using static")

        X  = nba_build_features(df)"""

count = nba.count(dynamic_block)
if count > 1:
    # Keep the first occurrence, replace subsequent ones with just the feature build call
    first = nba.index(dynamic_block)
    after_first = first + len(dynamic_block)
    rest = nba[after_first:]
    rest = rest.replace(dynamic_block, "        X  = nba_build_features(df)")
    nba = nba[:after_first] + rest
    print(f"  sports/nba.py: Removed {count - 1} duplicate dynamic average blocks")
else:
    print(f"  sports/nba.py: Only {count} dynamic block found (OK)")

with open("sports/nba.py", "w") as f:
    f.write(nba)

print("\nDone. git add . && git commit -m 'Fix: cache dynamic averages, remove triple-fetch' && git push")
