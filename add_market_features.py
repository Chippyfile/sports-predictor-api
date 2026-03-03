#!/usr/bin/env python3
"""
Add Vegas market lines as ML features for all three sports.
The market spread is the single strongest public predictor available.

Key design: historical data may not have market lines, so features must
gracefully handle missing values (fillna with 0 for spread, model total for OU).
A `has_market` flag lets the model learn when market data is present vs absent.
"""

# ═══════════════════════════════════════════════════════════════
# NBA: Add market_spread, market_total, spread_vs_model, has_market
# ═══════════════════════════════════════════════════════════════
with open("sports/nba.py") as f:
    nba = f.read()

# Add market feature computation before feature_cols
old_nba_features = '''    # ── Turnover quality ──
    df["to_margin_diff"] = df["away_turnovers"] - df["home_turnovers"]
    df["steals_to_h"] = df["home_steals"] / df["home_turnovers"].clip(0.5)
    df["steals_to_a"] = df["away_steals"] / df["away_turnovers"].clip(0.5)
    df["steals_to_diff"] = df["steals_to_h"] - df["steals_to_a"]

    feature_cols = ['''

new_nba_features = '''    # ── Turnover quality ──
    df["to_margin_diff"] = df["away_turnovers"] - df["home_turnovers"]
    df["steals_to_h"] = df["home_steals"] / df["home_turnovers"].clip(0.5)
    df["steals_to_a"] = df["away_steals"] / df["away_turnovers"].clip(0.5)
    df["steals_to_diff"] = df["steals_to_h"] - df["steals_to_a"]

    # ── Market line features (strongest public predictor) ──
    df["market_spread"] = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(
        df.get("market_ou_total", df.get("ou_total", 0)), errors="coerce"
    ).fillna(0)
    df["has_market"] = ((df["market_spread"] != 0) | (df["market_total"] != 0)).astype(int)
    # Spread difference: model prediction vs market line (positive = model more bullish on home)
    df["spread_vs_market"] = df["score_diff_pred"] - df["market_spread"]

    feature_cols = ['''

if old_nba_features in nba:
    nba = nba.replace(old_nba_features, new_nba_features, 1)
    print("  sports/nba.py: Added market feature computation")
else:
    print("  WARNING: Could not find NBA turnover quality block")

# Add market features to feature_cols list
old_nba_fcols = '''        "rest_diff", "away_travel",
    ]'''
new_nba_fcols = '''        "rest_diff", "away_travel",
        # Market line signal (Vegas spread is strongest public predictor)
        "market_spread", "market_total", "spread_vs_market", "has_market",
    ]'''

if old_nba_fcols in nba:
    nba = nba.replace(old_nba_fcols, new_nba_fcols, 1)
    print("  sports/nba.py: Added market features to feature_cols")
else:
    print("  WARNING: Could not find NBA feature_cols end")

with open("sports/nba.py", "w") as f:
    f.write(nba)


# ═══════════════════════════════════════════════════════════════
# NCAA: Add market_spread, market_total, spread_vs_model, has_market
# ═══════════════════════════════════════════════════════════════
with open("sports/ncaa.py") as f:
    ncaa = f.read()

# Add market computation before feature_cols
old_ncaa_before = '    df["importance"] = pd.to_numeric(df["importance_multiplier"], errors="coerce").fillna(1.0)'
new_ncaa_before = '''    df["importance"] = pd.to_numeric(df["importance_multiplier"], errors="coerce").fillna(1.0)

    # ── Market line features ──
    df["market_spread"] = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(
        df.get("market_ou_total", df.get("ou_total", 0)), errors="coerce"
    ).fillna(0)
    df["has_market"] = ((df["market_spread"] != 0) | (df["market_total"] != 0)).astype(int)
    _ncaa_pred_spread = pd.to_numeric(df.get("spread_home", 0), errors="coerce").fillna(0)
    df["spread_vs_market"] = _ncaa_pred_spread - df["market_spread"]'''

if old_ncaa_before in ncaa:
    ncaa = ncaa.replace(old_ncaa_before, new_ncaa_before, 1)
    print("  sports/ncaa.py: Added market feature computation")
else:
    print("  WARNING: Could not find NCAA importance line")

# Add to NCAA feature_cols - find the end of the list
old_ncaa_fcols_end = '''        "rest_diff", "either_b2b",'''
new_ncaa_fcols_end = '''        "rest_diff", "either_b2b",
        # Market line signal
        "market_spread", "market_total", "spread_vs_market", "has_market",'''

# Need to be careful - there might be more after either_b2b
if old_ncaa_fcols_end in ncaa:
    ncaa = ncaa.replace(old_ncaa_fcols_end, new_ncaa_fcols_end, 1)
    print("  sports/ncaa.py: Added market features to feature_cols")
else:
    print("  WARNING: Could not find NCAA feature_cols rest_diff line")

with open("sports/ncaa.py", "w") as f:
    f.write(ncaa)


# ═══════════════════════════════════════════════════════════════
# MLB: Add market_spread, market_total, has_market
# ═══════════════════════════════════════════════════════════════
with open("sports/mlb.py") as f:
    mlb = f.read()

# Add market computation before feature_cols in mlb_build_features
old_mlb_before = '    feature_cols = [\n        # Offensive differential (primary signal)\n        "woba_diff",'
new_mlb_before = '''    # ── Market line features ──
    df["market_spread"] = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(
        df.get("market_ou_total", df.get("ou_total", 0)), errors="coerce"
    ).fillna(0)
    df["has_market"] = ((df["market_spread"] != 0) | (df["market_total"] != 0)).astype(int)
    df["spread_vs_market"] = df["run_diff_pred"] - df["market_spread"]

    feature_cols = [
        # Offensive differential (primary signal)
        "woba_diff",'''

if old_mlb_before in mlb:
    mlb = mlb.replace(old_mlb_before, new_mlb_before, 1)
    print("  sports/mlb.py: Added market feature computation")
else:
    print("  WARNING: Could not find MLB feature_cols start")

# Add market features to MLB feature_cols
old_mlb_fcols_end = '        "platoon_diff", "sp_fip_spread", "both_lineups_confirmed",\n    ]'
new_mlb_fcols_end = '''        "platoon_diff", "sp_fip_spread", "both_lineups_confirmed",
        # Market line signal
        "market_spread", "market_total", "spread_vs_market", "has_market",
    ]'''

if old_mlb_fcols_end in mlb:
    mlb = mlb.replace(old_mlb_fcols_end, new_mlb_fcols_end, 1)
    print("  sports/mlb.py: Added market features to feature_cols")
else:
    print("  WARNING: Could not find MLB feature_cols end")

with open("sports/mlb.py", "w") as f:
    f.write(mlb)


print("\nDone. 4 new features per sport:")
print("  market_spread:     raw Vegas spread (most powerful single feature)")
print("  market_total:      raw Vegas O/U total")
print("  spread_vs_market:  model spread minus Vegas spread (disagreement signal)")
print("  has_market:        binary flag (1 if lines available, 0 if not)")
print("")
print("Note: historical data may have market_spread=0 (no lines).")
print("The has_market flag lets the model learn to weight these features")
print("only when they're available.")
print("")
print("git add . && git commit -m 'Add Vegas market lines as ML features' && git push")
