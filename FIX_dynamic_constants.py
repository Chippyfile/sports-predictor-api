# ══════════════════════════════════════════════════════════════
# FIX: dynamic_constants.py — NBA_DEFAULT_AVERAGES stale values
# These fallbacks are used when compute_nba_league_averages() fails.
# v20 fixes in nba.py didn't propagate here.
# ══════════════════════════════════════════════════════════════

# BEFORE (current):
NBA_DEFAULT_AVERAGES_OLD = {
    "ppg": 110, "opp_ppg": 110, "fgpct": 0.46, "threepct": 0.36,    # ← 0.36 should be 0.365
    "ftpct": 0.77, "assists": 25, "turnovers": 14, "tempo": 100,
    "orb_pct": 0.25, "fta_rate": 0.28, "ato_ratio": 1.7,            # ← 1.7 should be 1.8
    "steals": 7.5, "blocks": 5.0,
}

# AFTER (fixed):
NBA_DEFAULT_AVERAGES_NEW = {
    "ppg": 113, "opp_ppg": 113,             # Updated: 2024-25 NBA avg is ~113
    "fgpct": 0.471, "threepct": 0.365,      # Fixed: was 0.36
    "ftpct": 0.780, "assists": 25, "turnovers": 14, "tempo": 99.5,
    "orb_pct": 0.245, "fta_rate": 0.270, "ato_ratio": 1.8,   # Fixed: was 1.7
    "steals": 7.5, "blocks": 5.0,
    "opp_fgpct": 0.471, "opp_threepct": 0.365,  # NEW: these were missing
}

# ══════════════════════════════════════════════════════════════
# In dynamic_constants.py, replace lines 213-217 with:
# ══════════════════════════════════════════════════════════════
# NBA_DEFAULT_AVERAGES = {
#     "ppg": 113, "opp_ppg": 113, "fgpct": 0.471, "threepct": 0.365,
#     "ftpct": 0.780, "assists": 25, "turnovers": 14, "tempo": 99.5,
#     "orb_pct": 0.245, "fta_rate": 0.270, "ato_ratio": 1.8,
#     "steals": 7.5, "blocks": 5.0,
#     "opp_fgpct": 0.471, "opp_threepct": 0.365,
# }
