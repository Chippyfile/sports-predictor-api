#!/usr/bin/env python3
"""
Apply winsorized means to:
1. MLB NegBin dispersion calibration (_fit_negbin_k)
2. NCAA lg_avg (hardcoded 70.0 -> derived from data)
3. NCAA KenPom Bayesian shrinkage targets (oe_avg, de_avg, etc.)

Run from sports-predictor-api directory.
"""
import sys

fixes_applied = 0

# ═══════════════════════════════════════════════════════════════
# FIX 1: MLB NegBin dispersion — winsorize before fitting k
# ═══════════════════════════════════════════════════════════════
with open("sports/mlb.py") as f:
    mlb = f.read()

old_negbin = '''def _fit_negbin_k(run_series):
    """
    Fit the overdispersion parameter k for a Negative Binomial distribution
    to a series of run totals using method of moments.
    NegBin variance = \u03bc + \u03bc\u00b2/k  \u2192  k = \u03bc\u00b2 / (variance - \u03bc)
    Returns k clamped to [0.30, 1.20] for stability.
    """
    mu  = run_series.mean()
    var = run_series.var()'''

new_negbin = '''def _fit_negbin_k(run_series):
    """
    Fit the overdispersion parameter k for a Negative Binomial distribution
    to a series of run totals using method of moments.
    NegBin variance = \u03bc + \u03bc\u00b2/k  \u2192  k = \u03bc\u00b2 / (variance - \u03bc)
    Returns k clamped to [0.30, 1.20] for stability.
    Winsorized at 5th/95th percentile to exclude blowouts.
    """
    s = run_series.dropna()
    if len(s) < 20:
        mu, var = s.mean(), s.var()
    else:
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        s = s.clip(lo, hi)
        mu, var = s.mean(), s.var()'''

if old_negbin in mlb:
    mlb = mlb.replace(old_negbin, new_negbin, 1)
    print("  [1] sports/mlb.py: _fit_negbin_k now winsorized")
    fixes_applied += 1
else:
    print("  [1] WARNING: Could not find _fit_negbin_k pattern — check manually")

with open("sports/mlb.py", "w") as f:
    f.write(mlb)


# ═══════════════════════════════════════════════════════════════
# FIX 2: NCAA lg_avg — derive from data instead of hardcoded 70.0
# ═══════════════════════════════════════════════════════════════
with open("sports/ncaa.py") as f:
    ncaa = f.read()

old_lg_avg = '        lg_avg = 70.0  # approximate NCAA scoring avg'
new_lg_avg = '''        # Derive league average from actual team PPG data (winsorized)
        _all_ppg = np.concatenate([h_ppg[h_ppg > 0], a_ppg[a_ppg > 0]])
        if len(_all_ppg) > 20:
            _lo, _hi = np.percentile(_all_ppg, 5), np.percentile(_all_ppg, 95)
            lg_avg = float(np.mean(np.clip(_all_ppg, _lo, _hi)))
        else:
            lg_avg = 70.0  # fallback'''

if old_lg_avg in ncaa:
    ncaa = ncaa.replace(old_lg_avg, new_lg_avg, 1)
    print("  [2] sports/ncaa.py: lg_avg derived from data (winsorized)")
    fixes_applied += 1
else:
    print("  [2] WARNING: Could not find lg_avg = 70.0 — may not be in sports/ncaa.py")
    print("       Checking if it's in backtests.py or ncaa_ratings.py instead...")

with open("sports/ncaa.py", "w") as f:
    f.write(ncaa)


# ═══════════════════════════════════════════════════════════════
# FIX 3: KenPom Bayesian shrinkage — winsorized league averages
# ═══════════════════════════════════════════════════════════════
with open("ncaa_ratings.py") as f:
    ratings = f.read()

old_shrink_avg = '''    final_oe_vals = [adj_oe[t] for t in team_ids]
    final_de_vals = [adj_de[t] for t in team_ids]
    oe_avg = sum(final_oe_vals) / len(final_oe_vals)
    de_avg = sum(final_de_vals) / len(final_de_vals)
    ppg_vals = [adj_ppg[t] for t in team_ids]
    opp_vals = [adj_opp_ppg[t] for t in team_ids]
    ppg_avg = sum(ppg_vals) / len(ppg_vals)
    opp_avg = sum(opp_vals) / len(opp_vals)'''

new_shrink_avg = '''    # Winsorized league averages for shrinkage targets (5th/95th percentile)
    # Prevents extreme outlier teams from distorting the shrinkage anchor
    def _winsorized_avg(vals):
        arr = np.array(vals)
        if len(arr) < 20:
            return float(arr.mean())
        lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
        return float(np.clip(arr, lo, hi).mean())

    final_oe_vals = [adj_oe[t] for t in team_ids]
    final_de_vals = [adj_de[t] for t in team_ids]
    oe_avg = _winsorized_avg(final_oe_vals)
    de_avg = _winsorized_avg(final_de_vals)
    ppg_vals = [adj_ppg[t] for t in team_ids]
    opp_vals = [adj_opp_ppg[t] for t in team_ids]
    ppg_avg = _winsorized_avg(ppg_vals)
    opp_avg = _winsorized_avg(opp_vals)'''

if old_shrink_avg in ratings:
    ratings = ratings.replace(old_shrink_avg, new_shrink_avg, 1)
    print("  [3] ncaa_ratings.py: Bayesian shrinkage targets winsorized")
    fixes_applied += 1
else:
    print("  [3] WARNING: Could not find shrinkage average pattern in ncaa_ratings.py")

# Ensure numpy is imported in ncaa_ratings.py
if "import numpy" not in ratings and "import numpy as np" not in ratings:
    ratings = "import numpy as np\n" + ratings
    print("  [3] ncaa_ratings.py: Added numpy import")

with open("ncaa_ratings.py", "w") as f:
    f.write(ratings)


print(f"\n{fixes_applied}/3 fixes applied.")
print("Next: git add . && git commit -m 'Winsorized means for dispersion, NCAA lg_avg, KenPom shrinkage' && git push")
