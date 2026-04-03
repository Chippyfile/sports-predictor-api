#!/usr/bin/env python3
"""
patch_ncaa_serve.py — Apply all fixes to ncaa_full_predict.py
=============================================================
Run: python3 patch_ncaa_serve.py

Fixes:
  1. feature_cols → feature_names fallback (new model key)
  2. hca_pts=3.0 → rolling_hca computed from recent games
  3. SIGMA=16.0 → 6.0
  4. base-10 sigmoid → base-e (correct logistic)
  5. SIGMA_ATS=10.0 → 6.0
  6. ATS base-10 → base-e
  7. Fix neutral_em_diff with rolling_hca
"""
import re

FILE = "ncaa_full_predict.py"

with open(FILE, "r") as f:
    code = f.read()

changes = 0

# ══════════════════════════════════════════════════════════
# FIX 1: feature_cols key fallback
# ══════════════════════════════════════════════════════════
old = '    feature_cols = bundle["feature_cols"]'
new = '    feature_cols = bundle.get("feature_names") or bundle.get("feature_cols")'
if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print(f"  ✅ Fix 1: feature_cols → feature_names fallback")
else:
    print(f"  ⚠️  Fix 1: pattern not found")

# ══════════════════════════════════════════════════════════
# FIX 2: hca_pts=3.0 → rolling_hca from recent games
# ══════════════════════════════════════════════════════════
old_hca = '''    # hca_pts: 0 for neutral is correct, but for non-neutral fill conference HCA
    if "hca_pts" in X.columns and r["hca_pts"] == 0 and not neutral_site:
        X.loc[:, "hca_pts"] = 3.0  # Default HCA'''

new_hca = '''    # ═══ Rolling HCA: compute from team's recent home/away margins ═══
    # Query last 20 home + 20 away games from ncaa_historical
    _rolling_hca_val = 0.0
    if not neutral_site:
        try:
            _home_tid = str(game.get("home_team_id", ""))
            if _home_tid:
                _hca_rows = sb_get("ncaa_historical",
                    f"home_team_id=eq.{_home_tid}&neutral_site=eq.false"
                    f"&actual_home_score=not.is.null&select=actual_home_score,actual_away_score"
                    f"&order=game_date.desc&limit=20")
                _away_rows = sb_get("ncaa_historical",
                    f"away_team_id=eq.{_home_tid}&neutral_site=eq.false"
                    f"&actual_home_score=not.is.null&select=actual_home_score,actual_away_score"
                    f"&order=game_date.desc&limit=20")
                _home_margins = [float(r["actual_home_score"]) - float(r["actual_away_score"]) for r in _hca_rows if r.get("actual_home_score") is not None]
                _away_margins = [float(r["actual_home_score"]) - float(r["actual_away_score"]) for r in _away_rows if r.get("actual_home_score") is not None]
                if len(_home_margins) >= 5 and len(_away_margins) >= 5:
                    import numpy as _np
                    _rolling_hca_val = (_np.mean(_home_margins) - _np.mean([-m for m in _away_margins])) / 2
                    print(f"  [full_predict] rolling_hca={_rolling_hca_val:.2f} (from {len(_home_margins)}H+{len(_away_margins)}A games)")
                else:
                    _rolling_hca_val = 6.6  # League-avg fallback
                    print(f"  [full_predict] rolling_hca=6.6 (fallback, only {len(_home_margins)}H+{len(_away_margins)}A)")
        except Exception as _e:
            _rolling_hca_val = 6.6
            print(f"  [full_predict] rolling_hca=6.6 (error: {_e})")

    # Inject rolling_hca into feature matrix
    if "rolling_hca" in X.columns:
        X.loc[:, "rolling_hca"] = _rolling_hca_val
    elif "rolling_hca" in feature_cols:
        X["rolling_hca"] = _rolling_hca_val

    # Fix neutral_em_diff: strip rolling_hca instead of old static HCA
    if "neutral_em_diff" in X.columns:
        _h_em = float(game.get("home_adj_em", 0) or 0)
        _a_em = float(game.get("away_adj_em", 0) or 0)
        _raw_em = _h_em - _a_em
        X.loc[:, "neutral_em_diff"] = _raw_em - (0 if neutral_site else _rolling_hca_val)

    # Legacy hca_pts: fill if still in feature list (old models)
    if "hca_pts" in X.columns and r["hca_pts"] == 0 and not neutral_site:
        X.loc[:, "hca_pts"] = _rolling_hca_val'''

if old_hca in code:
    code = code.replace(old_hca, new_hca, 1)
    changes += 1
    print(f"  ✅ Fix 2: rolling_hca + neutral_em_diff fix")
else:
    print(f"  ⚠️  Fix 2: hca_pts pattern not found — check manually")

# ══════════════════════════════════════════════════════════
# FIX 3: SIGMA = 16.0 → 6.0
# ══════════════════════════════════════════════════════════
old = '    SIGMA = 16.0'
new = '    SIGMA = 6.0  # Empirically calibrated (Brier-optimal)'
if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print(f"  ✅ Fix 3: SIGMA 16.0 → 6.0")
else:
    print(f"  ⚠️  Fix 3: SIGMA=16.0 not found")

# ══════════════════════════════════════════════════════════
# FIX 4: base-10 sigmoid → base-e (correct logistic)
# ══════════════════════════════════════════════════════════
old = '    margin_prob = 1.0 / (1.0 + 10.0 ** (-margin / SIGMA))'
new = '    import math as _math\n    margin_prob = 1.0 / (1.0 + _math.exp(-margin / SIGMA))'
if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print(f"  ✅ Fix 4: base-10 → base-e sigmoid")
else:
    print(f"  ⚠️  Fix 4: margin_prob base-10 not found")

# ══════════════════════════════════════════════════════════
# FIX 5: SIGMA_ATS = 10.0 → 6.0
# ══════════════════════════════════════════════════════════
old = '                SIGMA_ATS = 10.0'
new = '                SIGMA_ATS = 6.0  # Empirically calibrated'
if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print(f"  ✅ Fix 5: SIGMA_ATS 10.0 → 6.0")
else:
    print(f"  ⚠️  Fix 5: SIGMA_ATS=10.0 not found")

# ══════════════════════════════════════════════════════════
# FIX 6: ATS base-10 → base-e
# ══════════════════════════════════════════════════════════
old = '                ats_cover_prob = round(1.0 / (1.0 + 10.0 ** (-margin_vs_spread / SIGMA_ATS)), 4)'
new = '                ats_cover_prob = round(1.0 / (1.0 + _math.exp(-margin_vs_spread / SIGMA_ATS)), 4)'
if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print(f"  ✅ Fix 6: ATS base-10 → base-e sigmoid")
else:
    print(f"  ⚠️  Fix 6: ATS base-10 not found")

# ══════════════════════════════════════════════════════════
# WRITE
# ══════════════════════════════════════════════════════════
with open(FILE, "w") as f:
    f.write(code)

print(f"\n  {changes}/6 fixes applied to {FILE}")
if changes == 6:
    print(f"  ✅ All fixes applied! Push to Railway.")
else:
    print(f"  ⚠️  Some fixes missing — check manually.")
