#!/bin/bash
# prep_mega_sweep.sh — Install LightGBM and verify environment
# Run from sports-predictor-api root before mega_sweep.py

echo "═══════════════════════════════════════════════════════"
echo "  MEGA SWEEP PREP — Installing Dependencies"
echo "═══════════════════════════════════════════════════════"

# Install LightGBM (the new learner we're testing)
echo ""
echo "1. Installing LightGBM..."
pip install lightgbm --break-system-packages -q 2>/dev/null || pip install lightgbm -q
echo "   Done."

# Verify all ML libraries
echo ""
echo "2. Verifying ML libraries..."
python3 -c "
libs = {}
try:
    import xgboost; libs['XGBoost'] = xgboost.__version__
except: libs['XGBoost'] = 'MISSING'
try:
    import catboost; libs['CatBoost'] = catboost.__version__
except: libs['CatBoost'] = 'MISSING'
try:
    import lightgbm; libs['LightGBM'] = lightgbm.__version__
except: libs['LightGBM'] = 'MISSING'
try:
    import sklearn; libs['sklearn'] = sklearn.__version__
except: libs['sklearn'] = 'MISSING'

for name, ver in libs.items():
    status = '✓' if ver != 'MISSING' else '✗'
    print(f'   {status} {name}: {ver}')
"

echo ""
echo "3. Verifying database connection..."
python3 -c "
from db import sb_get
rows = sb_get('nba_predictions', 'result_entered=eq.true&select=id&limit=1')
if rows:
    print('   ✓ Supabase connected')
else:
    print('   ✗ Supabase connection failed')
"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Ready! Run the sweep with:"
echo ""
echo "  python3 mega_sweep.py --sport nba    # NBA only (~10 min)"
echo "  python3 mega_sweep.py --sport ncaa   # NCAA only (~10 min)"
echo "  python3 mega_sweep.py --sport mlb    # MLB only (~15 min)"
echo "  python3 mega_sweep.py --sport all    # All three (~35 min)"
echo "═══════════════════════════════════════════════════════"
