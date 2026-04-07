#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# ALL-SPORTS RETRAIN — MLB + NBA + NCAA (ATS + O/U)
# 
# Verified script inventory (as of April 7, 2026):
#   MLB ATS:  mlb_ensemble_retrain.py  → mlb_model_v8.pkl  → save_model("mlb")
#   MLB O/U:  mlb_retrain_ou_v2.py     → mlb_ou_v2.pkl     → save_model("mlb_ou")
#   NBA ATS:  nba_retrain_v27.py       → save_model("nba")  (built-in upload)
#   NBA O/U:  nba_retrain_ou_v2.py     → save_model("nba_ou") (built-in upload)
#   NCAA ATS: ncaa_final_retrain.py    → save_model("ncaa") (built-in upload)
#   NCAA O/U: ncaa_retrain_ou_v5.py    → save_model("ncaa_ou") (--upload flag)
#
# Usage:
#   ./retrain_all.sh          # All sports
#   ./retrain_all.sh mlb      # MLB only
#   ./retrain_all.sh nba      # NBA only
#   ./retrain_all.sh ncaa     # NCAA only
#   ./retrain_all.sh mlb nba  # MLB + NBA
# ═══════════════════════════════════════════════════════════════

set -e
cd ~/Desktop/sports-predictor-api
source .venv/bin/activate

export SUPABASE_URL="https://lxaaqtqvlwjvyuedyauo.supabase.co"
export SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx4YWFxdHF2bHdqdnl1ZWR5YXVvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTgwNjM1NSwiZXhwIjoyMDg3MzgyMzU1fQ.m9D8hP71LjKnYT3MPxHPtYS4aSD6TX5lMA7286T_L_U"
export SUPABASE_ANON_KEY="$SUPABASE_KEY"

RAILWAY="https://sports-predictor-api-production.up.railway.app"
SPORTS="${@:-mlb nba ncaa}"

echo "═══════════════════════════════════════════════════"
echo "  ALL-SPORTS RETRAIN — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Sports: $SPORTS"
echo "═══════════════════════════════════════════════════"

reload_model() {
    local key=$1
    local result=$(curl -s -X POST "$RAILWAY/debug/reload-model/$key" 2>/dev/null)
    local info=$(echo "$result" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(f'features={d.get(\"features\",\"?\")}, MAE={d.get(\"mae\",\"?\")}')
except: print('error')" 2>/dev/null)
    echo "  ✓ $key reloaded: $info"
}

# ═══════════════════════════════════════════════════
#  MLB
# ═══════════════════════════════════════════════════
if echo "$SPORTS" | grep -q "mlb"; then
    echo ""
    echo "━━━ MLB ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ATS: mlb_ensemble_retrain.py → mlb_model_v8.pkl
    echo "▶ [1/4] MLB ATS — mlb_ensemble_retrain.py"
    python3 mlb_ensemble_retrain.py 2>&1 | tail -15
    echo ""
    python3 -c "
from db import save_model; import joblib
b = joblib.load('mlb_model_v8.pkl')
save_model('mlb', b)
print(f'  ✅ mlb uploaded: {len(b.get(\"feature_cols\",[]))} features, MAE={b.get(\"mae_cv\",\"?\")}')"

    # O/U: mlb_retrain_ou_v2.py → mlb_ou_v2.pkl
    echo ""
    echo "▶ [2/4] MLB O/U — mlb_retrain_ou_v2.py"
    python3 mlb_retrain_ou_v2.py 2>&1 | tail -15
    echo ""
    python3 -c "
from db import save_model; import joblib
b = joblib.load('mlb_ou_v2.pkl')
save_model('mlb_ou', b)
print(f'  ✅ mlb_ou uploaded: {b.get(\"model_type\",\"?\")}')"

    echo ""
    echo "▶ Reloading MLB on Railway..."
    reload_model "mlb"
    reload_model "mlb_ou"
fi

# ═══════════════════════════════════════════════════
#  NBA
# ═══════════════════════════════════════════════════
if echo "$SPORTS" | grep -q "nba"; then
    echo ""
    echo "━━━ NBA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ATS: nba_retrain_v27.py (has built-in save_model("nba"))
    echo "▶ [1/4] NBA ATS — nba_retrain_v27.py"
    python3 nba_retrain_v27.py 2>&1 | tail -15

    # O/U: nba_retrain_ou_v2.py (has built-in save_model("nba_ou"))
    echo ""
    echo "▶ [2/4] NBA O/U — nba_retrain_ou_v2.py"
    python3 nba_retrain_ou_v2.py 2>&1 | tail -15

    echo ""
    echo "▶ Reloading NBA on Railway..."
    reload_model "nba"
    reload_model "nba_ou"
fi

# ═══════════════════════════════════════════════════
#  NCAA
# ═══════════════════════════════════════════════════
if echo "$SPORTS" | grep -q "ncaa"; then
    echo ""
    echo "━━━ NCAA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ATS: ncaa_final_retrain.py (has built-in save_model("ncaa"))
    echo "▶ [1/4] NCAA ATS — ncaa_final_retrain.py"
    python3 ncaa_final_retrain.py 2>&1 | tail -15

    # O/U: ncaa_retrain_ou_v5.py (use --upload flag)
    echo ""
    echo "▶ [2/4] NCAA O/U — ncaa_retrain_ou_v5.py --upload"
    python3 ncaa_retrain_ou_v5.py --upload 2>&1 | tail -15

    echo ""
    echo "▶ Reloading NCAA on Railway..."
    reload_model "ncaa"
    reload_model "ncaa_ou"
fi

# ═══════════════════════════════════════════════════
#  VERIFY ALL
# ═══════════════════════════════════════════════════
echo ""
echo "━━━ FINAL VERIFICATION ━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
for model in ncaa ncaa_ou nba nba_ou mlb mlb_ou; do
    result=$(curl -s "$RAILWAY/debug/reload-model/$model" -X POST 2>/dev/null)
    info=$(echo "$result" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    s=d.get('status','?')
    f=d.get('features','?')
    m=d.get('mae','?')
    print(f'{s} — features={f}, MAE={m}')
except: print('not loaded')" 2>/dev/null)
    echo "  $model: $info"
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ RETRAIN COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Scripts used:"
echo "    MLB ATS:  mlb_ensemble_retrain.py  (Apr 7)"
echo "    MLB O/U:  mlb_retrain_ou_v2.py     (Apr 6)"
echo "    NBA ATS:  nba_retrain_v27.py       (Apr 5)"
echo "    NBA O/U:  nba_retrain_ou_v2.py     (Apr 5)"
echo "    NCAA ATS: ncaa_final_retrain.py    (Apr 4)"
echo "    NCAA O/U: ncaa_retrain_ou_v5.py    (Apr 5)"
echo "═══════════════════════════════════════════════════"
