#!/bin/bash
# ──────────────────────────────────────────────────────────────
# local_train.sh — Weekly local training for sports prediction models
# 
# Replaces Railway's /cron/auto-train endpoint.
# Trains all in-season sports locally (no timeout limits),
# saves models to Supabase, and logs results.
#
# Usage:
#   ./local_train.sh              # Train all in-season sports
#   ./local_train.sh nba          # Train NBA only
#   ./local_train.sh mlb nba      # Train MLB and NBA
#   ./local_train.sh all          # Force all 5 sports
#   ./local_train.sh --calibrate  # Also calibrate MLB dispersion
#
# Setup:
#   1. Place in sports-predictor-api/ repo root
#   2. chmod +x local_train.sh
#   3. Run manually or add to crontab:
#      0 4 * * 1 cd ~/Desktop/sports-predictor-api && ./local_train.sh >> logs/train.log 2>&1
#      (Every Monday at 4 AM)
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export SUPABASE_URL="https://lxaaqtqvlwjvyuedyauo.supabase.co"
export SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx4YWFxdHF2bHdqdnl1ZWR5YXVvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTgwNjM1NSwiZXhwIjoyMDg3MzgyMzU1fQ.m9D8hP71LjKnYT3MPxHPtYS4aSD6TX5lMA7286T_L_U"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# ── Determine which sports to train ──
CALIBRATE=false
SPORTS=()

for arg in "$@"; do
    if [[ "$arg" == "--calibrate" ]]; then
        CALIBRATE=true
    elif [[ "$arg" == "all" ]]; then
        SPORTS=(mlb nba ncaa nfl ncaaf)
    else
        SPORTS+=("$arg")
    fi
done

# Default: auto-detect in-season sports
if [[ ${#SPORTS[@]} -eq 0 ]]; then
    MONTH=$(date +%m)
    case $MONTH in
        01|02|03)  SPORTS=(nba ncaa) ;;          # NBA + March Madness
        04|05|06)  SPORTS=(nba mlb) ;;            # NBA playoffs + MLB early
        07|08)     SPORTS=(mlb) ;;                # MLB midseason
        09)        SPORTS=(mlb nfl ncaaf) ;;      # MLB late + Football starts
        10|11)     SPORTS=(nfl ncaaf nba ncaa) ;; # Football + NBA/NCAA tip-off
        12)        SPORTS=(nfl nba ncaa) ;;        # Football + NBA/NCAA
    esac
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Local Model Training — $(date)"
echo "  Sports: ${SPORTS[*]}"
echo "═══════════════════════════════════════════════════════════"

# ── Train each sport ──
RESULTS=""
for sport in "${SPORTS[@]}"; do
    echo ""
    echo "── Training ${sport^^} ──────────────────────────────────"
    START=$(date +%s)
    
    OUTPUT=$(python3 -c "
from main import train_${sport}
import json
result = train_${sport}()
print(json.dumps(result, indent=2, default=str))
" 2>&1) || true
    
    END=$(date +%s)
    DURATION=$((END - START))
    
    echo "$OUTPUT"
    echo "  Duration: ${DURATION}s"
    
    # Extract key metrics
    STATUS=$(echo "$OUTPUT" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('{',1)[1].rsplit('}',1)[0].join(['{','}'])); print(d.get('status','error'))" 2>/dev/null || echo "error")
    MAE=$(echo "$OUTPUT" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('{',1)[1].rsplit('}',1)[0].join(['{','}'])); print(d.get('mae_cv','?'))" 2>/dev/null || echo "?")
    N_TRAIN=$(echo "$OUTPUT" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('{',1)[1].rsplit('}',1)[0].join(['{','}'])); print(d.get('n_train','?'))" 2>/dev/null || echo "?")
    
    RESULTS="${RESULTS}\n  ${sport^^}: status=${STATUS}, mae=${MAE}, n_train=${N_TRAIN}, time=${DURATION}s"
done

# ── Optional: Calibrate MLB dispersion ──
if [[ "$CALIBRATE" == true ]]; then
    echo ""
    echo "── Calibrating MLB Dispersion ──────────────────────────"
    python3 -c "
from main import calibrate_mlb_dispersion
import json
result = calibrate_mlb_dispersion()
print(json.dumps(result, indent=2, default=str))
" 2>&1 || echo "  Calibration failed (need 30+ completed games)"
fi

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training Complete — $(date)"
echo -e "$RESULTS"
echo "═══════════════════════════════════════════════════════════"

# Save log
echo "Log saved to: $LOG_FILE"
