#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# NCAAB v17 Re-Audit — Deploy, Train, Backtest
# ═══════════════════════════════════════════════════════════════
# Changes in v17 (R1-R10):
#   main.py:              R1(neutral_em_diff), R2(heur_capped), R3(conf_game),
#                         R4(isotonic), R6(bias correction), R7(ElasticNet), R8(interactions)
#   ncaaSync.js:          R5(rest days computed from schedule)
#   NCAACalendarTab.jsx:  R2/R3(pass conference/date), R9(ML-adjusted MC)
#   ncaaUtils.js:         sigma=16.0 (unchanged from v16 calibration)
#   betUtils.js:          sigma=16.0 (unchanged)
# ═══════════════════════════════════════════════════════════════

RAILWAY_API="https://sports-predictor-api-production.up.railway.app"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  NCAAB v17 RE-AUDIT — DEPLOY & TRAIN"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Copy files ──
echo "📦 Step 1: Copy files to repos"
echo ""
echo "  Railway backend:"
echo "    cp main.py ~/sports-predictor-api/main.py"
echo "    cd ~/sports-predictor-api"
echo "    git add main.py && git commit -m 'NCAAB v17: R1-R8 re-audit fixes' && git push"
echo ""
echo "  Vercel frontend:"
echo "    cp ncaaSync.js ~/mlb-predictor/src/sports/ncaa/ncaaSync.js"
echo "    cp ncaaUtils.js ~/mlb-predictor/src/sports/ncaa/ncaaUtils.js"
echo "    cp NCAACalendarTab.jsx ~/mlb-predictor/src/sports/ncaa/NCAACalendarTab.jsx"
echo "    cp betUtils.js ~/mlb-predictor/src/utils/betUtils.js"
echo "    cd ~/mlb-predictor"
echo "    git add -A && git commit -m 'NCAAB v17: R2/R3/R5/R9 frontend wiring' && git push"
echo ""
echo "  ⚠️  Supabase: home_rest_days and away_rest_days columns will auto-create"
echo "     on first UPSERT. No manual schema change needed."
echo ""
read -p "Press ENTER once both deploys complete..."
echo ""

# ── Step 2: Health check ──
echo "🔍 Step 2: Health check..."
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "${RAILWAY_API}/health" 2>/dev/null)
if [ "$HEALTH" = "200" ]; then echo "  ✅ Railway healthy"; else echo "  ❌ HTTP $HEALTH"; fi
echo ""

# ── Step 3: Train ──
echo "🏀 Step 3: Training NCAAB v17 model..."
echo "  (29 features, ElasticNet stacking, bias correction, isotonic calibration)"
echo ""

curl -s -X POST "${RAILWAY_API}/train/ncaa" --max-time 180 | python3 -c "
import json, sys
d = json.load(sys.stdin)
if d.get('status') == 'trained':
    print(f'  ✅ Trained: {d[\"n_train\"]} games, MAE={d[\"mae_cv\"]:.3f}')
    print(f'     Model: {d[\"model_type\"]}')
    print(f'     Features: {len(d[\"features\"])}')
    print(f'     Bias correction: {d.get(\"bias_correction\", \"N/A\")}')
    print(f'     Meta weights: {d.get(\"meta_weights\", \"N/A\")}')
else:
    print(f'  ⚠️  {d}')
" 2>/dev/null
echo ""

# ── Step 4: Backtest ──
echo "📊 Step 4: Walk-forward backtest..."
echo "  ⏳ 3-5 minutes..."
echo ""

BACKTEST=$(curl -s -X POST "${RAILWAY_API}/backtest/ncaa" \
  -H "Content-Type: application/json" \
  -d '{"min_train": 200}' \
  --max-time 600 2>/dev/null)

echo "$BACKTEST" | python3 -c "
import json, sys
d = json.load(sys.stdin)
agg = d.get('aggregate', {})
sigma = d.get('sigma_calibration', {})
tiers = d.get('confidence_tiers', [])

ml = agg.get('ml_overall_accuracy', 0)
h = agg.get('heur_overall_accuracy', 0)
n = agg.get('total_games_tested', 0)

print(f'  Games tested:      {n}')
print(f'  ML accuracy:       {ml:.1%} (was 73.5% in v16)')
print(f'  Heuristic acc:     {h:.1%}')
print(f'  ML lift:           {(ml-h)*100:+.1f}%')
print(f'  ML Brier:          {agg.get(\"ml_overall_brier\", \"?\")}')
print(f'  ML MAE margin:     {agg.get(\"ml_overall_mae_margin\", \"?\")}')
print()

if tiers:
    print('  Confidence tiers:')
    for t in tiers:
        print(f'    {t[\"min_confidence\"]:>3s}+: {t[\"accuracy\"]:.1%} ({t[\"n_games\"]} games)')
    print()

if sigma:
    print(f'  Sigma calibration: optimal={sigma.get(\"optimal_sigma\")}, brier={sigma.get(\"brier_at_optimal\")}')
    rec = sigma.get('recommendation', '')
    if 'SIGMA' in rec:
        print(f'  → {rec}')
" 2>/dev/null
echo ""

# ── Step 5: Train other sports ──
echo "🏈⚾ Step 5: Training other sports..."
for SPORT in mlb nba nfl ncaaf; do
    R=$(curl -s -X POST "${RAILWAY_API}/train/${SPORT}" --max-time 180 2>/dev/null)
    S=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null)
    N=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('n_train','?'))" 2>/dev/null)
    echo "  ${SPORT}: ${S} (n=${N})"
done
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  After deploying, run a Full Backfill from the NCAA Calendar"
echo "  tab to populate rest_days and regenerate predictions."
echo ""
echo "  Quick commands:"
echo "    curl -X POST ${RAILWAY_API}/train/ncaa | python3 -m json.tool"
echo "    curl -X POST ${RAILWAY_API}/backtest/ncaa -d '{\"min_train\":200}' | python3 -m json.tool"
echo ""
