#!/bin/bash
# NBA Feature Diagnostic — Run these against Railway to find the spread inversion bug
# 
# Usage: bash nba_debug_features.sh
# Requires: curl, jq (optional for pretty-printing)

API="https://sports-predictor-api-production.up.railway.app"

echo "========================================="
echo "  NBA v27 Feature Diagnostic"
echo "========================================="
echo ""

# 1. Check model is loaded correctly
echo "--- 1. Model reload check ---"
curl -s "$API/debug/reload-model/nba" | python3 -m json.tool 2>/dev/null || curl -s "$API/debug/reload-model/nba"
echo ""
echo ""

# 2. PHX @ MEM — the worst inversion (should be PHX -13, model says MEM -2.4)
echo "--- 2. PHX @ MEM full prediction (feature values) ---"
curl -s -X POST "$API/predict/nba/full" \
  -H "Content-Type: application/json" \
  -d '{"home_team":"MEM","away_team":"PHX","game_date":"2026-03-30"}' \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('ML Margin:', data.get('ml_margin'))
print('ML Win Prob Home:', data.get('ml_win_prob_home'))
print('Feature Coverage:', data.get('feature_coverage'))
print()
# Show all feature values sorted by absolute value
feats = data.get('features', data.get('feature_values', {}))
if feats:
    print('--- Feature values (sorted by |value|) ---')
    for k, v in sorted(feats.items(), key=lambda x: abs(x[1] or 0), reverse=True):
        flag = '  ⚠️ ZERO' if v == 0 else ''
        print(f'  {k:35s} = {v:10.4f}{flag}')
else:
    print('No feature values in response — check if endpoint returns them')
print()
# Show diagnostics/warnings
for key in ['diagnostics', 'warnings', 'data_sources', 'diag']:
    if key in data:
        print(f'--- {key} ---')
        print(json.dumps(data[key], indent=2))
" 2>/dev/null
echo ""

# 3. PHI @ MIA — slight inversion
echo "--- 3. PHI @ MIA full prediction ---"
curl -s -X POST "$API/predict/nba/full" \
  -H "Content-Type: application/json" \
  -d '{"home_team":"MIA","away_team":"PHI","game_date":"2026-03-30"}' \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('ML Margin:', data.get('ml_margin'))
print('ML Win Prob Home:', data.get('ml_win_prob_home'))
print('Feature Coverage:', data.get('feature_coverage'))
feats = data.get('features', data.get('feature_values', {}))
if feats:
    zeros = [k for k, v in feats.items() if v == 0]
    nonzeros = [(k, v) for k, v in feats.items() if v != 0]
    print(f'{len(zeros)} zero features, {len(nonzeros)} non-zero')
    print('Top 10 by magnitude:')
    for k, v in sorted(nonzeros, key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f'  {k:35s} = {v:10.4f}')
    if zeros:
        print(f'Zero features: {zeros[:20]}')
" 2>/dev/null
echo ""

# 4. Check rolling stats for MEM and PHX
echo "--- 4. Rolling stats (MEM) ---"
curl -s "$API/nba/test-rolling?team=MEM" | python3 -m json.tool 2>/dev/null || echo "(endpoint may not exist)"
echo ""

echo "--- 5. Enrichment (MEM) ---"
curl -s "$API/nba/test-extract?team=MEM" | python3 -m json.tool 2>/dev/null || echo "(endpoint may not exist)"
echo ""

echo "--- 6. Enrichment (PHX) ---"
curl -s "$API/nba/test-extract?team=PHX" | python3 -m json.tool 2>/dev/null || echo "(endpoint may not exist)"
echo ""

echo "========================================="
echo "  KEY THINGS TO CHECK:"
echo "========================================="
echo "1. Is ml_margin NEGATIVE for PHX@MEM? (means MEM favored = BUG)"
echo "2. How many features are zero? (>20 zeros = data pipeline broken)"
echo "3. What is elo_diff? (should be PHX >> MEM)"  
echo "4. What is win_pct_diff? (should be strongly negative = PHX favored)"
echo "5. What is net_rating_diff? (should be negative = PHX better)"
echo "6. Are enrichment/rolling tables populated for both teams?"
echo "7. Check TRAIN_RANGES clamping — any features being clamped to wrong range?"
