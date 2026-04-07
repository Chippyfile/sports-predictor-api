#!/bin/bash
cd ~/Desktop/sports-predictor-api

echo "=== Checking each candidate file for references ==="
echo ""

SAFE=""
KEEP=""

check_file() {
  local f="$1"
  [ ! -e "$f" ] && return
  
  local base=$(basename "$f" .py | sed 's/\..*$//')
  local refs=$(grep -rl "$base" *.py sports/*.py 2>/dev/null | grep -v "$f" | grep -v __pycache__ || true)
  local size=$(du -h "$f" | cut -f1)
  
  if [ -n "$refs" ]; then
    echo "  KEEP  $f ($size) — referenced by: $refs"
    KEEP="$KEEP\n  $f"
  else
    echo "  SAFE  $f ($size)"
    SAFE="$SAFE $f"
  fi
}

# Old/duplicate retrain scripts
for f in \
  deploy_and_train_v17.sh \
  local_train.sh \
  mlb_ou_retrain.py \
  mlb_retrain.py \
  nba_ou_retrain.py \
  nba_ou_train_v2.py \
  nba_v27_retrain.py \
  nba_v27_train.py \
  nba_v29_retrain.py \
  ncaa_retrain_hca_travel.py \
  ncaa_retrain_ou_v4.py \
  ncaa_retrain_spread.py \
  ncaa_train_total.py \
  retrain_and_upload.py \
  "retrain_and_upload.py.pre_select_fix" \
  retrain_nba.py \
  training_data_fixes.py \
  dump_nba_training_data.py \
  dump_training_data.py \
; do check_file "$f"; done

echo ""
echo "--- Old pkl files ---"
for f in \
  mlb_ou_model_v1.pkl \
  mlb_model_local.pkl \
  mlb_training_data_enriched.parquet \
  nba_v29_ensemble.pkl \
  nba_v27_ensemble.pkl \
  nba_v27_audit.pkl \
  nba_model_local.pkl \
  nba_ou_model.pkl \
  ncaa_ats_v4.pkl \
  ncaa_model_local.pkl \
  ncaa_ou_model_clean.pkl \
  ncaa_ou_model_v2.pkl \
  ncaa_ou_model_v3.pkl \
  ncaa_ou_model.pkl \
; do check_file "$f"; done

echo ""
echo "--- One-off dev/debug scripts ---"
for f in \
  mlb_ou_v2_eval.py \
  mlb_ou_v2_predict.py \
  mlb_ou_v2_production.py \
  mlb_ou_v2_sweep.py \
  mlb_ou_merge_debug.py \
  mlb_savant_parse.py \
  mlb_new_features.py \
  mlb_feature_backfill.py \
  mlb_backfill_fix.py \
  mlb_add_2021.py \
  nba_parity_audit.py \
  n_train \
  trained_at \
  ncaa_odds_backfill.py \
; do check_file "$f"; done

echo ""
echo "=== Production files (MUST KEEP) ==="
for f in \
  main.py config.py db.py requirements.txt Procfile \
  mlb_ensemble_retrain.py mlb_retrain_ou_v2.py mlb_full_predict.py mlb_ou_v2_serve.py \
  mlb_model_v8.pkl mlb_ou_v2.pkl mlb_training_data.parquet \
  nba_retrain_v27.py nba_retrain_ou_v2.py nba_full_predict.py \
  nba_training_data.parquet nba_ou_v2.pkl \
  ncaa_final_retrain.py ncaa_retrain_ou_v5.py ncaa_full_predict.py \
  ncaa_training_data.parquet ncaa_ou_v4.pkl \
  retrain_all.sh \
; do
  if [ -e "$f" ]; then
    echo "  OK $f"
  else
    echo "  MISSING $f"
  fi
done

echo ""
echo "=== sports/ directory ==="
ls sports/*.py 2>/dev/null

echo ""
echo "=== Safe to delete (copy this command) ==="
echo "rm -f $SAFE"
