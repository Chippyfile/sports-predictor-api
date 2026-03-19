#!/bin/bash
# Run from ~/Desktop/sports-predictor-api/

# 1. Remove tracked files that shouldn't be in the repo
git rm --cached FIX_dynamic_constants.py
git rm --cached build_player_ratings.py
git rm --cached check_parquet_columns.py
git rm --cached dump_nba_training_data.py
git rm --cached enrich_nba_historical.py
git rm --cached enrich_nba_v2.py
git rm --cached espn_nba_summary_pregame_raw.json
git rm --cached espn_nba_summary_raw.json
git rm --cached explore_espn_nba_summary.py
git rm --cached ingest_mgm_odds.py
git rm --cached nba_build_features_v21.py
git rm --cached nba_build_features_v22.py
git rm --cached nba_elo.py
git rm --cached nba_elo_ratings.json
git rm --cached nba_elo_snapshots.parquet
git rm --cached nba_feature_parity_map.py
git rm --cached nba_model_local.pkl
git rm --cached nba_predictions_all.parquet
git rm --cached nba_scrape_checkpoint.json
git rm --cached nba_training_data.parquet
git rm --cached retrain_nba.py
git rm --cached scrape_nba_summaries.py

# 2. Append to .gitignore if not already there
cat >> .gitignore << 'EOF'

# ── Data files (too large for git, not needed on Railway) ──
*.parquet
*.pkl
*_raw.json
nba_elo_ratings.json
nba_scrape_checkpoint.json
conf_tourney_cache.json

# ── Local dev/analysis scripts (not needed on Railway) ──
FIX_*.py
check_*.py
build_player_ratings.py
dump_nba_training_data.py
enrich_nba*.py
explore_*.py
ingest_mgm_odds.py
nba_build_features_v2*.py
nba_feature_parity_map.py
retrain_nba.py
scrape_nba_summaries.py
nba_elo.py
compute_advanced_features.py
fix_conf_tourney_and_pyth.py
EOF

echo ""
echo "Done. Now run:"
echo "  git add .gitignore"
echo "  git commit -m 'remove accidentally committed data/dev files'"
echo "  git push"
