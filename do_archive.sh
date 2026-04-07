#!/bin/bash
cd ~/Desktop/sports-predictor-api
mkdir -p archive

echo "Moving 194 files to archive/..."

# Space-named file
mv "NBA referee 2013_2026 - Sheet1.csv" archive/ 2>/dev/null

# Batch 1: misc data/logs
for f in all_odds.csv allplayers.csv arch_sweep_results.txt backfill_2026_checkpoint.json backfill_advanced_features.py backfill_log.json bracket_results.json bracket_results_frontend.json build_travel_distance.py calibration_curve.json calibration_map.json capture_pickcenter.py check_imports.sh cleanup_repo.sh clutch_ft.log clutch_ft_cache.json compute_rolling_hca.py conf_fix.log conf_tourney_cache.json conference_lookup.json covers_to_espn_mapping.json daily_injury_update.py depth_sweep_results.txt enrich_nba_v2.py enrich_nba_v3.py env.example espn_nba_summary_pregame_raw.json espn_nba_summary_raw.json extract_clutch_ft.py gameinfo.csv gitignore gl2021.txt h2h_lookup.json hhi_fix.log injury_update_log.json lineup_sweep_results.txt mae_cv; do
  [ -e "$f" ] && mv "$f" archive/
done

# Batch 2: MLB old
for f in mlb-odds-2014.xlsx mlb-odds-2015.xlsx mlb-odds-2016.xlsx mlb-odds-2017.xlsx mlb-odds-2018.xlsx mlb-odds-2019.xlsx mlb-odds-2020.xlsx mlb-odds-2021.xlsx mlb_backfill_advanced.py mlb_backfill_final.py mlb_backfill_platoon.py mlb_backfill_rolling.py mlb_backfill_sp_ip.py mlb_backfill_sp_rolling_fip.py mlb_backfill_weather.py mlb_cron_endpoint.py mlb_ensemble_feature_results.json mlb_ensemble_results.json mlb_feature_backfill.py mlb_feature_select_ensemble.py mlb_fix_leakage.py mlb_heuristic_ab_test.py mlb_model_local.pkl mlb_model_select.py mlb_model_selection_results.json mlb_odds_2014_2021.csv mlb_ou_model_v1.pkl mlb_ou_v2_production.py mlb_push_enriched.py mlb_serve_verify.py mlb_training_data_enriched.parquet; do
  [ -e "$f" ] && mv "$f" archive/
done

# Batch 3: NBA old
for f in nba_architecture_sweep.py nba_ats_eliminate.py nba_ats_feature_validation.py nba_audit_fixes.py nba_debug_features.sh nba_elo.py nba_elo_ratings.json nba_elo_snapshots.parquet nba_feature_comparison.py nba_feature_comparison.xlsx nba_lineup_features.parquet nba_model_local.pkl nba_ou_model.pkl nba_pbp_features.parquet nba_pbp_features_v2.parquet nba_pbp_v2_checkpoint.json nba_player_boxscores.parquet nba_predictions_all.parquet nba_pull_checkpoint.json nba_referee_log.parquet nba_referee_profiles.json nba_scrape_checkpoint.json nba_summary_data.parquet nba_sweep.py; do
  [ -e "$f" ] && mv "$f" archive/
done

# Batch 4: NBA v27/v29 sweep files
for f in nba_v27_arch_sweep.py nba_v27_arch_sweep_results.json nba_v27_audit.pkl nba_v27_backward_elim.py nba_v27_backward_results.json nba_v27_backward_round2.py nba_v27_deploy.py nba_v27_eliminate.py nba_v27_elimination_results.json nba_v27_ensemble.pkl nba_v27_ensemble_sweep.py nba_v27_ensemble_sweep_results.json nba_v27_feature_ranking.csv nba_v27_features.parquet nba_v27_fold_depth_sweep.py nba_v27_lasso_sweep.py nba_v27_lasso_sweep_results.json nba_v27_model_sweep.py nba_v27_model_sweep_results.json nba_v27_phase5_relaxed.py nba_v27_relaxed_results.json nba_v27_round2_results.json nba_v27_train.py nba_v27_verify.py nba_v29_ensemble.pkl nba_walkforward_detail.csv nba_walkforward_results.csv; do
  [ -e "$f" ] && mv "$f" archive/
done

# Batch 5: NCAA old
for f in ncaa-daily-cron.yml ncaa_asym_ou_sweep.py ncaa_ats_v4.pkl ncaa_backfill_historical.py ncaa_backfill_route.py ncaa_boxscore_cache.json ncaa_calibrate_constants.py ncaa_combined_sweep.py ncaa_csv_create_missing.py ncaa_csv_spread_ingest.py ncaa_daily_prefetch.py ncaa_expand_progress.json ncaa_extract_cache.json ncaa_hca_paired.py ncaa_hca_validate.py ncaa_hhi_players_cache.json ncaa_historical_export.csv ncaa_kenpom_sweep.py ncaa_ml_cap_sweep.py ncaa_model_local.pkl ncaa_ou_v4_predict.py ncaa_ou_walkforward.py ncaa_over2u_test.py ncaa_retrain_ou_v4.py ncaa_sweep.py ncaa_sweep2.py ncaa_team_locations.json ncaa_v4_asym_sweep.py ncaa_venue_cache.json ncaa_verify_all.py ncaab_injuries.json; do
  [ -e "$f" ] && mv "$f" archive/
done

# Batch 6: misc old
for f in no_market_calibration.json no_market_isotonic.joblib odds_api_cache.json odds_backfill.log odds_patches_matched.json patch_ncaa_serve.py patch_zero_features.py pickcenter_cache.json pitching.csv player_ratings.json populate_nba_enrichment.py prep_mega_sweep.sh push_output.log recent_form_lookup.json referee_profiles.json retrain_and_upload.py retrain_and_upload.py.pre_select_fix retrain_nba.py rolling_fix.log rolling_player_avg_w10.csv run_shap_analysis.py scrape_log.txt scrape_ncaab_injuries.py select_cols.txt shap_results_154.csv shap_results_161.csv shap_results.json shap_top20.json sweep_checkpoint_ncaa_20260310_1525.json sweep_checkpoint_ncaa_partial.json sweep_results_ncaa_20260310_1525.csv team_conferences_2026.json team_name_to_abbr.json teamstats.csv true_walkforward_results.csv true_walkforward_summary.json upload_nba_ref_profiles.py walk_forward_no_market_summary.json walk_forward_results.csv walk_forward_summary.json; do
  [ -e "$f" ] && mv "$f" archive/
done

# Batch 7: huge cache files (last)
for f in ncaa_raw_summary_cache.json ncaa_raw_summary_cache.json.bak ncaa_raw_summary_cache.jsonl; do
  [ -e "$f" ] && mv "$f" archive/
done

echo ""
echo "=== Remaining files ==="
find . -maxdepth 1 -type f | grep -v '/\.' | sort | sed 's|./||'
echo ""
echo "Total remaining: $(find . -maxdepth 1 -type f | grep -v '/\.' | wc -l)"
echo "Archive size: $(du -sh archive/ | cut -f1)"
