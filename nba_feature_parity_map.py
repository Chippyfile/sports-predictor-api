"""
NCAA → NBA Feature Parity Mapping
Every single NCAA feature mapped to NBA equivalent.

NCAA model: 131 features (CatBoost solo, MAE 8.830)
NBA current: 50 features (CatBoost solo, MAE 10.872)
NBA target: 120+ features

Categories:
  ✅ HAVE = already in NBA v21 (50 features)
  🔧 COMPUTE = can derive from existing data (no new sources)
  📡 ESPN = need ESPN /summary scrape
  📊 EXTERNAL = need player CSVs or Basketball-Reference
  ❌ N/A = NCAA-specific, no NBA equivalent
"""

# ══════════════════════════════════════════════════════════════
# FULL NCAA FEATURE LIST (131 features from pkl)
# → NBA equivalent + data requirement
# ══════════════════════════════════════════════════════════════

FEATURE_MAP = {
    # ── CATEGORY 1: CORE EFFICIENCY (NCAA has KenPom, NBA needs equivalent) ──
    "neutral_em_diff":      {"nba": "net_rtg_diff",         "status": "✅ HAVE",   "note": "NBA net rating ≈ KenPom EM"},
    "hca_pts":              {"nba": "hca_pts",              "status": "🔧 COMPUTE", "note": "Conference-specific HCA (NBA: division-specific)"},
    "neutral":              {"nba": "neutral",              "status": "🔧 COMPUTE", "note": "Neutral site flag — rare in NBA but exists"},
    "adj_oe_diff":          {"nba": "adj_oe_diff",          "status": "🔧 COMPUTE", "note": "Offensive efficiency diff (have PPG + pace)"},
    "adj_de_diff":          {"nba": "adj_de_diff",          "status": "🔧 COMPUTE", "note": "Defensive efficiency diff (have OPP_PPG + pace)"},

    # ── CATEGORY 2: RAW STAT DIFFERENTIALS ──
    "ppg_diff":             {"nba": "ppg_diff",             "status": "✅ HAVE",   "note": ""},
    "opp_ppg_diff":         {"nba": "opp_ppg_diff",         "status": "✅ HAVE",   "note": "SHAP ~0 — subsumed by net_rtg_diff"},
    "fgpct_diff":           {"nba": "fgpct_diff",           "status": "✅ HAVE",   "note": ""},
    "threepct_diff":        {"nba": "threepct_diff",        "status": "✅ HAVE",   "note": "SHAP #4 at 6.3%"},
    "orb_pct_diff":         {"nba": "orb_pct_diff",         "status": "✅ HAVE",   "note": ""},
    "fta_rate_diff":        {"nba": "fta_rate_diff",        "status": "✅ HAVE",   "note": ""},
    "ato_diff":             {"nba": "ato_ratio_diff",       "status": "✅ HAVE",   "note": ""},
    "def_fgpct_diff":       {"nba": "opp_fgpct_diff",      "status": "✅ HAVE",   "note": ""},
    "steals_diff":          {"nba": "steals_diff",          "status": "✅ HAVE",   "note": ""},
    "blocks_diff":          {"nba": "blocks_diff",          "status": "✅ HAVE",   "note": ""},

    # ── CATEGORY 3: ADVANCED SHOOTING ──
    "efg_diff":             {"nba": "efg_diff",             "status": "✅ HAVE",   "note": ""},
    "twopt_diff":           {"nba": "twopt_diff",           "status": "🔧 COMPUTE", "note": "twopt% = (FGM - 3PM) / (FGA - 3PA) — need FGA/3PA"},
    "three_rate_diff":      {"nba": "three_rate_diff",      "status": "🔧 COMPUTE", "note": "3PA / FGA — need three_att_rate"},
    "assist_rate_diff":     {"nba": "assist_rate_diff",     "status": "🔧 COMPUTE", "note": "AST / FGM — derivable from assists + ppg/fgpct"},
    "drb_pct_diff":         {"nba": "drb_pct_diff",         "status": "🔧 COMPUTE", "note": "DRB / (DRB + opp ORB) — have def_reb"},
    "ppp_diff":             {"nba": "ppp_diff",             "status": "🔧 COMPUTE", "note": "Points per possession — PPG / pace"},
    "opp_efg_diff":         {"nba": "opp_efg_diff",         "status": "🔧 COMPUTE", "note": "Opponent eFG% — from opp_fgpct + opp_threepct"},
    "opp_to_rate_diff":     {"nba": "opp_to_rate_diff",     "status": "🔧 COMPUTE", "note": "Forced turnovers / opponent possessions"},
    "opp_fta_rate_diff":    {"nba": "opp_fta_rate_diff",    "status": "🔧 COMPUTE", "note": "Opponent FTA rate allowed"},
    "opp_orb_pct_diff":     {"nba": "opp_orb_pct_diff",     "status": "🔧 COMPUTE", "note": "Opponent offensive rebounding % allowed"},

    # ── CATEGORY 4: MARKET/BETTING ──
    "mkt_spread":           {"nba": "market_spread",        "status": "✅ HAVE",   "note": "SHAP #3 at 10.8%"},
    "mkt_total":            {"nba": "market_total",         "status": "✅ HAVE",   "note": ""},
    "mkt_spread_vs_model":  {"nba": "spread_vs_market",     "status": "✅ HAVE",   "note": "SHAP #2 at 14.9%"},
    "has_mkt":              {"nba": "has_market",           "status": "✅ HAVE",   "note": ""},
    "spread_regime":        {"nba": "spread_regime",        "status": "🔧 COMPUTE", "note": "Bucketed spread (heavy fav/slight fav/pick/underdog)"},

    # ── CATEGORY 5: CONTEXT ──
    "sos_diff":             {"nba": "sos_diff",             "status": "✅ HAVE",   "note": "Currently all 0.5 — need real SOS computation"},
    "form_diff":            {"nba": "form_diff",            "status": "✅ HAVE",   "note": "Enriched ✅"},
    "rank_diff":            {"nba": "seed_diff",            "status": "🔧 COMPUTE", "note": "NBA standings seed instead of KenPom rank"},
    "win_pct_diff":         {"nba": "win_pct_diff",         "status": "✅ HAVE",   "note": "Enriched ✅"},
    "to_margin_diff":       {"nba": "to_margin_diff",       "status": "✅ HAVE",   "note": ""},
    "tempo_avg":            {"nba": "tempo_avg",            "status": "✅ HAVE",   "note": "Enriched from PPG/FG%"},
    "season_phase":         {"nba": "season_phase",         "status": "🔧 COMPUTE", "note": "0-1 scaled by game date within season"},
    "is_conf_tourney":      {"nba": "is_playoff",           "status": "✅ HAVE",   "note": "Currently all 0 — need tagging"},
    "is_early":             {"nba": "is_early_season",      "status": "🔧 COMPUTE", "note": "First 20 games of season flag"},
    "importance":           {"nba": "importance",           "status": "🔧 COMPUTE", "note": "Playoff race tightness / elimination game"},
    "is_midweek":           {"nba": "is_midweek",           "status": "🔧 COMPUTE", "note": "Tuesday/Wednesday/Thursday game"},

    # ── CATEGORY 6: ELO ──
    "elo_diff":             {"nba": "elo_diff",             "status": "✅ HAVE",   "note": "SHAP #9 at 2.6%"},

    # ── CATEGORY 7: MOMENTUM/FORM ──
    "luck_diff":            {"nba": "luck_diff",            "status": "🔧 COMPUTE", "note": "W% - Pythagorean W% (close game over/underperformance)"},
    "consistency_diff":     {"nba": "consistency_diff",     "status": "🔧 COMPUTE", "note": "Scoring variance (std dev of margins)"},
    "pyth_residual_diff":   {"nba": "pyth_residual_diff",   "status": "🔧 COMPUTE", "note": "Actual W% - Expected W% from point diff"},
    "margin_trend_diff":    {"nba": "margin_trend_diff",    "status": "🔧 COMPUTE", "note": "Slope of margin over last 10 games"},
    "margin_accel_diff":    {"nba": "margin_accel_diff",    "status": "🔧 COMPUTE", "note": "Change in margin trend (2nd derivative)"},
    "opp_adj_form_diff":    {"nba": "opp_adj_form_diff",    "status": "🔧 COMPUTE", "note": "Form weighted by opponent quality"},
    "wl_momentum_diff":     {"nba": "wl_momentum_diff",     "status": "🔧 COMPUTE", "note": "Weighted recent W/L streak (exponential decay)"},
    "recovery_diff":        {"nba": "recovery_diff",        "status": "🔧 COMPUTE", "note": "Games since last loss / bounce-back rate"},
    "after_loss_either":    {"nba": "after_loss_either",    "status": "🔧 COMPUTE", "note": "Is either team coming off a loss?"},
    "streak_diff":          {"nba": "streak_diff",          "status": "🔧 COMPUTE", "note": "Current W/L streak length diff"},
    "days_since_loss_diff": {"nba": "days_since_loss_diff", "status": "🔧 COMPUTE", "note": "Calendar days since last loss"},
    "games_since_blowout_diff": {"nba": "games_since_blowout_diff", "status": "🔧 COMPUTE", "note": "Games since last 15+ pt loss"},
    "games_last_14_diff":   {"nba": "games_last_14_diff",   "status": "🔧 COMPUTE", "note": "Games played in last 14 days (schedule density)"},
    "momentum_halflife_diff": {"nba": "momentum_halflife_diff", "status": "🔧 COMPUTE", "note": "Exponentially weighted margin trend"},
    "win_aging_diff":       {"nba": "win_aging_diff",       "status": "🔧 COMPUTE", "note": "Recency-weighted win rate"},
    "overreaction_diff":    {"nba": "overreaction_diff",    "status": "🔧 COMPUTE", "note": "Spread vs model gap after big wins/losses"},
    "regression_diff":      {"nba": "regression_diff",      "status": "🔧 COMPUTE", "note": "W% regression toward Pythagorean expectation"},
    "info_gain_diff":       {"nba": "info_gain_diff",       "status": "🔧 COMPUTE", "note": "How much information last 5 games provide (entropy)"},
    "is_lookahead":         {"nba": "is_lookahead",         "status": "🔧 COMPUTE", "note": "Next game is against a top-10 opponent (letdown)"},

    # ── CATEGORY 8: SCORING DISTRIBUTION ──
    "ceiling_diff":         {"nba": "ceiling_diff",         "status": "🔧 COMPUTE", "note": "90th percentile scoring margin"},
    "floor_diff":           {"nba": "floor_diff",           "status": "🔧 COMPUTE", "note": "10th percentile scoring margin"},
    "margin_skew_diff":     {"nba": "margin_skew_diff",     "status": "🔧 COMPUTE", "note": "Skewness of scoring margin distribution"},
    "scoring_entropy_diff": {"nba": "scoring_entropy_diff", "status": "🔧 COMPUTE", "note": "Entropy of scoring distribution (predictability)"},
    "bimodal_diff":         {"nba": "bimodal_diff",         "status": "🔧 COMPUTE", "note": "Bimodality of margin distribution"},
    "scoring_var_diff":     {"nba": "scoring_var_diff",     "status": "🔧 COMPUTE", "note": "Variance of scoring margin"},
    "score_kurtosis_diff":  {"nba": "score_kurtosis_diff",  "status": "🔧 COMPUTE", "note": "Kurtosis of scoring margin distribution"},
    "scoring_source_entropy_diff": {"nba": "scoring_source_entropy_diff", "status": "📊 EXTERNAL", "note": "Need player-level scoring breakdown"},
    "ft_dependency_diff":   {"nba": "ft_dependency_diff",   "status": "🔧 COMPUTE", "note": "FT points as % of total points"},
    "three_value_diff":     {"nba": "three_value_diff",     "status": "🔧 COMPUTE", "note": "3PT points as % of total points"},
    "concentration_diff":   {"nba": "concentration_diff",   "status": "📊 EXTERNAL", "note": "Top scorer's share of team points — need player stats"},
    "to_conversion_diff":   {"nba": "to_conversion_diff",   "status": "🔧 COMPUTE", "note": "Steals that lead to points (steals × transition efficiency)"},
    "three_divergence_diff": {"nba": "three_divergence_diff", "status": "🔧 COMPUTE", "note": "3P% vs league avg divergence"},
    "ppp_divergence_diff":  {"nba": "ppp_divergence_diff",  "status": "🔧 COMPUTE", "note": "PPP vs league avg divergence"},

    # ── CATEGORY 9: DEFENSE ADVANCED ──
    "def_stability_diff":   {"nba": "def_stability_diff",   "status": "🔧 COMPUTE", "note": "Variance of opponent scoring (consistent defense)"},
    "opp_suppression_diff": {"nba": "opp_suppression_diff", "status": "🔧 COMPUTE", "note": "Points below opponent's season avg"},
    "def_versatility_diff": {"nba": "def_versatility_diff", "status": "🔧 COMPUTE", "note": "Defense performance vs different opponent types"},
    "steal_foul_diff":      {"nba": "steal_foul_diff",      "status": "🔧 COMPUTE", "note": "Steals per foul — efficient defense metric"},
    "block_foul_diff":      {"nba": "block_foul_diff",      "status": "🔧 COMPUTE", "note": "Blocks per foul — rim protection efficiency"},
    "transition_dep_diff":  {"nba": "transition_dep_diff",  "status": "📊 EXTERNAL", "note": "Fast break points dependency — need play type data"},
    "paint_pts_diff":       {"nba": "paint_pts_diff",       "status": "📊 EXTERNAL", "note": "Points in the paint — need play type data"},
    "fastbreak_diff":       {"nba": "fastbreak_diff",       "status": "📊 EXTERNAL", "note": "Fast break points — need play type data"},

    # ── CATEGORY 10: REST/FATIGUE ──
    "fatigue_diff":         {"nba": "fatigue_diff",         "status": "🔧 COMPUTE", "note": "Composite fatigue: rest + travel + schedule density"},
    "rest_effect_diff":     {"nba": "rest_effect_diff",     "status": "🔧 COMPUTE", "note": "Performance change based on rest days"},
    "season_pct_avg":       {"nba": "season_pct_avg",       "status": "🔧 COMPUTE", "note": "Average season completion % of both teams"},

    # ── CATEGORY 11: INTERACTIONS ──
    "fatigue_x_quality":    {"nba": "fatigue_x_quality",    "status": "🔧 COMPUTE", "note": "Fatigue × opponent quality interaction"},
    "rest_x_defense":       {"nba": "rest_x_defense",       "status": "🔧 COMPUTE", "note": "Rest advantage × defensive quality"},
    "form_x_familiarity":   {"nba": "form_x_familiarity",   "status": "🔧 COMPUTE", "note": "Form × style familiarity interaction"},
    "consistency_x_spread": {"nba": "consistency_x_spread", "status": "🔧 COMPUTE", "note": "Consistency × spread regime interaction"},

    # ── CATEGORY 12: MATCHUP-SPECIFIC ──
    "matchup_efg":          {"nba": "matchup_efg",          "status": "🔧 COMPUTE", "note": "eFG% vs opponent's defensive eFG% allowed"},
    "matchup_to":           {"nba": "matchup_to",           "status": "🔧 COMPUTE", "note": "TO rate vs opponent's forced TO rate"},
    "matchup_orb":          {"nba": "matchup_orb",          "status": "🔧 COMPUTE", "note": "ORB% vs opponent's ORB% allowed"},
    "matchup_ft":           {"nba": "matchup_ft",           "status": "🔧 COMPUTE", "note": "FTA rate vs opponent's FTA rate allowed"},
    "style_familiarity":    {"nba": "style_familiarity",    "status": "🔧 COMPUTE", "note": "Pace similarity between teams"},
    "pace_leverage":        {"nba": "pace_leverage",        "status": "🔧 COMPUTE", "note": "Faster team benefits from pace mismatch"},
    "pace_control_diff":    {"nba": "pace_control_diff",    "status": "🔧 COMPUTE", "note": "Which team controls the pace"},
    "pace_adj_margin_diff": {"nba": "pace_adj_margin_diff", "status": "✅ HAVE",   "note": ""},
    "common_opp_diff":      {"nba": "common_opp_diff",      "status": "🔧 COMPUTE", "note": "Performance diff against common opponents"},
    "n_common_opps":        {"nba": "n_common_opps",        "status": "🔧 COMPUTE", "note": "Number of common opponents"},
    "venue_advantage":      {"nba": "venue_advantage",      "status": "🔧 COMPUTE", "note": "Home record strength / altitude factor (Denver)"},

    # ── CATEGORY 13: REFEREE ──
    "ref_home_whistle":     {"nba": "ref_home_whistle",     "status": "📡 ESPN",   "note": "Need ESPN /summary officials + build profiles"},
    "ref_ou_bias":          {"nba": "ref_ou_bias",          "status": "📡 ESPN",   "note": "Ref's avg total vs league avg"},
    "ref_foul_rate":        {"nba": "ref_foul_rate",        "status": "📡 ESPN",   "note": "Ref's avg fouls per game"},
    "ref_pace_impact":      {"nba": "ref_pace_impact",      "status": "📡 ESPN",   "note": "Ref's avg pace vs league avg"},

    # ── CATEGORY 14: ROLLING GAME-LEVEL STATS ──
    "roll_star1_share_diff":  {"nba": "roll_star1_share_diff",  "status": "📊 EXTERNAL", "note": "Top scorer's scoring share (last 10) — need box scores"},
    "roll_top3_share_diff":   {"nba": "roll_top3_share_diff",   "status": "📊 EXTERNAL", "note": "Top 3 scorers' share — need box scores"},
    "roll_bench_share_diff":  {"nba": "roll_bench_share_diff",  "status": "📊 EXTERNAL", "note": "Bench scoring share — need box scores"},
    "roll_bench_pts_diff":    {"nba": "roll_bench_pts_diff",    "status": "📊 EXTERNAL", "note": "Bench points per game — need box scores"},
    "roll_run_diff":          {"nba": "roll_run_diff",          "status": "📡 ESPN",     "note": "Scoring runs from PBP — ESPN /summary plays[]"},
    "roll_drought_diff":      {"nba": "roll_drought_diff",      "status": "📡 ESPN",     "note": "Scoring droughts — ESPN /summary plays[]"},
    "roll_lead_change_avg":   {"nba": "roll_lead_change_avg",   "status": "📡 ESPN",     "note": "Lead changes per game — ESPN /summary plays[]"},
    "roll_dominance_diff":    {"nba": "roll_dominance_diff",    "status": "📡 ESPN",     "note": "Time spent in lead — ESPN /summary plays[]"},
    "roll_star_dep_diff":     {"nba": "roll_star_dep_diff",     "status": "📊 EXTERNAL", "note": "Star dependency index — need player box scores"},
    "roll_top3_dep_diff":     {"nba": "roll_top3_dep_diff",     "status": "📊 EXTERNAL", "note": "Top-3 dependency — need player box scores"},
    "roll_bench_diff":        {"nba": "roll_bench_diff",        "status": "📊 EXTERNAL", "note": "Bench depth — need player box scores"},
    "roll_rotation_diff":     {"nba": "roll_rotation_diff",     "status": "📊 EXTERNAL", "note": "Rotation size — need player minutes"},
    "roll_hhi_diff":          {"nba": "roll_hhi_diff",          "status": "📊 EXTERNAL", "note": "Herfindahl-Hirschman scoring concentration"},
    "roll_clutch_ft_diff":    {"nba": "roll_clutch_ft_diff",    "status": "📡 ESPN",     "note": "FT% in close games — ESPN /summary plays[]"},
    "roll_garbage_diff":      {"nba": "roll_garbage_diff",      "status": "📡 ESPN",     "note": "Garbage time performance — ESPN /summary plays[]"},
    "roll_ats_diff_gated":    {"nba": "roll_ats_diff_gated",    "status": "🔧 COMPUTE", "note": "ATS performance (gated by has_market)"},
    "roll_ats_margin_gated":  {"nba": "roll_ats_margin_gated",  "status": "🔧 COMPUTE", "note": "ATS margin (gated by has_market)"},
    "has_ats_data":           {"nba": "has_ats_data",           "status": "🔧 COMPUTE", "note": "Flag for ATS data availability"},
    "clutch_ratio_diff":      {"nba": "clutch_ratio_diff",      "status": "📡 ESPN",     "note": "Close game win rate — need game margins"},
    "garbage_adj_ppp_diff":   {"nba": "garbage_adj_ppp_diff",   "status": "📡 ESPN",     "note": "PPP excluding garbage time"},

    # ── CATEGORY 15: ESPN SIGNAL ──
    "espn_wp_edge":         {"nba": "espn_wp_edge",         "status": "📡 ESPN",   "note": "ESPN BPI prediction vs model — /summary predictor"},
    "crowd_pct":            {"nba": "crowd_pct",            "status": "📡 ESPN",   "note": "Attendance / capacity — /summary gameInfo"},

    # ── CATEGORY 16: INJURY ──
    # (already in v21 but all zeros for historical — will activate with ESPN scrape)

    # ── CATEGORY 17: NBA-ONLY FEATURES (not in NCAA) ──
    # These are ADDITIONAL features NBA can have that NCAA can't
    "NBA_ONLY_lineup_value_diff":     {"nba": "lineup_value_diff",     "status": "📊 EXTERNAL", "note": "Sum of starter ratings diff — BIGGEST NBA EDGE"},
    "NBA_ONLY_lineup_missing_value":  {"nba": "lineup_missing_value",  "status": "📊 EXTERNAL", "note": "Value of injured/resting players"},
    "NBA_ONLY_matchup_history":       {"nba": "matchup_h2h_diff",      "status": "📡 ESPN",     "note": "Head-to-head season series — /summary seasonseries"},
    "NBA_ONLY_espn_pregame_wp":       {"nba": "espn_pregame_wp",       "status": "📡 ESPN",     "note": "ESPN's BPI pre-game probability"},
    "NBA_ONLY_altitude":              {"nba": "altitude_factor",        "status": "🔧 COMPUTE", "note": "Denver = 5280ft, affects visiting teams"},
    "NBA_ONLY_timezone_travel":       {"nba": "timezone_diff",          "status": "🔧 COMPUTE", "note": "East→West or West→East timezone shift"},
}


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from collections import Counter

    status_counts = Counter(v["status"] for v in FEATURE_MAP.values())
    total = len(FEATURE_MAP)

    print("=" * 70)
    print("  NCAA → NBA Feature Parity Analysis")
    print("=" * 70)
    print(f"\n  NCAA features: 131")
    print(f"  NBA v21 current: 50")
    print(f"  Total mapped (including NBA-only): {total}")
    print()

    for status, count in sorted(status_counts.items()):
        pct = count / total * 100
        print(f"  {status:15s}  {count:3d} features  ({pct:.0f}%)")

    # Features we can add with NO new data sources
    compute_only = [k for k, v in FEATURE_MAP.items() if v["status"] == "🔧 COMPUTE"]
    print(f"\n  🔧 COMPUTE ({len(compute_only)} features) — derivable from existing parquet data:")
    for f in compute_only:
        note = FEATURE_MAP[f]["note"]
        nba_name = FEATURE_MAP[f]["nba"]
        print(f"    {nba_name:35s}  {note}")

    # Features needing ESPN scrape
    espn_only = [k for k, v in FEATURE_MAP.items() if v["status"] == "📡 ESPN"]
    print(f"\n  📡 ESPN ({len(espn_only)} features) — need ESPN /summary historical scrape:")
    for f in espn_only:
        note = FEATURE_MAP[f]["note"]
        nba_name = FEATURE_MAP[f]["nba"]
        print(f"    {nba_name:35s}  {note}")

    # Features needing external data
    external_only = [k for k, v in FEATURE_MAP.items() if v["status"] == "📊 EXTERNAL"]
    print(f"\n  📊 EXTERNAL ({len(external_only)} features) — need player stats CSVs or Basketball-Reference:")
    for f in external_only:
        note = FEATURE_MAP[f]["note"]
        nba_name = FEATURE_MAP[f]["nba"]
        print(f"    {nba_name:35s}  {note}")

    # What this gives us
    have = len([v for v in FEATURE_MAP.values() if v["status"] == "✅ HAVE"])
    compute = len([v for v in FEATURE_MAP.values() if v["status"] == "🔧 COMPUTE"])
    espn = len([v for v in FEATURE_MAP.values() if v["status"] == "📡 ESPN"])
    external = len([v for v in FEATURE_MAP.values() if v["status"] == "📊 EXTERNAL"])

    print(f"\n{'='*70}")
    print(f"  ROADMAP TO 120+ FEATURES:")
    print(f"  Phase 1 (now):          {have} features  ← current v21")
    print(f"  Phase 2 (+compute):     {have + compute} features  ← from existing parquet, NO new data")
    print(f"  Phase 3 (+ESPN scrape): {have + compute + espn} features  ← ESPN /summary historical scrape")
    print(f"  Phase 4 (+player data): {have + compute + espn + external} features  ← player CSVs")
    print(f"{'='*70}")

    print(f"\n  DATA YOU NEED TO FIND:")
    print(f"  1. Player game logs CSV (2021-2026): pts, reb, ast, min, starter flag per game")
    print(f"     → Unlocks: {len([k for k in external_only if 'player' in FEATURE_MAP[k]['note'].lower() or 'box score' in FEATURE_MAP[k]['note'].lower() or 'scoring' in FEATURE_MAP[k]['note'].lower()])} features")
    print(f"  2. ESPN /summary scrape for historical games (officials, PBP, injuries)")
    print(f"     → Unlocks: {espn} features")
    print(f"  3. Everything else is ALREADY in your parquet — just need computation")
    print(f"     → Unlocks: {compute} features (FREE, no new data)")
