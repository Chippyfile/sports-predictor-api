#!/usr/bin/env python3
"""
wire_orphaned_features.py — Add ~20 orphaned Supabase columns to ncaa_build_features
═══════════════════════════════════════════════════════════════════════════════════════
These columns were computed, pushed to Supabase, and never wired into the feature builder.

New features (20 differentials + 3 standalone = 23 total):
  1.  adj_oe_diff          — offensive efficiency gap (split from adj_em)
  2.  adj_de_diff          — defensive efficiency gap (lower = better D)
  3.  scoring_var_diff     — scoring volatility difference
  4.  score_kurtosis_diff  — scoring distribution shape difference  
  5.  clutch_ratio_diff    — clutch performance rate difference
  6.  garbage_adj_ppp_diff — garbage-time-adjusted PPP difference
  7.  days_since_loss_diff — confidence/momentum gap
  8.  games_since_blowout_loss_diff — recovery from blowout losses
  9.  games_last_14_diff   — 14-day schedule density difference
  10. rest_effect_diff     — pre-computed rest impact difference
  11. momentum_halflife_diff — momentum decay difference
  12. win_aging_diff       — win quality aging difference
  13. centrality_diff      — schedule network strength difference
  14. dow_effect_diff      — day-of-week performance split difference
  15. conf_balance_diff    — conference balance metric difference
  16. n_common_opps        — number of shared opponents (standalone)
  17. revenge_margin       — prior meeting margin (standalone)
  18. is_lookahead         — lookahead/trap game flag (standalone)
  19. is_postseason        — postseason game flag (standalone)
  20. espn_ml_edge         — ESPN ML implied prob vs model (derived)

Usage:
  python3 wire_orphaned_features.py --dry-run   # preview changes
  python3 wire_orphaned_features.py              # apply to sports/ncaa.py
"""
import sys, re, argparse


def patch_file(filepath, dry_run=False):
    with open(filepath) as f:
        txt = f.read()

    original = txt
    changes = []

    # ═══════════════════════════════════════════════════════════════
    # 1. Add new raw_cols defaults
    # ═══════════════════════════════════════════════════════════════
    # Find the marker near end of raw_cols dict
    marker = '        "importance_multiplier": 1.0,'

    new_defaults = '''
        # ── ORPHANED FEATURES (computed & pushed but never wired) ──
        "home_adj_oe": 105.0, "away_adj_oe": 105.0,
        "home_adj_de": 105.0, "away_adj_de": 105.0,
        "home_scoring_var": 12.0, "away_scoring_var": 12.0,
        "home_score_kurtosis": 0.0, "away_score_kurtosis": 0.0,
        "home_clutch_ratio": 0.5, "away_clutch_ratio": 0.5,
        "home_garbage_adj_ppp": 1.0, "away_garbage_adj_ppp": 1.0,
        "home_days_since_loss": 5, "away_days_since_loss": 5,
        "home_games_since_blowout_loss": 10, "away_games_since_blowout_loss": 10,
        "home_games_last_14": 4, "away_games_last_14": 4,
        "home_rest_effect": 0.0, "away_rest_effect": 0.0,
        "home_momentum_halflife": 1.0, "away_momentum_halflife": 1.0,
        "home_win_aging": 1.0, "away_win_aging": 1.0,
        "home_centrality": 1.0, "away_centrality": 1.0,
        "home_dow_effect": 0.0, "away_dow_effect": 0.0,
        "home_conf_balance": 0.0, "away_conf_balance": 0.0,
        "n_common_opps": 0, "revenge_margin": 0.0,
        "is_lookahead": 0, "is_postseason": 0,
        "espn_ml_home": 0, "espn_ml_away": 0,'''

    if "home_adj_oe" not in txt and marker in txt:
        txt = txt.replace(marker, marker + new_defaults)
        changes.append("Added 20+ orphaned raw column defaults")
    elif "home_adj_oe" in txt:
        changes.append("SKIP: orphaned raw columns already present")

    # ═══════════════════════════════════════════════════════════════
    # 2. Add differential computations before feature_cols = [
    # ═══════════════════════════════════════════════════════════════
    new_computations = '''
    # ── Orphaned features (were in Supabase but not in model) ──
    df["adj_oe_diff"] = df["home_adj_oe"] - df["away_adj_oe"]
    df["adj_de_diff"] = df["away_adj_de"] - df["home_adj_de"]  # flipped: lower D is better
    df["scoring_var_diff"] = df["home_scoring_var"] - df["away_scoring_var"]
    df["score_kurtosis_diff"] = df["home_score_kurtosis"] - df["away_score_kurtosis"]
    df["clutch_ratio_diff"] = df["home_clutch_ratio"] - df["away_clutch_ratio"]
    df["garbage_adj_ppp_diff"] = df["home_garbage_adj_ppp"] - df["away_garbage_adj_ppp"]
    df["days_since_loss_diff"] = df["home_days_since_loss"] - df["away_days_since_loss"]
    df["games_since_blowout_diff"] = df["home_games_since_blowout_loss"] - df["away_games_since_blowout_loss"]
    df["games_last_14_diff"] = df["home_games_last_14"] - df["away_games_last_14"]
    df["rest_effect_diff"] = df["home_rest_effect"] - df["away_rest_effect"]
    df["momentum_halflife_diff"] = df["home_momentum_halflife"] - df["away_momentum_halflife"]
    df["win_aging_diff"] = df["home_win_aging"] - df["away_win_aging"]
    df["centrality_diff"] = df["home_centrality"] - df["away_centrality"]
    df["dow_effect_diff"] = df["home_dow_effect"] - df["away_dow_effect"]
    df["conf_balance_diff"] = df["home_conf_balance"] - df["away_conf_balance"]
    # ESPN moneyline edge: convert to implied prob delta vs model
    _espn_ml_h = pd.to_numeric(df["espn_ml_home"], errors="coerce").fillna(0)
    _espn_ml_a = pd.to_numeric(df["espn_ml_away"], errors="coerce").fillna(0)
    _has_espn_ml = (_espn_ml_h != 0).astype(float)
    _espn_imp_h = np.where(_espn_ml_h > 0, 100 / (_espn_ml_h + 100), -_espn_ml_h / (-_espn_ml_h + 100))
    _espn_imp_a = np.where(_espn_ml_a > 0, 100 / (_espn_ml_a + 100), -_espn_ml_a / (-_espn_ml_a + 100))
    _espn_vig_total = _espn_imp_h + _espn_imp_a
    _espn_true_h = np.where(_espn_vig_total > 0, _espn_imp_h / _espn_vig_total, 0.5)
    df["espn_ml_edge"] = np.where(_has_espn_ml, _espn_true_h - 0.5, 0.0)  # deviation from 50%

'''

    feature_cols_marker = '    feature_cols = ['
    if "adj_oe_diff" not in txt and feature_cols_marker in txt:
        txt = txt.replace(feature_cols_marker, new_computations + feature_cols_marker)
        changes.append("Added 16 differential computations + espn_ml_edge")
    elif "adj_oe_diff" in txt:
        changes.append("SKIP: differential computations already present")

    # ═══════════════════════════════════════════════════════════════
    # 3. Add new features to feature_cols list
    # ═══════════════════════════════════════════════════════════════
    # Insert after the last referee feature
    ref_marker = '        "ref_pace_impact", "has_ref_data",'

    new_feature_names = '''
        # ── Orphaned features (newly wired) ──
        "adj_oe_diff", "adj_de_diff",
        "scoring_var_diff", "score_kurtosis_diff",
        "clutch_ratio_diff", "garbage_adj_ppp_diff",
        "days_since_loss_diff", "games_since_blowout_diff",
        "games_last_14_diff", "rest_effect_diff",
        "momentum_halflife_diff", "win_aging_diff",
        "centrality_diff", "dow_effect_diff", "conf_balance_diff",
        "n_common_opps", "revenge_margin",
        "is_lookahead", "is_postseason",
        "espn_ml_edge",'''

    if "adj_oe_diff" not in txt and ref_marker in txt:
        txt = txt.replace(ref_marker, ref_marker + new_feature_names)
        changes.append("Added 20 features to feature_cols list")
    elif "adj_oe_diff" in txt:
        changes.append("SKIP: features already in feature_cols")

    # ═══════════════════════════════════════════════════════════════
    # 4. Also patch the predict function's game dict builder
    # ═══════════════════════════════════════════════════════════════
    # Find the predict function's raw data assembly
    predict_marker = '        "importance_multiplier": game.get("importance_multiplier", 1.0),'

    predict_additions = '''
        # ── Orphaned features ──
        "home_adj_oe": game.get("home_adj_oe", 105.0),
        "away_adj_oe": game.get("away_adj_oe", 105.0),
        "home_adj_de": game.get("home_adj_de", 105.0),
        "away_adj_de": game.get("away_adj_de", 105.0),
        "home_scoring_var": game.get("home_scoring_var", 12.0),
        "away_scoring_var": game.get("away_scoring_var", 12.0),
        "home_score_kurtosis": game.get("home_score_kurtosis", 0.0),
        "away_score_kurtosis": game.get("away_score_kurtosis", 0.0),
        "home_clutch_ratio": game.get("home_clutch_ratio", 0.5),
        "away_clutch_ratio": game.get("away_clutch_ratio", 0.5),
        "home_garbage_adj_ppp": game.get("home_garbage_adj_ppp", 1.0),
        "away_garbage_adj_ppp": game.get("away_garbage_adj_ppp", 1.0),
        "home_days_since_loss": game.get("home_days_since_loss", 5),
        "away_days_since_loss": game.get("away_days_since_loss", 5),
        "home_games_since_blowout_loss": game.get("home_games_since_blowout_loss", 10),
        "away_games_since_blowout_loss": game.get("away_games_since_blowout_loss", 10),
        "home_games_last_14": game.get("home_games_last_14", 4),
        "away_games_last_14": game.get("away_games_last_14", 4),
        "home_rest_effect": game.get("home_rest_effect", 0.0),
        "away_rest_effect": game.get("away_rest_effect", 0.0),
        "home_momentum_halflife": game.get("home_momentum_halflife", 1.0),
        "away_momentum_halflife": game.get("away_momentum_halflife", 1.0),
        "home_win_aging": game.get("home_win_aging", 1.0),
        "away_win_aging": game.get("away_win_aging", 1.0),
        "home_centrality": game.get("home_centrality", 1.0),
        "away_centrality": game.get("away_centrality", 1.0),
        "home_dow_effect": game.get("home_dow_effect", 0.0),
        "away_dow_effect": game.get("away_dow_effect", 0.0),
        "home_conf_balance": game.get("home_conf_balance", 0.0),
        "away_conf_balance": game.get("away_conf_balance", 0.0),
        "n_common_opps": game.get("n_common_opps", 0),
        "revenge_margin": game.get("revenge_margin", 0.0),
        "is_lookahead": game.get("is_lookahead", 0),
        "is_postseason": game.get("is_postseason", 0),
        "espn_ml_home": game.get("espn_ml_home", 0),
        "espn_ml_away": game.get("espn_ml_away", 0),'''

    if "home_adj_oe" not in txt.split("def predict_ncaa")[1] if "def predict_ncaa" in txt else "" and predict_marker in txt:
        txt = txt.replace(predict_marker, predict_marker + predict_additions)
        changes.append("Added orphaned columns to predict_ncaa game dict")

    # ═══════════════════════════════════════════════════════════════
    # Validate
    # ═══════════════════════════════════════════════════════════════
    try:
        compile(txt, filepath, "exec")
        changes.append("Syntax: OK")
    except SyntaxError as e:
        changes.append(f"SYNTAX ERROR: {e}")
        print(f"\n  SYNTAX ERROR — not writing file: {e}")
        return False

    # Report
    print(f"\n  Changes ({len(changes)}):")
    for c in changes:
        print(f"    {'✓' if 'SKIP' not in c and 'ERROR' not in c else '⚠'} {c}")

    if txt == original:
        print("\n  No changes needed.")
        return True

    if dry_run:
        print(f"\n  DRY RUN — no files written.")
        # Count new features
        new_feat_count = len(re.findall(r'"adj_oe_diff"|"adj_de_diff"|"scoring_var"|"score_kurtosis"|"clutch_ratio"|"garbage_adj_ppp"|"days_since_loss"|"games_since_blowout"|"games_last_14"|"rest_effect"|"momentum_halflife"|"win_aging"|"centrality"|"dow_effect"|"conf_balance"|"n_common_opps"|"revenge_margin"|"is_lookahead"|"is_postseason"|"espn_ml_edge"', txt))
        print(f"  Would add ~20 new features (141 → ~161 total)")
        return True

    # Backup + write
    with open(filepath + ".pre_orphaned", "w") as f:
        f.write(original)
    with open(filepath, "w") as f:
        f.write(txt)
    print(f"\n  Backup: {filepath}.pre_orphaned")
    print(f"  Patched: {filepath}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  WIRING ORPHANED SUPABASE FEATURES")
    print("=" * 70)

    ok = patch_file("sports/ncaa.py", dry_run=args.dry_run)

    if ok and not args.dry_run:
        print(f"\n{'=' * 70}")
        print(f"  NEXT STEPS:")
        print(f"  1. Fix db.py pagination: sed -i '' 's/limit = 1000/limit = 500/' db.py")
        print(f"  2. Retrain: python3 -u retrain_and_upload.py")
        print(f"  3. Check feature count is ~161")
        print(f"  4. Deploy: git add . && git push")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
