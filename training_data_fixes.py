"""
TRAINING DATA FIXES: Address all 4 forensic audit issues.

Run AFTER loading the training data, BEFORE ncaa_build_features().
Patches the DataFrame in-place.

Issue 1: pyth_residual 0% in 2026 → compute inline from ppg/opp_ppg/wins/losses
Issue 2: ESPN spread 0% in 2024-2025, DK spread exists for 2026 → unify sources
Issue 3: Dead features (espn_ml_edge, espn_predictor_edge, consistency_x_spread) → flag for removal
Issue 4: Opening lines exist but unused → create opening_spread column for honest eval

Usage:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
"""

import numpy as np
import pandas as pd


def fix_pyth_residual(df):
    """
    Issue 1: pyth_residual is 0% in 2026 (was 92% in all prior seasons).
    
    Compute inline using the same formula as fix_conf_tourney_and_pyth.py:
        pyth_pct = ppg^EXP / (ppg^EXP + opp_ppg^EXP)
        residual = actual_win_pct - pyth_pct
    
    Also fills 'luck' which is the same concept in KenPom's framework.
    """
    EXP = 11.5
    fixed = 0
    
    for side in ["home", "away"]:
        pyth_col = f"{side}_pyth_residual"
        luck_col = f"{side}_luck"
        ppg_col = f"{side}_ppg"
        opp_col = f"{side}_opp_ppg"
        wins_col = f"{side}_wins"
        losses_col = f"{side}_losses"
        
        # Also check record columns as fallback
        rec_wins = f"{side}_record_wins"
        rec_losses = f"{side}_record_losses"
        
        if pyth_col not in df.columns:
            continue
        
        pyth = pd.to_numeric(df[pyth_col], errors="coerce").fillna(0)
        needs_fix = pyth == 0
        
        if needs_fix.sum() == 0:
            continue
        
        ppg = pd.to_numeric(df[ppg_col], errors="coerce").fillna(75)
        opp_ppg = pd.to_numeric(df[opp_col], errors="coerce").fillna(72)
        
        # Get wins/losses from multiple sources
        wins = pd.to_numeric(df.get(wins_col, 0), errors="coerce").fillna(0)
        losses = pd.to_numeric(df.get(losses_col, 0), errors="coerce").fillna(0)
        
        # Fallback to record columns
        if rec_wins in df.columns:
            rec_w = pd.to_numeric(df[rec_wins], errors="coerce").fillna(0)
            rec_l = pd.to_numeric(df[rec_losses], errors="coerce").fillna(0)
            no_wl = (wins + losses) == 0
            wins = wins.where(~no_wl, rec_w)
            losses = losses.where(~no_wl, rec_l)
        
        total = wins + losses
        actual_pct = wins / total.clip(lower=1)
        
        ppg_exp = ppg ** EXP
        opp_exp = opp_ppg ** EXP
        pyth_pct = ppg_exp / (ppg_exp + opp_exp).clip(lower=1e-10)
        
        residual = actual_pct - pyth_pct
        
        # Only fill where currently 0 AND we have valid inputs
        can_compute = needs_fix & (total > 0) & (ppg > 0) & (opp_ppg > 0)
        
        df.loc[can_compute, pyth_col] = residual[can_compute].round(4)
        fixed += can_compute.sum()
        
        # Also fill luck (same formula in KenPom)
        if luck_col in df.columns:
            luck = pd.to_numeric(df[luck_col], errors="coerce").fillna(0)
            luck_needs_fix = (luck == 0) & can_compute
            df.loc[luck_needs_fix, luck_col] = residual[luck_needs_fix].round(4)
    
    print("  Fix 1 (pyth_residual): patched %d values" % fixed)
    
    # Verify
    for s in sorted(df["season"].unique()):
        mask = df["season"] == s
        n = mask.sum()
        nz = (pd.to_numeric(df.loc[mask, "home_pyth_residual"], errors="coerce").fillna(0) != 0).sum()
        if n > 0:
            print("    %d: %d/%d (%.1f%%)" % (int(s), nz, n, nz/n*100))
    
    return df


def fix_spread_sources(df):
    """
    Issue 2: Unify spread sources across seasons.
    
    Current state:
      2015-2023: espn_spread (65-91%), market_spread_home (12%)
      2024-2025: espn_spread (0%), market_spread_home (11%), odds_api_spread_close (25-27%)
      2026: espn_spread (74%, actually from dk_spread_close), dk_spread_open/close (74%)
    
    Fix: Create unified columns:
      - closing_spread: best available closing line
      - opening_spread: best available opening line (for honest eval)
      - spread_source: which source provided the data
    """
    n = len(df)
    
    # Parse all spread sources
    espn = pd.to_numeric(df.get("espn_spread", 0), errors="coerce").fillna(0)
    mkt = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    dk_close = pd.to_numeric(df.get("dk_spread_close", 0), errors="coerce").fillna(0)
    dk_open = pd.to_numeric(df.get("dk_spread_open", 0), errors="coerce").fillna(0)
    oa_close = pd.to_numeric(df.get("odds_api_spread_close", 0), errors="coerce").fillna(0)
    oa_open = pd.to_numeric(df.get("odds_api_spread_open", 0), errors="coerce").fillna(0)
    
    # Closing line: ESPN > DK close > OA close > market_spread_home
    closing = np.where(espn != 0, espn,
              np.where(dk_close != 0, dk_close,
              np.where(oa_close != 0, oa_close,
              np.where(mkt != 0, mkt, 0))))
    
    # Opening line: DK open > OA open > (closing as fallback for no-movement games)
    opening = np.where(dk_open != 0, dk_open,
              np.where(oa_open != 0, oa_open, 0))
    
    df["closing_spread"] = closing.astype(float)
    df["opening_spread"] = opening.astype(float)
    
    # Also fix espn_spread for 2024-2025 using OA close as fallback
    # This helps mkt_spread in the feature builder
    if "espn_spread" in df.columns:
        espn_zero = espn == 0
        fill_oa = espn_zero & (oa_close != 0)
        fill_dk = espn_zero & (dk_close != 0) & (oa_close == 0)
        df.loc[fill_oa, "espn_spread"] = oa_close[fill_oa]
        df.loc[fill_dk, "espn_spread"] = dk_close[fill_dk]
        filled_espn = fill_oa.sum() + fill_dk.sum()
    else:
        filled_espn = 0
    
    # Same for over/under
    if "espn_over_under" in df.columns:
        espn_ou = pd.to_numeric(df["espn_over_under"], errors="coerce").fillna(0)
        dk_ou_close = pd.to_numeric(df.get("dk_total_close", 0), errors="coerce").fillna(0)
        oa_ou_close = pd.to_numeric(df.get("odds_api_total_close", 0), errors="coerce").fillna(0)
        ou_zero = espn_ou == 0
        fill_ou_oa = ou_zero & (oa_ou_close != 0)
        fill_ou_dk = ou_zero & (dk_ou_close != 0) & (oa_ou_close == 0)
        df.loc[fill_ou_oa, "espn_over_under"] = oa_ou_close[fill_ou_oa]
        df.loc[fill_ou_dk, "espn_over_under"] = dk_ou_close[fill_ou_dk]
    
    # Summary
    has_closing = (df["closing_spread"] != 0).sum()
    has_opening = (df["opening_spread"] != 0).sum()
    
    print("  Fix 2 (spread unification):")
    print("    espn_spread patched: %d rows (from OA/DK close)" % filled_espn)
    print("    closing_spread: %d/%d (%.1f%%)" % (has_closing, n, has_closing/n*100))
    print("    opening_spread: %d/%d (%.1f%%)" % (has_opening, n, has_opening/n*100))
    
    # Per-season
    for s in sorted(df["season"].unique()):
        mask = df["season"] == s
        ns = mask.sum()
        nc = (df.loc[mask, "closing_spread"] != 0).sum()
        no = (df.loc[mask, "opening_spread"] != 0).sum()
        ne = (pd.to_numeric(df.loc[mask, "espn_spread"], errors="coerce").fillna(0) != 0).sum()
        if ns > 0:
            print("    %d: close=%d (%.0f%%), open=%d (%.0f%%), espn(patched)=%d (%.0f%%)" % (
                int(s), nc, nc/ns*100, no, no/ns*100, ne, ne/ns*100))
    
    return df


def fix_roll_ats(df):
    """
    Issue 2b: roll_ats_n/margin is 0% in 2025, degraded in 2024.
    
    roll_ats_pct is 98-100% everywhere, so ATS percentage exists.
    But roll_ats_n (number of ATS games) and roll_ats_margin are missing.
    
    Fix: if roll_ats_pct exists but roll_ats_n is 0, estimate n=5.
    For margin: if pct exists but margin is 0, estimate margin = (pct - 0.5) * 5
    (ATS winners average ~2-3 pt margin, so pct above 0.5 * scaling factor)
    """
    total_fixed = 0
    for side in ["home", "away"]:
        n_col = f"{side}_roll_ats_n"
        margin_col = f"{side}_roll_ats_margin"
        pct_col = f"{side}_roll_ats_pct"
        
        if n_col not in df.columns:
            continue
        
        ats_n = pd.to_numeric(df[n_col], errors="coerce").fillna(0)
        ats_pct = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.5)
        ats_margin = pd.to_numeric(df.get(margin_col, 0), errors="coerce").fillna(0)
        
        # Where n=0 but pct exists (not default 0.5), estimate n=5
        needs_n_fix = (ats_n == 0) & (ats_pct != 0.5) & (ats_pct != 0)
        if needs_n_fix.sum() > 0:
            df.loc[needs_n_fix, n_col] = 5
            total_fixed += needs_n_fix.sum()
            print("  Fix 2b (%s n): estimated n=5 for %d rows" % (n_col, needs_n_fix.sum()))
        
        # Where margin=0 but pct exists, estimate margin from pct
        # Typical ATS margin ~ (pct - 0.5) * 5 points per game
        needs_margin_fix = (ats_margin == 0) & (ats_pct != 0.5) & (ats_pct != 0) & (ats_n > 0)
        if needs_margin_fix.sum() > 0:
            estimated_margin = (ats_pct[needs_margin_fix] - 0.5) * 5.0
            df.loc[needs_margin_fix, margin_col] = estimated_margin.round(2)
            print("  Fix 2b (%s margin): estimated for %d rows" % (margin_col, needs_margin_fix.sum()))
    
    if total_fixed > 0:
        # Verify per season
        for s in sorted(df["season"].unique()):
            mask = df["season"] == s
            ns = mask.sum()
            if ns > 0:
                n_nz = (pd.to_numeric(df.loc[mask, "home_roll_ats_n"], errors="coerce").fillna(0) > 0).sum()
                m_nz = (pd.to_numeric(df.loc[mask, "home_roll_ats_margin"], errors="coerce").fillna(0) != 0).sum()
    
    return df


def flag_dead_features(df):
    """
    Issue 3: Identify features that are always or nearly always zero.
    These add noise without signal and should be considered for removal.
    """
    dead = []
    always_zero = [
        "espn_ml_home", "espn_ml_away",  # 0% all seasons
        "espn_predictor_home_pct",  # 0% all seasons
        "attendance", "venue_capacity",  # 0% all seasons
        "home_injury_penalty", "away_injury_penalty", "injury_diff",  # 0% all seasons
        "home_missing_starters", "away_missing_starters",  # 0% all seasons
        "home_starter_mins", "away_starter_mins",  # 0% all seasons
        "spread_home", "win_pct_home", "ou_total",  # 0% all seasons
    ]
    
    low_coverage = [
        "is_sandwich",  # <2% all seasons
        "is_lookahead",  # <6% all seasons
        "is_ncaa_tournament",  # <2.3%, 0% in 2026
        "is_postseason",  # same as ncaa_tournament
        "is_letdown",  # <6%
    ]
    
    # Features that broke in specific seasons
    broken_2026 = [
        "consistency_x_spread",  # 11% → 0% in 2026
        "luck_x_spread",  # 10% → 0% in 2026
    ]
    
    print("  Fix 3 (dead features flagged):")
    print("    Always zero (%d): %s" % (len(always_zero), ", ".join(always_zero[:5]) + "..."))
    print("    Low coverage (%d): %s" % (len(low_coverage), ", ".join(low_coverage)))
    print("    Broken 2026 (%d): %s" % (len(broken_2026), ", ".join(broken_2026)))
    print("    NOTE: These are flagged, not removed. Remove from feature_cols in ncaa.py to clean up.")
    
    return dead, always_zero, low_coverage, broken_2026


def create_opening_line_columns(df):
    """
    Issue 4: Opening lines exist in odds_api_spread_open (2024-2025) and 
    dk_spread_open (2026) but aren't used for evaluation.
    
    Creates unified opening_spread and closing_spread columns
    (already done in fix_spread_sources, this just reports).
    """
    if "opening_spread" not in df.columns:
        print("  Fix 4: Run fix_spread_sources first")
        return df
    
    seasons_with_open = {}
    for s in sorted(df["season"].unique()):
        mask = df["season"] == s
        n = mask.sum()
        has_open = (df.loc[mask, "opening_spread"] != 0).sum()
        has_close = (df.loc[mask, "closing_spread"] != 0).sum()
        if has_open > 0:
            seasons_with_open[int(s)] = (has_open, n)
    
    total_open = sum(v[0] for v in seasons_with_open.values())
    print("  Fix 4 (opening lines ready):")
    print("    Total games with real opening lines: %d" % total_open)
    for s, (no, n) in sorted(seasons_with_open.items()):
        print("    %d: %d/%d (%.1f%%)" % (s, no, n, no/n*100))
    
    return df


def apply_training_fixes(df):
    """Apply all training data fixes in order."""
    print("\n" + "=" * 70)
    print("  APPLYING TRAINING DATA FIXES")
    print("=" * 70)
    
    # Ensure numeric types
    for col in ["season", "home_ppg", "away_ppg", "home_opp_ppg", "away_opp_ppg",
                "home_wins", "away_wins", "home_losses", "away_losses",
                "home_record_wins", "away_record_wins", "home_record_losses", "away_record_losses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = fix_pyth_residual(df)
    df = fix_spread_sources(df)
    df = fix_roll_ats(df)
    dead, always_zero, low_cov, broken = flag_dead_features(df)
    df = create_opening_line_columns(df)
    
    print("\n  All fixes applied. Ready for ncaa_build_features().")
    return df


if __name__ == "__main__":
    from dump_training_data import load_cached
    
    print("Loading data...")
    df = load_cached()
    if df is None:
        print("No cached data")
        exit(1)
    
    df = df[df["actual_home_score"].notna()].copy()
    df = df[df["season"] != 2021].copy()
    print("Loaded %d rows" % len(df))
    
    df = apply_training_fixes(df)
    
    # Quick verification
    print("\n\nVERIFICATION:")
    for col in ["home_pyth_residual", "away_pyth_residual", "closing_spread", "opening_spread"]:
        if col in df.columns:
            nz = (pd.to_numeric(df[col], errors="coerce").fillna(0) != 0).sum()
            print("  %s: %d/%d non-zero (%.1f%%)" % (col, nz, len(df), nz/len(df)*100))
