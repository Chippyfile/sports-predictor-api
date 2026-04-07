#!/usr/bin/env python3
"""
mlb_new_features.py — Build and test new MLB features
======================================================
PART 1: Features computable from existing training data (1-7)
PART 2: Statcast features from Baseball Savant CSV downloads (8-11)
PART 3: Correlation test against ATS margin and O/U residual targets

Usage:
    python3 mlb_new_features.py                    # Compute + test (skip Savant)
    python3 mlb_new_features.py --savant           # Also download Statcast data
    python3 mlb_new_features.py --write            # Save to parquet
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict

MODERN_TO_RETRO = {"LAA":"ANA","CWS":"CHA","CHC":"CHN","KC":"KCA","LAD":"LAN","NYY":"NYA","NYM":"NYN","SD":"SDN","SF":"SFN","STL":"SLN","TB":"TBA","WSH":"WAS"}
RETRO_TO_MODERN = {v: k for k, v in MODERN_TO_RETRO.items()}

# Division lookup (retrosheet team codes)
DIVISIONS = {
    # AL East
    "NYA": "ALE", "BOS": "ALE", "BAL": "ALE", "TBA": "ALE", "TOR": "ALE",
    # AL Central
    "CLE": "ALC", "MIN": "ALC", "CHA": "ALC", "DET": "ALC", "KCA": "ALC",
    # AL West
    "HOU": "ALW", "SEA": "ALW", "TEX": "ALW", "ANA": "ALW", "OAK": "ALW",
    # NL East
    "ATL": "NLE", "NYN": "NLE", "PHI": "NLE", "MIA": "NLE", "WAS": "NLE",
    # NL Central
    "MIL": "NLC", "CHN": "NLC", "SLN": "NLC", "PIT": "NLC", "CIN": "NLC",
    # NL West
    "LAN": "NLW", "SDN": "NLW", "SFN": "NLW", "ARI": "NLW", "COL": "NLW",
}


# ═══════════════════════════════════════════════════════════
# PART 1: COMPUTABLE FROM EXISTING DATA
# ═══════════════════════════════════════════════════════════

def compute_features_from_data(df):
    """Compute features 1-7 using only existing training data columns."""
    df = df.sort_values("game_date").reset_index(drop=True)
    n = len(df)
    print(f"  Computing new features on {n} games...")

    # ── Feature 1: sp_form_x_park ──
    # Pitcher deterioration amplified by park factor
    sp_form = df.get("sp_form_combined", pd.Series(0, index=df.index)).fillna(0)
    pf = pd.to_numeric(df.get("park_factor", 1.0), errors="coerce").fillna(1.0)
    df["sp_form_x_park"] = sp_form * pf
    print(f"    1. sp_form_x_park: computed ({df['sp_form_x_park'].notna().sum()} non-null)")

    # ── Feature 2: games_played_reliability ──
    # Signal reliability based on season sample size
    # Need to compute games played per team up to each game date
    df["games_played_reliability"] = np.nan
    team_game_counts = defaultdict(int)
    for idx, row in df.iterrows():
        ht, at = row.get("home_team", ""), row.get("away_team", "")
        h_games = team_game_counts.get(f"{row.get('season',0)}_{ht}", 0)
        a_games = team_game_counts.get(f"{row.get('season',0)}_{at}", 0)
        min_games = min(h_games, a_games)
        df.at[idx, "games_played_reliability"] = min(1.0, min_games / 50.0)
        # Increment counts
        team_game_counts[f"{row.get('season',0)}_{ht}"] = h_games + 1
        team_game_counts[f"{row.get('season',0)}_{at}"] = a_games + 1
    print(f"    2. games_played_reliability: computed ({df['games_played_reliability'].notna().sum()} non-null)")

    # ── Feature 3: rest_x_sp_form ──
    # Rest days interaction with pitcher form
    rest_h = pd.to_numeric(df.get("home_rest_days", 1), errors="coerce").fillna(1)
    rest_a = pd.to_numeric(df.get("away_rest_days", 1), errors="coerce").fillna(1)
    rest_combined = rest_h + rest_a
    df["rest_x_sp_form"] = rest_combined * sp_form
    print(f"    3. rest_x_sp_form: computed")

    # ── Feature 4: scoring_consistency_diff ──
    # Coefficient of variation of runs scored (last 10 games)
    df["home_scoring_cv"] = np.nan
    df["away_scoring_cv"] = np.nan
    team_runs_history = defaultdict(list)  # team → [runs scored in recent games]

    for idx, row in df.iterrows():
        ht, at = row.get("home_team", ""), row.get("away_team", "")
        season = row.get("season", 0)

        # Get recent scoring history
        h_key = f"{season}_{ht}"
        a_key = f"{season}_{at}"
        h_hist = team_runs_history.get(h_key, [])
        a_hist = team_runs_history.get(a_key, [])

        if len(h_hist) >= 5:
            recent = h_hist[-10:]
            mean_r = np.mean(recent)
            if mean_r > 0:
                df.at[idx, "home_scoring_cv"] = np.std(recent) / mean_r
        if len(a_hist) >= 5:
            recent = a_hist[-10:]
            mean_r = np.mean(recent)
            if mean_r > 0:
                df.at[idx, "away_scoring_cv"] = np.std(recent) / mean_r

        # Record this game's runs
        h_runs = row.get("actual_home_runs")
        a_runs = row.get("actual_away_runs")
        if pd.notna(h_runs):
            team_runs_history[h_key].append(float(h_runs))
        if pd.notna(a_runs):
            team_runs_history[a_key].append(float(a_runs))

    df["scoring_consistency_diff"] = df["home_scoring_cv"].fillna(0.5) - df["away_scoring_cv"].fillna(0.5)
    df["scoring_cv_combined"] = df["home_scoring_cv"].fillna(0.5) + df["away_scoring_cv"].fillna(0.5)
    print(f"    4. scoring_consistency_diff: computed ({df['home_scoring_cv'].notna().sum()} non-null)")

    # ── Feature 5: day_night indicator ──
    # Day games (especially after night games) historically lower scoring
    if "day_night" in df.columns:
        df["is_day_game"] = (df["day_night"] == "D").astype(int)
    else:
        # Proxy: games before 5pm local are day games
        # Without time data, default to 0
        df["is_day_game"] = 0
    print(f"    5. is_day_game: computed ({df['is_day_game'].sum()} day games)")

    # ── Feature 6: division_game ──
    # Same-division teams are more familiar — lower scoring, tighter games
    def get_div(team):
        return DIVISIONS.get(team, "UNK")

    df["division_game"] = df.apply(
        lambda r: 1 if get_div(r.get("home_team", "")) == get_div(r.get("away_team", ""))
                       and get_div(r.get("home_team", "")) != "UNK"
                  else 0,
        axis=1
    )
    print(f"    6. division_game: computed ({df['division_game'].sum()} division games / {n} total)")

    # ── Feature 7: home_bounce_back ──
    # Home team lost their last game (bounce-back effect)
    df["home_bounce_back"] = 0
    df["away_bounce_back"] = 0
    team_last_result = {}  # team → True if last game was a loss

    for idx, row in df.iterrows():
        ht, at = row.get("home_team", ""), row.get("away_team", "")
        season = row.get("season", 0)
        h_key = f"{season}_{ht}"
        a_key = f"{season}_{at}"

        # Check if last game was a loss
        if h_key in team_last_result and team_last_result[h_key]:
            df.at[idx, "home_bounce_back"] = 1
        if a_key in team_last_result and team_last_result[a_key]:
            df.at[idx, "away_bounce_back"] = 1

        # Record this game's result
        h_runs = row.get("actual_home_runs")
        a_runs = row.get("actual_away_runs")
        if pd.notna(h_runs) and pd.notna(a_runs):
            team_last_result[h_key] = float(h_runs) < float(a_runs)  # home lost
            team_last_result[a_key] = float(a_runs) < float(h_runs)  # away lost
    print(f"    7. home_bounce_back: computed ({df['home_bounce_back'].sum()} bounce-backs)")

    # ── Bonus interactions ──
    # BP quality × SP form (when starter is bad, bullpen matters more)
    bp_h = pd.to_numeric(df.get("home_bullpen_era", 4.1), errors="coerce").fillna(4.1)
    bp_a = pd.to_numeric(df.get("away_bullpen_era", 4.1), errors="coerce").fillna(4.1)
    bp_combined = bp_h + bp_a
    df["bp_x_sp_form"] = bp_combined * sp_form.clip(lower=0)  # only when SP deteriorating
    print(f"    +. bp_x_sp_form: computed")

    # Streak momentum (run diff in last 5 vs season)
    df["home_streak_momentum"] = np.nan
    df["away_streak_momentum"] = np.nan
    team_rd_history = defaultdict(list)

    for idx, row in df.iterrows():
        ht, at = row.get("home_team", ""), row.get("away_team", "")
        season = row.get("season", 0)
        h_key = f"{season}_{ht}"
        a_key = f"{season}_{at}"

        h_hist = team_rd_history.get(h_key, [])
        a_hist = team_rd_history.get(a_key, [])

        if len(h_hist) >= 5:
            recent_avg = np.mean(h_hist[-5:])
            season_avg = np.mean(h_hist)
            df.at[idx, "home_streak_momentum"] = recent_avg - season_avg
        if len(a_hist) >= 5:
            recent_avg = np.mean(a_hist[-5:])
            season_avg = np.mean(a_hist)
            df.at[idx, "away_streak_momentum"] = recent_avg - season_avg

        h_runs = row.get("actual_home_runs")
        a_runs = row.get("actual_away_runs")
        if pd.notna(h_runs) and pd.notna(a_runs):
            team_rd_history[h_key].append(float(h_runs) - float(a_runs))
            team_rd_history[a_key].append(float(a_runs) - float(h_runs))

    df["streak_momentum_diff"] = df["home_streak_momentum"].fillna(0) - df["away_streak_momentum"].fillna(0)
    df["streak_momentum_combined"] = df["home_streak_momentum"].fillna(0) + df["away_streak_momentum"].fillna(0)
    print(f"    +. streak_momentum_diff: computed ({df['home_streak_momentum'].notna().sum()} non-null)")

    return df


# ═══════════════════════════════════════════════════════════
# PART 2: STATCAST FROM BASEBALL SAVANT
# ═══════════════════════════════════════════════════════════

def download_savant_team_stats():
    """Download team-level Statcast stats from Baseball Savant (2015-2025).
    Returns DataFrame with: team, season, xwoba, barrel_rate, hard_hit_pct, xera.
    """
    import requests

    all_data = []
    for year in range(2015, 2027):
        for stat_type in ["batter", "pitcher"]:
            url = (
                f"https://baseballsavant.mlb.com/leaderboard/expected_statistics"
                f"?type={stat_type}&year={year}&position=&team=&min=1"
                f"&csv=true"
            )
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                if r.ok and len(r.text) > 100:
                    from io import StringIO
                    chunk = pd.read_csv(StringIO(r.text))
                    chunk["season"] = year
                    chunk["stat_type"] = stat_type
                    all_data.append(chunk)
                    print(f"    {year} {stat_type}: {len(chunk)} rows")
                else:
                    print(f"    {year} {stat_type}: no data ({r.status_code})")
            except Exception as e:
                print(f"    {year} {stat_type}: error — {e}")
            time.sleep(1)  # rate limit

    if not all_data:
        print("  ❌ No Savant data downloaded")
        return None

    savant = pd.concat(all_data, ignore_index=True)
    savant.to_csv("savant_team_stats.csv", index=False)
    print(f"\n  Saved: savant_team_stats.csv ({len(savant)} rows)")
    return savant


def merge_savant_features(df, savant_path="savant_team_stats.csv"):
    """Merge Savant team stats into training data."""
    if not os.path.exists(savant_path):
        print(f"  ⚠ {savant_path} not found — run with --savant to download")
        return df

    savant = pd.read_csv(savant_path)
    print(f"  Loaded Savant data: {len(savant)} rows")

    # We need to aggregate to team-level per season
    # Savant data is player-level — need to group by team
    # Actually, if we download with &team= filter, we get team aggregates
    # For now, check what columns we have
    print(f"  Savant columns: {list(savant.columns)[:15]}")

    # TODO: Map Savant team names to retrosheet codes and merge
    # This depends on the exact format of the downloaded data
    print(f"  ⚠ Savant merge needs team mapping — check savant_team_stats.csv format")
    return df


# ═══════════════════════════════════════════════════════════
# PART 3: CORRELATION TEST
# ═══════════════════════════════════════════════════════════

def test_correlations(df):
    """Test all new features against ATS margin and O/U residual targets."""
    # Targets
    df["actual_margin"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) - \
                          pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    df["actual_total"] = pd.to_numeric(df.get("actual_home_runs", 0), errors="coerce").fillna(0) + \
                         pd.to_numeric(df.get("actual_away_runs", 0), errors="coerce").fillna(0)
    mkt_spread = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
    mkt_ou = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0)

    has_spread = np.abs(mkt_spread) > 0.1
    has_ou = mkt_ou > 0

    # ATS residual (how much did the game deviate from market spread?)
    df["ats_residual"] = df["actual_margin"] - (-mkt_spread)
    # O/U residual (how much did the total deviate from market O/U?)
    df["ou_residual"] = df["actual_total"] - mkt_ou

    new_features = [
        # Part 1: Computable
        "sp_form_x_park",
        "games_played_reliability",
        "rest_x_sp_form",
        "scoring_consistency_diff",
        "scoring_cv_combined",
        "is_day_game",
        "division_game",
        "home_bounce_back",
        "away_bounce_back",
        "bp_x_sp_form",
        "streak_momentum_diff",
        "streak_momentum_combined",
        # Existing features for comparison
        "sp_form_combined",
        "park_factor",
    ]

    print(f"\n{'='*75}")
    print(f"  CORRELATION WITH TARGETS")
    print(f"{'='*75}")

    print(f"\n  {'Feature':<30s} {'ATS r':>8s} {'n':>7s}  {'O/U r':>8s} {'n':>7s}  {'Margin r':>9s}")
    print(f"  {'-'*75}")

    for feat in new_features:
        if feat not in df.columns:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce")

        # ATS correlation (with games that have market spread)
        ats_mask = has_spread & vals.notna()
        ats_r = vals[ats_mask].corr(df.loc[ats_mask, "ats_residual"]) if ats_mask.sum() > 100 else np.nan
        ats_n = ats_mask.sum()

        # O/U correlation (with games that have market O/U)
        ou_mask = has_ou & vals.notna()
        ou_r = vals[ou_mask].corr(df.loc[ou_mask, "ou_residual"]) if ou_mask.sum() > 100 else np.nan
        ou_n = ou_mask.sum()

        # Raw margin correlation (all games)
        mar_mask = vals.notna()
        mar_r = vals[mar_mask].corr(df.loc[mar_mask, "actual_margin"]) if mar_mask.sum() > 100 else np.nan

        flag_ats = "★" if pd.notna(ats_r) and abs(ats_r) >= 0.03 else ""
        flag_ou = "★" if pd.notna(ou_r) and abs(ou_r) >= 0.03 else ""
        flag_mar = "★" if pd.notna(mar_r) and abs(mar_r) >= 0.05 else ""

        print(f"  {feat:<30s} {ats_r:>+7.4f} {ats_n:>7d}  {ou_r:>+7.4f} {ou_n:>7d}  {mar_r:>+8.4f} {flag_ats}{flag_ou}{flag_mar}")

    # ── Also test O/U direction accuracy for top features ──
    print(f"\n  O/U DIRECTION TEST (does the feature predict over/under?)")
    print(f"  {'Feature':<30s} {'High→Over%':>12s} {'n':>6s} {'Low→Under%':>12s} {'n':>6s}")
    print(f"  {'-'*70}")

    for feat in ["sp_form_x_park", "sp_form_combined", "scoring_cv_combined",
                   "streak_momentum_combined", "bp_x_sp_form", "division_game"]:
        if feat not in df.columns:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce")
        mask = has_ou & vals.notna()
        sub = df[mask].copy()
        sub["_val"] = vals[mask]
        sub["went_over"] = sub["actual_total"] > mkt_ou[mask]
        sub["went_under"] = sub["actual_total"] < mkt_ou[mask]

        if feat in ["division_game", "is_day_game"]:
            # Binary feature — compare rates
            on = sub["_val"] == 1
            off = sub["_val"] == 0
            if on.sum() > 50 and off.sum() > 50:
                on_under = sub.loc[on, "went_under"].mean()
                off_under = sub.loc[off, "went_under"].mean()
                print(f"  {feat:<30s} {feat}=1: {1-on_under:.1%} over {on.sum():>6d}  {feat}=0: {off_under:.1%} under {off.sum():>6d}")
        else:
            # Continuous — split at median
            med = sub["_val"].median()
            high = sub["_val"] > med
            low = sub["_val"] <= med
            if high.sum() > 50 and low.sum() > 50:
                high_over = sub.loc[high, "went_over"].mean()
                low_under = sub.loc[low, "went_under"].mean()
                print(f"  {feat:<30s} above med: {high_over:.1%} over {high.sum():>6d}  below med: {low_under:.1%} under {low.sum():>6d}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    write = "--write" in sys.argv
    fetch_savant = "--savant" in sys.argv

    print("=" * 70)
    print("  MLB NEW FEATURE ENGINEERING + TESTING")
    print("=" * 70)

    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_training_data.parquet")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {len(df)} games, {len(df.columns)} columns")

    # Filter same as production
    df = df[df["season"] != 2020].copy()
    print(f"  After dropping 2020: {len(df)} games")

    # ── Part 1: Compute features from existing data ──
    print(f"\n{'='*50}\n  PART 1: FEATURES FROM EXISTING DATA\n{'='*50}")
    df = compute_features_from_data(df)

    # ── Part 2: Statcast download (optional) ──
    if fetch_savant:
        print(f"\n{'='*50}\n  PART 2: STATCAST DOWNLOAD\n{'='*50}")
        download_savant_team_stats()
    if os.path.exists("savant_team_stats.csv"):
        df = merge_savant_features(df)

    # ── Part 3: Test correlations ──
    test_correlations(df)

    # ── Save ──
    if write:
        df.to_parquet(parquet_path, index=False)
        print(f"\n  ✅ Saved: {len(df)} games, {len(df.columns)} columns")
    else:
        print(f"\n  New columns: {len(df.columns)} total")
        print(f"  Run with --write to save to parquet")


if __name__ == "__main__":
    main()
