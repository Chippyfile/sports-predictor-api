"""
COMPUTE MISSING TRAINING FEATURES
====================================
These 5 features exist in ncaa_full_predict.py (live) but are 100% zero
in training because ncaa_historical lacks the columns. This script computes
them inline from the training DataFrame, using only PRIOR games (no leakage).

Features:
  1. h2h_margin_avg      — rolling avg margin from prior games between same two teams
  2. h2h_home_win_rate   — home win rate from prior H2H matchups
  3. conf_strength_diff  — conference quality gap (computed from season results)
  4. cross_conf_flag     — 1 if teams from different conferences
  5. recent_form_diff    — last 5 games win rate diff

All use expanding-window lookups: for each game, only prior games are visible.

Usage:
  from compute_h2h_conf_form import compute_missing_features
  df = compute_missing_features(df)
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def compute_missing_features(df):
    """
    Compute all 5 missing features from the training DataFrame.
    Requires: game_date, home_team_id, away_team_id, actual_home_score,
              actual_away_score, home_conference, away_conference, season
    """
    df = df.copy()

    # Ensure types
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["home_team_id"] = df["home_team_id"].astype(str)
    df["away_team_id"] = df["away_team_id"].astype(str)
    df["actual_home_score"] = pd.to_numeric(df.get("actual_home_score", 0), errors="coerce")
    df["actual_away_score"] = pd.to_numeric(df.get("actual_away_score", 0), errors="coerce")

    if "home_conference" in df.columns:
        df["home_conference"] = df["home_conference"].fillna("").astype(str)
    else:
        df["home_conference"] = ""
    if "away_conference" in df.columns:
        df["away_conference"] = df["away_conference"].fillna("").astype(str)
    else:
        df["away_conference"] = ""

    # Sort by date for proper expanding-window computation
    df = df.sort_values("game_date").reset_index(drop=True)

    has_scores = df["actual_home_score"].notna() & df["actual_away_score"].notna()

    # ══════════════════════════════════════════════════════════
    # 1 & 2: H2H MARGIN AVG + HOME WIN RATE
    # ══════════════════════════════════════════════════════════
    # For each game, look at all prior games between these two teams
    # (in either home/away order) and compute:
    #   h2h_margin_avg: average margin from team A's perspective (as home team in current game)
    #   h2h_home_win_rate: fraction of prior H2H where current home team won
    #
    # Key: (min(id_a, id_b), max(id_a, id_b)) → list of (date, margin_for_first_id)

    print("  Computing h2h_margin_avg + h2h_home_win_rate...")

    # Build H2H lookup: for each pair of teams, track all prior game margins
    # Use a canonical key (sorted team IDs) so A vs B and B vs A map to same entry
    h2h_history = defaultdict(list)  # key → list of (home_team_id, margin)

    h2h_margin = np.zeros(len(df))
    h2h_win_rate = np.zeros(len(df))

    for idx, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]
        margin = row["actual_home_score"] - row["actual_away_score"]

        # Canonical key for this matchup
        pair_key = tuple(sorted([h_id, a_id]))

        # Look up prior H2H games
        prior = h2h_history[pair_key]
        if prior:
            # Compute from current home team's perspective
            margins_from_home_perspective = []
            home_wins = 0
            for prev_home, prev_margin in prior:
                if prev_home == h_id:
                    # Same home team as current game
                    margins_from_home_perspective.append(prev_margin)
                    if prev_margin > 0:
                        home_wins += 1
                else:
                    # Teams were flipped — invert margin
                    margins_from_home_perspective.append(-prev_margin)
                    if -prev_margin > 0:
                        home_wins += 1

            h2h_margin[idx] = np.mean(margins_from_home_perspective)
            h2h_win_rate[idx] = home_wins / len(prior)
        # else: stays 0 (no prior matchups)

        # Update history AFTER computing (no leakage)
        if has_scores.iloc[idx] and not np.isnan(margin):
            h2h_history[pair_key].append((h_id, margin))

    df["h2h_margin_avg"] = h2h_margin
    df["h2h_home_win_rate"] = h2h_win_rate

    h2h_nonzero = (h2h_margin != 0).sum()
    print(f"    h2h_margin_avg: {h2h_nonzero}/{len(df)} non-zero ({h2h_nonzero/len(df)*100:.1f}%)")
    print(f"    mean={h2h_margin[h2h_margin != 0].mean():.2f}, "
          f"std={h2h_margin[h2h_margin != 0].std():.2f}") if h2h_nonzero > 0 else None

    # ══════════════════════════════════════════════════════════
    # 3 & 4: CONF STRENGTH DIFF + CROSS CONF FLAG
    # ══════════════════════════════════════════════════════════
    # Conference strength = average margin of victory for all teams in a conference
    # during the CURRENT season, computed from prior games only.
    #
    # cross_conf_flag = 1 if home and away are in different conferences
    #
    # We track per-season, per-conference rolling stats and compute strength
    # as avg margin per game for all teams in that conference.

    print("  Computing conf_strength_diff + cross_conf_flag...")

    # Track per-team margins within each season, then aggregate by conference
    # team_season_margins: (season, team_id) → list of margins
    team_season_margins = defaultdict(list)
    # Cache conference membership: (season, team_id) → conference
    team_conf_cache = {}

    conf_strength_arr = np.zeros(len(df))
    cross_conf_arr = np.zeros(len(df), dtype=int)

    for idx, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]
        h_conf = row["home_conference"]
        a_conf = row["away_conference"]
        season = row.get("season", 0)
        margin = row["actual_home_score"] - row["actual_away_score"]

        # Cache conference membership
        if season and h_conf:
            team_conf_cache[(season, h_id)] = h_conf
        if season and a_conf:
            team_conf_cache[(season, a_id)] = a_conf

        # Cross-conference flag
        if h_conf and a_conf and h_conf != a_conf:
            cross_conf_arr[idx] = 1

        # Compute conference strength from prior games this season
        if season and h_conf and a_conf:
            # Get all teams in each conference this season
            h_conf_teams = [tid for (s, tid), c in team_conf_cache.items()
                           if s == season and c == h_conf]
            a_conf_teams = [tid for (s, tid), c in team_conf_cache.items()
                           if s == season and c == a_conf]

            # Home conference strength = avg margin of all teams in that conference
            h_margins = []
            for tid in h_conf_teams:
                h_margins.extend(team_season_margins.get((season, tid), []))
            a_margins = []
            for tid in a_conf_teams:
                a_margins.extend(team_season_margins.get((season, tid), []))

            h_strength = np.mean(h_margins) if h_margins else 0.0
            a_strength = np.mean(a_margins) if a_margins else 0.0

            conf_strength_arr[idx] = h_strength - a_strength

        # Update team margins AFTER computing (no leakage)
        if has_scores.iloc[idx] and not np.isnan(margin):
            team_season_margins[(season, h_id)].append(margin)
            team_season_margins[(season, a_id)].append(-margin)

    df["conf_strength_diff"] = conf_strength_arr
    df["cross_conf_flag"] = cross_conf_arr

    cs_nonzero = (conf_strength_arr != 0).sum()
    cc_nonzero = (cross_conf_arr != 0).sum()
    print(f"    conf_strength_diff: {cs_nonzero}/{len(df)} non-zero ({cs_nonzero/len(df)*100:.1f}%)")
    print(f"    cross_conf_flag: {cc_nonzero}/{len(df)} non-zero ({cc_nonzero/len(df)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════
    # 5: RECENT FORM DIFF
    # ══════════════════════════════════════════════════════════
    # Last 5 games win rate for each team, computed from prior games only.
    # recent_form_diff = home_team_win_rate_last5 - away_team_win_rate_last5

    print("  Computing recent_form_diff...")

    # Track each team's recent results (expanding, take last 5)
    team_results = defaultdict(list)  # team_id → list of 1/0 (win/loss)

    recent_form_arr = np.zeros(len(df))

    for idx, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]
        margin = row["actual_home_score"] - row["actual_away_score"]

        # Look up last 5 games for each team
        h_recent = team_results[h_id][-5:] if team_results[h_id] else []
        a_recent = team_results[a_id][-5:] if team_results[a_id] else []

        h_form = np.mean(h_recent) if h_recent else 0.5
        a_form = np.mean(a_recent) if a_recent else 0.5

        # Only compute diff if both have some history
        if h_recent and a_recent:
            recent_form_arr[idx] = h_form - a_form

        # Update results AFTER computing (no leakage)
        if has_scores.iloc[idx] and not np.isnan(margin):
            team_results[h_id].append(1 if margin > 0 else 0)
            team_results[a_id].append(1 if margin < 0 else 0)

    df["recent_form_diff"] = recent_form_arr

    rf_nonzero = (recent_form_arr != 0).sum()
    print(f"    recent_form_diff: {rf_nonzero}/{len(df)} non-zero ({rf_nonzero/len(df)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n  Feature computation complete:")
    print(f"    h2h_margin_avg:      {h2h_nonzero:>6d}/{len(df)} ({h2h_nonzero/len(df)*100:.1f}%)")
    print(f"    h2h_home_win_rate:   {h2h_nonzero:>6d}/{len(df)} ({h2h_nonzero/len(df)*100:.1f}%)")
    print(f"    conf_strength_diff:  {cs_nonzero:>6d}/{len(df)} ({cs_nonzero/len(df)*100:.1f}%)")
    print(f"    cross_conf_flag:     {cc_nonzero:>6d}/{len(df)} ({cc_nonzero/len(df)*100:.1f}%)")
    print(f"    recent_form_diff:    {rf_nonzero:>6d}/{len(df)} ({rf_nonzero/len(df)*100:.1f}%)")

    return df


if __name__ == "__main__":
    # Quick test with the training parquet
    import sys, os, time
    sys.path.insert(0, '.')

    from dump_training_data import load_cached

    print("=" * 60)
    print("  COMPUTE MISSING TRAINING FEATURES — TEST")
    print("=" * 60)

    df = load_cached()
    if df is None:
        print("  No cache found. Run dump_training_data.py first.")
        sys.exit(1)

    df = df[df["actual_home_score"].notna()].copy()
    print(f"  Loaded {len(df)} games")

    t0 = time.time()
    df = compute_missing_features(df)
    elapsed = time.time() - t0
    print(f"\n  Computed in {elapsed:.1f}s")

    # Correlation with margin
    margin = df["actual_home_score"].values - df["actual_away_score"].values
    for col in ["h2h_margin_avg", "h2h_home_win_rate", "conf_strength_diff", "cross_conf_flag", "recent_form_diff"]:
        vals = df[col].values
        nonzero = vals != 0
        if nonzero.sum() > 100:
            r = np.corrcoef(vals[nonzero], margin[nonzero])[0, 1]
            print(f"    {col:<25s} r={r:+.3f} (on {nonzero.sum()} non-zero games)")
        else:
            print(f"    {col:<25s} too few non-zero ({nonzero.sum()})")
