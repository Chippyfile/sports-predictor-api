"""
NBA Historical Data Enrichment v2
Computes ALL derivable features from game results — no external data needed.

Adds to existing enrichment (W/L, form, rest, travel, tempo):
  - Rolling margin stats (trend, accel, variance, skew, kurtosis, ceiling, floor)
  - Momentum features (streak, days since loss, win aging, halflife)
  - Pythagorean luck (actual W% vs expected)
  - Schedule context (games in last 14 days, midweek, season phase)
  - Opponent-quality-adjusted form
  - Common opponents
  - ATS rolling performance
  - Defensive consistency (opponent scoring variance)

Usage:
  python enrich_nba_v2.py                # Enrich parquet
  python enrich_nba_v2.py --push         # Also push to Supabase
"""

import os, sys, time, math, json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# City coordinates (same as nbaUtils.js)
NBA_CITY_COORDS = {
    "ATL": (33.749, -84.388), "BOS": (42.360, -71.059), "BKN": (40.693, -73.975),
    "CHA": (35.227, -80.843), "CHI": (41.882, -87.628), "CLE": (41.499, -81.694),
    "DAL": (32.777, -96.797), "DEN": (39.739, -104.990), "DET": (42.331, -83.046),
    "GSW": (37.775, -122.419), "HOU": (29.760, -95.370), "IND": (39.768, -86.158),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.150, -90.049),
    "MIA": (25.762, -80.192), "MIL": (43.039, -87.907), "MIN": (44.978, -93.265),
    "NOP": (29.951, -90.072), "NYK": (40.751, -73.993), "OKC": (35.468, -97.516),
    "ORL": (28.538, -81.379), "PHI": (39.953, -75.165), "PHX": (33.448, -112.074),
    "POR": (45.523, -122.677), "SAC": (38.582, -121.494), "SAS": (29.424, -98.494),
    "TOR": (43.653, -79.383), "UTA": (40.761, -111.891), "WAS": (38.907, -77.037),
}

# Timezone offsets (for travel direction feature)
NBA_TIMEZONE = {
    "ATL": -5, "BOS": -5, "BKN": -5, "CHA": -5, "CHI": -6, "CLE": -5,
    "DAL": -6, "DEN": -7, "DET": -5, "GSW": -8, "HOU": -6, "IND": -5,
    "LAC": -8, "LAL": -8, "MEM": -6, "MIA": -5, "MIL": -6, "MIN": -6,
    "NOP": -6, "NYK": -5, "OKC": -6, "ORL": -5, "PHI": -5, "PHX": -7,
    "POR": -8, "SAC": -8, "SAS": -6, "TOR": -5, "UTA": -7, "WAS": -5,
}

# Altitude in feet (Denver matters)
NBA_ALTITUDE = {"DEN": 5280}  # Only Denver is significant


def haversine(coord1, coord2):
    R = 3959
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _safe_std(arr):
    return float(np.std(arr)) if len(arr) >= 3 else 0.0

def _safe_skew(arr):
    if len(arr) < 5: return 0.0
    try: return float(scipy_stats.skew(arr))
    except: return 0.0

def _safe_kurtosis(arr):
    if len(arr) < 5: return 0.0
    try: return float(scipy_stats.kurtosis(arr))
    except: return 0.0


def enrich(df):
    """Full enrichment: basic + advanced rolling + momentum + context."""
    df = df.copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    
    for col in ["actual_home_score", "actual_away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    n = len(df)
    print(f"  Enriching {n} games (v2 — full feature set)...")
    
    # ══════════════════════════════════════════════════════════
    # PASS 1: Build per-team game logs with full detail
    # ══════════════════════════════════════════════════════════
    team_games = {}  # {team: [{date, won, margin, opp, home, opp_ppg_season, ...}, ...]}
    
    for idx, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        d = row["game_date"]
        hs = row.get("actual_home_score")
        as_ = row.get("actual_away_score")
        if pd.isna(hs) or pd.isna(as_): continue
        hs, as_ = float(hs), float(as_)
        margin = hs - as_
        
        # Market spread for ATS tracking
        mkt_spread = row.get("market_spread_home")
        if pd.notna(mkt_spread):
            mkt_spread = float(mkt_spread)
        else:
            mkt_spread = None
        
        for team in [h, a]:
            if team not in team_games:
                team_games[team] = []
        
        team_games[h].append({
            "date": d, "won": margin > 0, "margin": margin,
            "scored": hs, "allowed": as_, "opp": a, "home": True,
            "mkt_spread": mkt_spread,
            "ats_margin": margin - mkt_spread if mkt_spread is not None else None,
        })
        team_games[a].append({
            "date": d, "won": margin < 0, "margin": -margin,
            "scored": as_, "allowed": hs, "opp": h, "home": False,
            "mkt_spread": -mkt_spread if mkt_spread is not None else None,
            "ats_margin": (-margin) - (-mkt_spread) if mkt_spread is not None else None,
        })
    
    for team in team_games:
        team_games[team].sort(key=lambda g: g["date"])
    
    # ══════════════════════════════════════════════════════════
    # PASS 2: Compute all features for each game
    # ══════════════════════════════════════════════════════════
    
    # Initialize all output arrays
    cols = {}
    basic_cols = ["home_wins", "home_losses", "away_wins", "away_losses",
                  "home_form", "away_form", "home_days_rest", "away_days_rest",
                  "away_travel_dist"]
    advanced_cols = [
        # Momentum / trend
        "home_margin_trend", "away_margin_trend",       # slope of last 10 margins
        "home_margin_accel", "away_margin_accel",       # change in trend
        "home_streak", "away_streak",                   # current W/L streak (positive=W)
        "home_days_since_loss", "away_days_since_loss",
        "home_games_since_blowout", "away_games_since_blowout",
        "home_wl_momentum", "away_wl_momentum",         # exp-weighted W/L
        "home_momentum_halflife", "away_momentum_halflife",
        "home_win_aging", "away_win_aging",             # recency-weighted win rate
        "home_recovery", "away_recovery",               # games since last loss
        "home_after_loss", "away_after_loss",           # coming off a loss?
        # Scoring distribution
        "home_scoring_var", "away_scoring_var",         # margin std dev
        "home_score_kurtosis", "away_score_kurtosis",
        "home_margin_skew", "away_margin_skew",
        "home_ceiling", "away_ceiling",                 # 90th pct margin
        "home_floor", "away_floor",                     # 10th pct margin
        "home_scoring_entropy", "away_scoring_entropy",
        "home_bimodal", "away_bimodal",                 # |median - mean| of margins
        # Defense consistency
        "home_def_stability", "away_def_stability",     # std of pts allowed
        "home_opp_suppression", "away_opp_suppression", # avg pts below opp season avg
        # Pythagorean luck
        "home_pyth_luck", "away_pyth_luck",
        "home_pyth_residual", "away_pyth_residual",
        # Schedule context
        "home_games_last_14", "away_games_last_14",
        # ATS rolling
        "home_ats_margin_10", "away_ats_margin_10",     # avg ATS margin last 10
        "home_ats_record_10", "away_ats_record_10",     # ATS W% last 10
        # Opponent-adjusted form
        "home_opp_adj_form", "away_opp_adj_form",
        # Common opponents
        "n_common_opps", "common_opp_diff",
    ]
    
    for col in basic_cols + advanced_cols:
        cols[col] = np.zeros(n)
    
    # Also compute game-level context
    cols["is_midweek"] = np.zeros(n)
    cols["season_phase"] = np.zeros(n)
    cols["is_early_season"] = np.zeros(n)
    cols["altitude_factor"] = np.zeros(n)
    cols["timezone_diff"] = np.zeros(n)
    cols["after_loss_either"] = np.zeros(n)
    
    def _get_prior(team, game_date, max_n=82):
        """Get prior games for a team before game_date."""
        log = team_games.get(team, [])
        return [g for g in log if g["date"] < game_date][-max_n:]
    
    def _linear_slope(values):
        """Compute slope of linear fit over values."""
        if len(values) < 3: return 0.0
        x = np.arange(len(values), dtype=float)
        try:
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except: return 0.0
    
    def _exp_weighted_mean(values, halflife=5):
        """Exponentially weighted mean with given halflife."""
        if not values: return 0.0
        n_v = len(values)
        weights = np.array([2 ** (-(n_v - 1 - i) / halflife) for i in range(n_v)])
        return float(np.average(values, weights=weights))
    
    def _pythagorean_wpct(pts_for, pts_against, exponent=14):
        """Pythagorean expected win% (Morey: exponent 14 for NBA)."""
        if pts_for <= 0: return 0.5
        return pts_for ** exponent / (pts_for ** exponent + pts_against ** exponent)
    
    for idx, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        d = row["game_date"]
        
        h_prior = _get_prior(h, d)
        a_prior = _get_prior(a, d)
        
        # ── Basic: W/L, form, rest, travel ──
        h_wins = sum(1 for g in h_prior if g["won"])
        h_losses = len(h_prior) - h_wins
        a_wins = sum(1 for g in a_prior if g["won"])
        a_losses = len(a_prior) - a_wins
        
        cols["home_wins"][idx] = h_wins
        cols["home_losses"][idx] = h_losses
        cols["away_wins"][idx] = a_wins
        cols["away_losses"][idx] = a_losses
        
        # Form (last 5, weighted)
        for side, prior, prefix in [(h, h_prior, "home"), (a, a_prior, "away")]:
            last5 = prior[-5:]
            if last5:
                form = sum((1 if g["won"] else -1) * (i+1) for i, g in enumerate(last5)) / 15.0
            else:
                form = 0.0
            cols[f"{prefix}_form"][idx] = round(form, 4)
        
        # Rest days
        for side, prior, prefix in [(h, h_prior, "home"), (a, a_prior, "away")]:
            if prior:
                try:
                    last_d = pd.Timestamp(prior[-1]["date"])
                    game_d = pd.Timestamp(d)
                    rest = max(0, min(14, (game_d - last_d).days - 1))
                except: rest = 2
            else:
                rest = 7
            cols[f"{prefix}_days_rest"][idx] = rest
        
        # Travel distance (away team)
        if a_prior:
            prev = a_prior[-1]
            prev_city = a if prev["home"] else prev["opp"]
            if prev_city in NBA_CITY_COORDS and h in NBA_CITY_COORDS:
                cols["away_travel_dist"][idx] = round(haversine(NBA_CITY_COORDS[prev_city], NBA_CITY_COORDS[h]))
        
        # ── Margin trend / acceleration ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            margins = [g["margin"] for g in prior[-10:]]
            cols[f"{prefix}_margin_trend"][idx] = _linear_slope(margins)
            if len(margins) >= 6:
                first_half = margins[:len(margins)//2]
                second_half = margins[len(margins)//2:]
                cols[f"{prefix}_margin_accel"][idx] = np.mean(second_half) - np.mean(first_half)
        
        # ── Streak ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            streak = 0
            for g in reversed(prior):
                if g["won"]:
                    if streak >= 0: streak += 1
                    else: break
                else:
                    if streak <= 0: streak -= 1
                    else: break
            cols[f"{prefix}_streak"][idx] = streak
        
        # ── Days since loss / games since blowout ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            days_since = 30  # default
            games_since_blowout = 20  # default
            for i, g in enumerate(reversed(prior)):
                if not g["won"] and days_since == 30:
                    try:
                        days_since = (pd.Timestamp(d) - pd.Timestamp(g["date"])).days
                    except: pass
                if g["margin"] < -15 and games_since_blowout == 20:
                    games_since_blowout = i
            cols[f"{prefix}_days_since_loss"][idx] = min(days_since, 30)
            cols[f"{prefix}_games_since_blowout"][idx] = min(games_since_blowout, 20)
        
        # ── WL momentum (exp-weighted) ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            wl_vals = [1.0 if g["won"] else 0.0 for g in prior[-15:]]
            cols[f"{prefix}_wl_momentum"][idx] = _exp_weighted_mean(wl_vals, halflife=5) if wl_vals else 0.5
        
        # ── Momentum halflife (exp-weighted margin) ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            margin_vals = [g["margin"] for g in prior[-15:]]
            cols[f"{prefix}_momentum_halflife"][idx] = _exp_weighted_mean(margin_vals, halflife=5) if margin_vals else 0.0
        
        # ── Win aging (recency-weighted win rate) ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            if len(prior) >= 5:
                recent = prior[-20:]
                weights = np.linspace(0.5, 1.0, len(recent))
                win_vals = np.array([1.0 if g["won"] else 0.0 for g in recent])
                cols[f"{prefix}_win_aging"][idx] = float(np.average(win_vals, weights=weights))
            else:
                cols[f"{prefix}_win_aging"][idx] = 0.5
        
        # ── Recovery (games since last loss) ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            recovery = 0
            for g in reversed(prior):
                if g["won"]: recovery += 1
                else: break
            cols[f"{prefix}_recovery"][idx] = min(recovery, 20)
        
        # ── After loss flag ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            cols[f"{prefix}_after_loss"][idx] = 1 if (prior and not prior[-1]["won"]) else 0
        cols["after_loss_either"][idx] = max(cols["home_after_loss"][idx], cols["away_after_loss"][idx])
        
        # ── Scoring distribution (last 15 games) ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            margins = [g["margin"] for g in prior[-15:]]
            if len(margins) >= 5:
                cols[f"{prefix}_scoring_var"][idx] = _safe_std(margins)
                cols[f"{prefix}_score_kurtosis"][idx] = _safe_kurtosis(margins)
                cols[f"{prefix}_margin_skew"][idx] = _safe_skew(margins)
                cols[f"{prefix}_ceiling"][idx] = float(np.percentile(margins, 90))
                cols[f"{prefix}_floor"][idx] = float(np.percentile(margins, 10))
                cols[f"{prefix}_bimodal"][idx] = abs(float(np.median(margins)) - float(np.mean(margins)))
                # Scoring entropy: discretize margins into bins
                bins = np.histogram(margins, bins=5)[0]
                probs = bins / max(bins.sum(), 1)
                probs = probs[probs > 0]
                cols[f"{prefix}_scoring_entropy"][idx] = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0
        
        # ── Defensive stability ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            allowed = [g["allowed"] for g in prior[-15:]]
            if len(allowed) >= 5:
                cols[f"{prefix}_def_stability"][idx] = _safe_std(allowed)
            
            # Opponent suppression: how much below their season avg do opponents score?
            # Simplified: just use std of allowed points (lower = more consistent defense)
            cols[f"{prefix}_opp_suppression"][idx] = -np.mean(allowed) if allowed else 0
        
        # ── Pythagorean luck ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            if len(prior) >= 10:
                total_scored = sum(g["scored"] for g in prior)
                total_allowed = sum(g["allowed"] for g in prior)
                actual_wpct = sum(1 for g in prior if g["won"]) / len(prior)
                expected_wpct = _pythagorean_wpct(total_scored, total_allowed)
                cols[f"{prefix}_pyth_luck"][idx] = actual_wpct - expected_wpct
                cols[f"{prefix}_pyth_residual"][idx] = actual_wpct - expected_wpct
        
        # ── Schedule: games in last 14 days ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            cutoff = pd.Timestamp(d) - pd.Timedelta(days=14)
            recent_count = sum(1 for g in prior if pd.Timestamp(g["date"]) >= cutoff)
            cols[f"{prefix}_games_last_14"][idx] = recent_count
        
        # ── ATS rolling (last 10 games with market data) ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            ats_margins = [g["ats_margin"] for g in prior[-10:] if g["ats_margin"] is not None]
            if ats_margins:
                cols[f"{prefix}_ats_margin_10"][idx] = np.mean(ats_margins)
                cols[f"{prefix}_ats_record_10"][idx] = np.mean([1 if m > 0 else 0 for m in ats_margins])
        
        # ── Opponent-adjusted form ──
        for prior, prefix in [(h_prior, "home"), (a_prior, "away")]:
            last5 = prior[-5:]
            if last5:
                adj_vals = []
                for g in last5:
                    opp_prior = _get_prior(g["opp"], g["date"])
                    opp_wpct = sum(1 for og in opp_prior if og["won"]) / max(len(opp_prior), 1) if opp_prior else 0.5
                    weight = 0.5 + opp_wpct  # Better opponents weight more
                    adj_vals.append((1 if g["won"] else -1) * weight)
                cols[f"{prefix}_opp_adj_form"][idx] = np.mean(adj_vals)
        
        # ── Common opponents ──
        h_opps = {g["opp"]: g["margin"] for g in h_prior}
        a_opps = {g["opp"]: g["margin"] for g in a_prior}
        common = set(h_opps.keys()) & set(a_opps.keys())
        cols["n_common_opps"][idx] = len(common)
        if common:
            h_vs_common = np.mean([h_opps[o] for o in common])
            a_vs_common = np.mean([a_opps[o] for o in common])
            cols["common_opp_diff"][idx] = h_vs_common - a_vs_common
        
        # ── Game-level context ──
        try:
            game_dt = pd.Timestamp(d)
            # Midweek (Tue/Wed/Thu)
            cols["is_midweek"][idx] = 1 if game_dt.dayofweek in [1, 2, 3] else 0
            # Season phase (0 = start, 1 = end)
            season = int(row.get("season", game_dt.year))
            season_start = pd.Timestamp(f"{season-1}-10-18")
            season_end = pd.Timestamp(f"{season}-06-15")
            total_days = (season_end - season_start).days
            elapsed = (game_dt - season_start).days
            cols["season_phase"][idx] = max(0, min(1, elapsed / max(total_days, 1)))
            # Early season
            min_games = min(len(h_prior), len(a_prior))
            cols["is_early_season"][idx] = 1 if min_games < 20 else 0
        except: pass
        
        # ── Altitude + timezone ──
        cols["altitude_factor"][idx] = 1 if h in NBA_ALTITUDE else 0
        h_tz = NBA_TIMEZONE.get(h, -6)
        a_tz = NBA_TIMEZONE.get(a, -6)
        # Away team's timezone shift (positive = traveling east, negative = west)
        if a_prior:
            prev = a_prior[-1]
            prev_city = a if prev["home"] else prev["opp"]
            prev_tz = NBA_TIMEZONE.get(prev_city, -6)
            cols["timezone_diff"][idx] = abs(h_tz - prev_tz)
        
        if idx % 2000 == 0 and idx > 0:
            print(f"    Processed {idx}/{n} games...")
    
    # ── Apply all columns to DataFrame ──
    for col_name, arr in cols.items():
        df[col_name] = arr
    
    # ── Fix tempo (from v1) ──
    for side in ["home", "away"]:
        tempo_col = f"{side}_tempo"
        ppg_col = f"{side}_ppg"
        fgpct_col = f"{side}_fgpct"
        if all(c in df.columns for c in [tempo_col, ppg_col, fgpct_col]):
            ppg = pd.to_numeric(df[ppg_col], errors="coerce").fillna(112)
            fgpct = pd.to_numeric(df[fgpct_col], errors="coerce").fillna(0.471)
            tempo = pd.to_numeric(df[tempo_col], errors="coerce")
            is_default = (tempo == 100.0) | tempo.isna()
            if is_default.sum() > 0:
                fga_est = ppg / fgpct.clip(0.35)
                fta_est = ppg * 0.22
                pace_est = fga_est - 10.5 + 14.0 + 0.475 * fta_est
                df.loc[is_default, tempo_col] = pace_est[is_default].clip(90, 108).round(1)
    
    # ── Diagnostics ──
    new_cols = [c for c in cols.keys() if c not in ["home_wins", "home_losses", "away_wins",
                "away_losses", "home_form", "away_form", "home_days_rest",
                "away_days_rest", "away_travel_dist"]]
    print(f"\n  Enrichment v2 complete:")
    print(f"    Basic cols: 9 (W/L, form, rest, travel)")
    print(f"    Advanced cols: {len(new_cols)}")
    print(f"    Total new columns: {len(cols)}")
    print(f"    home_margin_trend range: [{df['home_margin_trend'].min():.2f}, {df['home_margin_trend'].max():.2f}]")
    print(f"    home_streak range: [{df['home_streak'].min():.0f}, {df['home_streak'].max():.0f}]")
    print(f"    home_pyth_luck range: [{df['home_pyth_luck'].min():.3f}, {df['home_pyth_luck'].max():.3f}]")
    print(f"    n_common_opps mean: {df['n_common_opps'].mean():.1f}")
    print(f"    home_scoring_var mean: {df['home_scoring_var'].mean():.1f}")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  NBA Historical Data Enrichment v2")
    print("=" * 60)
    
    parquet_path = "nba_training_data.parquet"
    if not os.path.exists(parquet_path):
        print(f"  ERROR: {parquet_path} not found")
        sys.exit(1)
    
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} rows")
    
    t0 = time.time()
    df = enrich(df)
    print(f"\n  Enrichment took {time.time()-t0:.1f}s")
    
    df.to_parquet(parquet_path, index=False)
    print(f"  ✅ Saved to {parquet_path} ({os.path.getsize(parquet_path)/1024:.0f} KB, {len(df.columns)} columns)")
