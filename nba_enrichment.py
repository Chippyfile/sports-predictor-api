"""
nba_enrichment.py — Per-team advanced features from game history
================================================================
Reads nba_game_stats (last 15 games per team) and computes enrichment
features that enrich_v2 couldn't compute from a single-row DataFrame.

Stores results in nba_team_enrichment (team_abbr PK).

Called from:
  - /cron/nba-daily (after grading, alongside rolling PBP recompute)
  - /nba/backfill-enrichment (one-time backfill)

Read from:
  - nba_full_predict.py at prediction time via get_enrichment_diffs()
"""

import numpy as np
import requests
from datetime import datetime

try:
    from config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    import os
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Abbreviation map (same as nba_full_predict.py)
_ABBR_MAP = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "PHO": "PHX", "WSH": "WAS"}
def _map(a): return _ABBR_MAP.get(a, a)

NBA_TEAMS = [
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
]

# League averages (static defaults, good enough for enrichment)
LG_PPG = 114.0
LG_THREE_RATE = 0.38
LG_THREE_PCT = 0.365
LG_MARGIN = 0.0


def _sb():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    return SUPABASE_URL, headers


def compute_team_enrichment(team_abbr, n_games=15):
    """Compute enrichment features for one team from last N games.
    
    Returns dict of enrichment features, or empty dict if insufficient data.
    """
    url, headers = _sb()
    team_abbr = _map(team_abbr)
    
    # Fetch last N games from nba_game_stats
    q = (f"{url}/rest/v1/nba_game_stats"
         f"?team_abbr=eq.{team_abbr}"
         f"&order=game_date.desc&limit={n_games}"
         f"&select=game_date,actual_margin,bench_pts,paint_pts,fast_break_pts,"
         f"second_chance_pts,largest_lead,q4_scoring,max_run,three_fg_rate,"
         f"ft_trip_rate,oreb")
    try:
        rows = requests.get(q, headers={**headers, "Prefer": ""}, timeout=10).json() or []
    except Exception as e:
        print(f"  [enrichment] fetch error for {team_abbr}: {e}")
        return {}
    
    if len(rows) < 5:
        print(f"  [enrichment] {team_abbr}: only {len(rows)} games, need 5+")
        return {}
    
    n = len(rows)
    margins = [float(r.get("actual_margin", 0) or 0) for r in rows]
    bench = [float(r.get("bench_pts", 0) or 0) for r in rows]
    paint = [float(r.get("paint_pts", 0) or 0) for r in rows]
    fast_break = [float(r.get("fast_break_pts", 0) or 0) for r in rows]
    second_chance = [float(r.get("second_chance_pts", 0) or 0) for r in rows]
    largest_leads = [float(r.get("largest_lead", 0) or 0) for r in rows]
    q4_scores = [float(r.get("q4_scoring", 0) or 0) for r in rows]
    max_runs = [float(r.get("max_run", 0) or 0) for r in rows]
    three_rates = [float(r.get("three_fg_rate", 0) or 0) for r in rows]
    ft_trip_rates = [float(r.get("ft_trip_rate", 0) or 0) for r in rows]
    
    margin_mean = float(np.mean(margins))
    margin_std = float(np.std(margins))
    
    # ── scoring_var: volatility of results ──
    scoring_var = round(margin_std, 2)
    
    # ── consistency: coefficient of variation (lower = more consistent) ──
    # Bounded to avoid division by near-zero means
    consistency = round(margin_std / max(abs(margin_mean), 1.0), 3)
    
    # ── ceiling: best performance in window ──
    ceiling = round(float(np.max(margins)), 1)
    
    # ── floor: worst performance ──
    floor = round(float(np.min(margins)), 1)
    
    # ── bimodal: kurtosis of margin distribution ──
    # Negative kurtosis = bimodal/flat, positive = peaked
    if n >= 5:
        m4 = float(np.mean((np.array(margins) - margin_mean) ** 4))
        bimodal = round(m4 / max(margin_std ** 4, 0.01) - 3.0, 3)  # excess kurtosis
    else:
        bimodal = 0.0
    
    # ── scoring_entropy: diversity of scoring sources ──
    # Based on proportions: bench, paint, fast_break, second_chance, "other"
    avg_bench = float(np.mean(bench))
    avg_paint = float(np.mean(paint))
    avg_fb = float(np.mean(fast_break))
    avg_sc = float(np.mean(second_chance))
    total_special = avg_bench + avg_paint + avg_fb + avg_sc
    if total_special > 0:
        props = np.array([avg_bench, avg_paint, avg_fb, avg_sc]) / total_special
        props = props[props > 0]  # remove zeros for log
        scoring_entropy = round(float(-np.sum(props * np.log2(props))), 3)
    else:
        scoring_entropy = 1.5  # default moderate entropy
    
    # ── def_stability: std of opponent implied scores ──
    # We don't have opponent scores directly, but margin = our_score - opp_score
    # If our avg score ≈ LG_PPG + margin_mean/2, then opp_score ≈ LG_PPG - margin_mean/2
    # Std of opp scores ≈ std of margins (approximately, since our scoring variance contributes too)
    # This is a rough proxy but better than zero
    def_stability = round(margin_std * 0.6, 2)  # ~60% of margin variance comes from defense
    
    # ── opp_suppression: how much team suppresses opponents below league average ──
    # Positive = good defense (opponents score below average)
    opp_suppression = round(margin_mean / 2.0, 2)  # rough proxy
    
    # ── three_value: value generated from 3-point shooting ──
    avg_3rate = float(np.mean(three_rates))
    three_value = round(avg_3rate * (avg_3rate - LG_THREE_RATE), 4)
    
    # ── ts_regression: deviation of TS-proxy from expected ──
    # We don't have per-game TS%, but three_fg_rate and ft_trip_rate are proxies
    avg_ft_rate = float(np.mean(ft_trip_rates))
    ts_proxy = avg_3rate * 0.4 + avg_ft_rate * 0.3 + 0.45  # rough TS approximation
    ts_regression = round(ts_proxy - 0.58, 3)  # 0.58 = league avg TS%
    
    # ── three_pt_regression: 3P% deviation from mean ──
    # We have three_fg_rate (3PM/FGA) not 3P%, but it correlates
    three_pt_regression = round(avg_3rate - LG_THREE_RATE, 4)
    
    # ── pace_leverage: how much pace matters for this team ──
    # Higher absolute margin teams benefit more from pace
    pace_leverage = round(abs(margin_mean) * 0.01, 4)
    
    # ── pace_control: consistency of play style ──
    # Teams with consistent scoring sources have more pace control
    pace_control = round(1.0 / max(consistency, 0.3), 3)
    
    # ── matchup-relevant stats (stored per-team, diff computed at prediction time) ──
    # These approximate what enrich_v2 computes for matchup features
    avg_ft_trip = float(np.mean(ft_trip_rates))
    avg_oreb = float(np.mean([float(r.get("oreb", 0) or 0) for r in rows]))
    
    return {
        "team_abbr": team_abbr,
        "updated_date": rows[0].get("game_date", datetime.now().strftime("%Y-%m-%d")),
        "games_counted": n,
        "scoring_var": scoring_var,
        "consistency": consistency,
        "ceiling": ceiling,
        "floor": floor,
        "bimodal": bimodal,
        "scoring_entropy": scoring_entropy,
        "def_stability": def_stability,
        "opp_suppression": opp_suppression,
        "three_value": three_value,
        "ts_regression": ts_regression,
        "three_pt_regression": three_pt_regression,
        "pace_leverage": pace_leverage,
        "pace_control": pace_control,
        # Per-team stats for matchup computation at prediction time
        "avg_ft_trip_rate": round(avg_ft_trip, 4),
        "avg_three_fg_rate": round(avg_3rate, 4),
        "avg_oreb": round(avg_oreb, 2),
        "avg_margin": round(margin_mean, 2),
        "margin_trend": round(float(np.mean(margins[:5]) - np.mean(margins[5:])), 2) if n >= 10 else 0.0,
    }


def save_team_enrichment(enrichment):
    """Upsert enrichment features to nba_team_enrichment table."""
    url, headers = _sb()
    team = enrichment["team_abbr"]
    try:
        resp = requests.post(
            f"{url}/rest/v1/nba_team_enrichment",
            json=enrichment,
            headers={**headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
            timeout=10
        )
        return resp.ok
    except Exception as e:
        print(f"  [enrichment] save error for {team}: {e}")
        return False


def recompute_all_enrichment():
    """Recompute enrichment features for all 30 NBA teams."""
    success = 0
    for team in NBA_TEAMS:
        enrichment = compute_team_enrichment(team)
        if enrichment:
            if save_team_enrichment(enrichment):
                success += 1
            else:
                print(f"  [enrichment] save failed for {team}")
        else:
            print(f"  [enrichment] skipped {team} (insufficient data)")
    print(f"  [enrichment] Updated {success}/{len(NBA_TEAMS)} teams")
    return success


def get_team_enrichment(team_abbr):
    """Read pre-computed enrichment features for a team. Returns dict or None."""
    url, headers = _sb()
    team_abbr = _map(team_abbr)
    q = f"{url}/rest/v1/nba_team_enrichment?team_abbr=eq.{team_abbr}&select=*&limit=1"
    try:
        rows = requests.get(q, headers={**headers, "Prefer": ""}, timeout=10).json() or []
        return rows[0] if rows else None
    except Exception:
        return None


def get_enrichment_diffs(home_abbr, away_abbr):
    """Get enrichment feature diffs for a matchup. Returns dict of feature overrides."""
    h = get_team_enrichment(home_abbr)
    a = get_team_enrichment(away_abbr)
    if not h or not a:
        return {}
    
    diffs = {}
    
    # Direct diff features (home - away)
    for feat in ["scoring_var", "consistency", "ceiling", "bimodal",
                 "scoring_entropy", "def_stability", "opp_suppression",
                 "three_value", "ts_regression", "three_pt_regression",
                 "pace_leverage", "pace_control"]:
        h_val = float(h.get(feat, 0) or 0)
        a_val = float(a.get(feat, 0) or 0)
        diffs[f"{feat}_diff"] = round(h_val - a_val, 4)
    
    # ── Matchup features (cross-team interactions) ──
    # matchup_efg: home team's shooting advantage vs away team's defense
    h_3rate = float(h.get("avg_three_fg_rate", 0.38) or 0.38)
    a_3rate = float(a.get("avg_three_fg_rate", 0.38) or 0.38)
    # eFG proxy advantage: difference in three_fg_rate (higher = more efficient)
    diffs["matchup_efg"] = round(h_3rate - a_3rate, 4)
    
    # matchup_to: home TO advantage (lower is better for home)
    # Approximate from margin (better teams = fewer TO situations)
    h_margin = float(h.get("avg_margin", 0) or 0)
    a_margin = float(a.get("avg_margin", 0) or 0)
    diffs["matchup_to"] = round((h_margin - a_margin) * 0.02, 4)
    
    # matchup_orb: home rebounding advantage
    h_oreb = float(h.get("avg_oreb", 5) or 5)
    a_oreb = float(a.get("avg_oreb", 5) or 5)
    diffs["matchup_orb"] = round(h_oreb - a_oreb, 2)
    
    # matchup_ft: FT generation advantage
    h_ft = float(h.get("avg_ft_trip_rate", 0.25) or 0.25)
    a_ft = float(a.get("avg_ft_trip_rate", 0.25) or 0.25)
    diffs["matchup_ft"] = round(h_ft - a_ft, 4)
    
    return diffs
