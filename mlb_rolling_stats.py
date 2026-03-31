#!/usr/bin/env python3
"""
mlb_rolling_stats.py — Compute, seed, and update MLB team rolling stats + ump profiles
═══════════════════════════════════════════════════════════════════════════════════════
Tables:
  mlb_team_rolling  — 30 rows, one per team, rolling 20-game stats
  mlb_ump_profiles  — ~100 rows, one per active umpire

Functions:
  seed_rolling_stats()      — Initial seed from mlb_predictions (run once)
  update_team_rolling()     — Update one team after a completed game
  update_ump_profile()      — Update umpire profile after a completed game
  get_rolling_features()    — Read features for a prediction (home_team, away_team, ump)

Usage:
  python3 mlb_rolling_stats.py              # Seed from current season data
  python3 mlb_rolling_stats.py --full       # Seed + ump profiles
"""
import sys, os, math
import numpy as np
import requests

# ── Supabase helpers ──
def _sb():
    from config import SUPABASE_URL, SUPABASE_KEY
    return (SUPABASE_URL, {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    })

def sb_get(table, query=""):
    url, headers = _sb()
    r = requests.get(f"{url}/rest/v1/{table}?{query}", headers=headers, timeout=15)
    return r.json() if r.ok else []

def sb_upsert(table, row):
    url, headers = _sb()
    r = requests.post(f"{url}/rest/v1/{table}", json=row, headers=headers, timeout=15)
    return r.ok


# ── Rolling stat computations ──
LOOKBACK = 20
MIN_GAMES = 5

def _pyth_wp(rs_arr, ra_arr):
    """Pythagorean win% from runs scored/allowed arrays."""
    rs, ra = sum(rs_arr), sum(ra_arr)
    if rs + ra == 0:
        return 0.5
    n = len(rs_arr)
    exp = max(1.5, ((rs + ra) / n) ** 0.287)
    return rs**exp / (rs**exp + ra**exp)

def _shannon_entropy(values, bins=8):
    """Shannon entropy of scoring distribution."""
    if len(values) < 3:
        return 2.5
    hist, _ = np.histogram(values, bins=bins, range=(0, 15))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def compute_rolling(games):
    """
    Compute rolling features from a list of recent games.
    Each game: {"rs": runs_scored, "ra": runs_allowed, "first_inn": first_inning_runs,
                "h": hits, "hr": home_runs, "ab": at_bats, "k": strikeouts, "sf": sac_flies}
    Returns dict of rolling features or None if insufficient games.
    """
    if len(games) < MIN_GAMES:
        return None
    
    recent = games[-LOOKBACK:]
    n = len(recent)
    
    rs = [g["rs"] for g in recent]
    ra = [g["ra"] for g in recent]
    wins = [1.0 if g["rs"] > g["ra"] else 0.0 for g in recent]
    
    # 1. Pythagorean residual
    pyth_wp = _pyth_wp(rs, ra)
    actual_wp = np.mean(wins)
    pyth_residual = round(actual_wp - pyth_wp, 4)
    
    # 2. BABIP luck
    babips = []
    for g in recent:
        denom = g.get("ab", 30) - g.get("k", 5) - g.get("hr", 1) + g.get("sf", 0)
        if denom > 0:
            babips.append((g.get("h", 8) - g.get("hr", 1)) / denom)
    babip_luck = round(np.mean(babips) - 0.300, 4) if babips else 0.0
    
    # 3. Scoring entropy
    entropy = round(_shannon_entropy(rs), 4)
    
    # 4. First inning rate
    first_inns = [g.get("first_inn", 0) for g in recent]
    first_inn_rate = round(np.mean(first_inns), 4)
    
    # 5. Clutch divergence
    margins = [abs(g["rs"] - g["ra"]) for g in recent]
    close_mask = [m <= 2 for m in margins]
    if sum(close_mask) >= 3:
        close_wp = np.mean([w for w, c in zip(wins, close_mask) if c])
        clutch_div = round(close_wp - actual_wp, 4)
    else:
        clutch_div = 0.0
    
    # 6. Opponent-adjusted form
    opp_quality = [r / max(np.mean(ra), 1) for r in ra]
    weighted_wins = np.mean([w * q for w, q in zip(wins, opp_quality)])
    opp_adj_form = round(weighted_wins - actual_wp, 4)
    
    return {
        "roll_pyth_residual": pyth_residual,
        "roll_babip_luck": babip_luck,
        "roll_scoring_entropy": entropy,
        "roll_first_inn_rate": first_inn_rate,
        "roll_clutch_divergence": clutch_div,
        "roll_opp_adj_form": opp_adj_form,
        "roll_runs_scored_avg": round(np.mean(rs), 3),
        "roll_runs_allowed_avg": round(np.mean(ra), 3),
        "roll_win_pct": round(actual_wp, 4),
        "games_counted": n,
    }


def get_rolling_features(home_team, away_team, ump_name=None):
    """
    Read rolling features for a prediction. Returns dict ready to merge into feature row.
    Called by predict_mlb() at serve time.
    """
    result = {
        "pyth_residual_diff": 0.0,
        "babip_luck_diff": 0.0,
        "scoring_entropy_diff": 0.0,
        "first_inn_rate_diff": 0.0,
        "clutch_divergence_diff": 0.0,
        "opp_adj_form_diff": 0.0,
        "ump_run_env": 8.5,
        "scoring_entropy_combined": 5.0,
        "first_inn_rate_combined": 0.8,
    }
    
    try:
        home_row = sb_get("mlb_team_rolling", f"team_abbr=eq.{home_team}&limit=1")
        away_row = sb_get("mlb_team_rolling", f"team_abbr=eq.{away_team}&limit=1")
        
        if home_row and away_row:
            h = home_row[0] if isinstance(home_row, list) else home_row
            a = away_row[0] if isinstance(away_row, list) else away_row
            
            for feat in ["pyth_residual", "babip_luck", "scoring_entropy",
                         "first_inn_rate", "clutch_divergence", "opp_adj_form"]:
                hv = float(h.get(f"roll_{feat}", 0) or 0)
                av = float(a.get(f"roll_{feat}", 0) or 0)
                result[f"{feat}_diff"] = round(hv - av, 4)
            
            he = float(h.get("roll_scoring_entropy", 2.5) or 2.5)
            ae = float(a.get("roll_scoring_entropy", 2.5) or 2.5)
            result["scoring_entropy_combined"] = round(he + ae, 3)
            
            hf = float(h.get("roll_first_inn_rate", 0.4) or 0.4)
            af = float(a.get("roll_first_inn_rate", 0.4) or 0.4)
            result["first_inn_rate_combined"] = round(hf + af, 3)
    except Exception as e:
        print(f"  [mlb_rolling] team read error: {e}")
    
    # Umpire profile
    if ump_name:
        try:
            ump_row = sb_get("mlb_ump_profiles", f"ump_name=eq.{ump_name}&limit=1")
            if ump_row:
                u = ump_row[0] if isinstance(ump_row, list) else ump_row
                result["ump_run_env"] = float(u.get("avg_total_runs", 8.5) or 8.5)
        except Exception as e:
            print(f"  [mlb_rolling] ump read error: {e}")
    
    return result


def update_team_rolling(team_abbr):
    """
    Recompute rolling stats for a team from recent mlb_predictions results.
    Called after grading a game.
    """
    rows = sb_get(
        "mlb_predictions",
        f"result_entered=eq.true&or=(home_team.eq.{team_abbr},away_team.eq.{team_abbr})"
        f"&order=game_date.desc&limit={LOOKBACK}"
        f"&select=game_date,home_team,away_team,actual_home_runs,actual_away_runs"
    )
    if not rows or len(rows) < MIN_GAMES:
        return False
    
    games = []
    for r in reversed(rows):  # oldest first
        is_home = r["home_team"] == team_abbr
        rs = float(r["actual_home_runs"] or 0) if is_home else float(r["actual_away_runs"] or 0)
        ra = float(r["actual_away_runs"] or 0) if is_home else float(r["actual_home_runs"] or 0)
        games.append({"rs": rs, "ra": ra, "first_inn": 0, "h": 0, "hr": 0, "ab": 30, "k": 5, "sf": 0})
    
    stats = compute_rolling(games)
    if not stats:
        return False
    
    stats["team_abbr"] = team_abbr
    stats["updated_date"] = rows[0]["game_date"]
    return sb_upsert("mlb_team_rolling", stats)


def update_ump_profile(ump_name, game_total_runs, game_date):
    """Update umpire's running average after a completed game."""
    existing = sb_get("mlb_ump_profiles", f"ump_name=eq.{ump_name}&limit=1")
    if existing:
        u = existing[0]
        n = int(u.get("games_umped", 0) or 0)
        avg = float(u.get("avg_total_runs", 8.5) or 8.5)
        # Exponential moving average (weight recent games more)
        new_n = n + 1
        alpha = min(0.1, 2.0 / (new_n + 1))
        new_avg = avg * (1 - alpha) + game_total_runs * alpha
    else:
        new_n = 1
        new_avg = game_total_runs
    
    return sb_upsert("mlb_ump_profiles", {
        "ump_name": ump_name,
        "games_umped": new_n,
        "avg_total_runs": round(new_avg, 3),
        "updated_date": game_date,
    })


# ═══════════════════════════════════════════════════════════
# SEED SCRIPT
# ═══════════════════════════════════════════════════════════
def seed_all():
    """Seed rolling stats from mlb_predictions + ump profiles from mlb_historical."""
    print("\n  Seeding team rolling stats from mlb_predictions...")
    
    # Get all completed games
    rows = sb_get(
        "mlb_predictions",
        "result_entered=eq.true&order=game_date.asc&limit=5000"
        "&select=game_date,home_team,away_team,actual_home_runs,actual_away_runs"
    )
    if not rows:
        print("  No completed games found!")
        return
    
    print(f"  {len(rows)} completed games")
    
    # Build per-team game lists
    team_games = {}
    for r in rows:
        ht, at = r["home_team"], r["away_team"]
        hrs = float(r.get("actual_home_runs") or 0)
        ars = float(r.get("actual_away_runs") or 0)
        
        for team, rs, ra in [(ht, hrs, ars), (at, ars, hrs)]:
            if team not in team_games:
                team_games[team] = []
            team_games[team].append({
                "rs": rs, "ra": ra, "first_inn": 0,
                "h": 0, "hr": 0, "ab": 30, "k": 5, "sf": 0
            })
    
    # Compute and upsert rolling stats
    seeded = 0
    for team, games in team_games.items():
        stats = compute_rolling(games)
        if stats:
            stats["team_abbr"] = team
            stats["updated_date"] = rows[-1]["game_date"]
            if sb_upsert("mlb_team_rolling", stats):
                seeded += 1
            print(f"    {team}: {stats['games_counted']} games, pyth_res={stats['roll_pyth_residual']:+.3f}, "
                  f"babip={stats['roll_babip_luck']:+.3f}, entropy={stats['roll_scoring_entropy']:.2f}")
    
    print(f"\n  ✅ Seeded {seeded} teams")
    
    # Seed umpire profiles from mlb_historical
    if "--full" in sys.argv:
        print("\n  Seeding umpire profiles from mlb_historical...")
        umps = sb_get("mlb_historical", "select=umpire,actual_home_runs,actual_away_runs,game_date&limit=50000")
        ump_stats = {}
        for r in umps:
            ump = r.get("umpire", "")
            if not ump:
                continue
            total = float(r.get("actual_home_runs") or 0) + float(r.get("actual_away_runs") or 0)
            if ump not in ump_stats:
                ump_stats[ump] = {"runs": [], "last_date": ""}
            ump_stats[ump]["runs"].append(total)
            ump_stats[ump]["last_date"] = r.get("game_date", "")
        
        ump_seeded = 0
        for ump, data in ump_stats.items():
            if len(data["runs"]) < 10:
                continue
            avg = round(np.mean(data["runs"]), 3)
            if sb_upsert("mlb_ump_profiles", {
                "ump_name": ump,
                "games_umped": len(data["runs"]),
                "avg_total_runs": avg,
                "updated_date": data["last_date"],
            }):
                ump_seeded += 1
        
        print(f"  ✅ Seeded {ump_seeded} umpire profiles")


if __name__ == "__main__":
    seed_all()
