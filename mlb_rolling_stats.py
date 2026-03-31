#!/usr/bin/env python3
"""
mlb_rolling_stats.py — Compute and store MLB team rolling stats + ump profiles
═══════════════════════════════════════════════════════════════════════════════
Enhanced: Fetches real batting/fielding/linescore data from MLB Stats API
Tables: mlb_team_rolling (30 rows), mlb_ump_profiles (~100 rows)

Usage:
  python3 mlb_rolling_stats.py              # Seed from current season
  python3 mlb_rolling_stats.py --full       # Seed + ump profiles from historical
"""
import sys, os, time
import numpy as np
import requests as _req

MLB_API = "https://statsapi.mlb.com/api/v1"
SEASON = 2026
LOOKBACK = 20
MIN_GAMES = 5

# MLB team IDs → abbreviations
MLB_TEAMS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

def _sb():
    from config import SUPABASE_URL, SUPABASE_KEY
    return (SUPABASE_URL, {
        "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates",
    })

def sb_get(table, query=""):
    url, headers = _sb()
    r = _req.get(f"{url}/rest/v1/{table}?{query}", headers=headers, timeout=15)
    return r.json() if r.ok else []

def sb_upsert(table, row):
    url, headers = _sb()
    r = _req.post(f"{url}/rest/v1/{table}", json=row, headers=headers, timeout=15)
    return r.ok


def _mlb_fetch(endpoint, params=None):
    """Fetch from MLB Stats API."""
    try:
        r = _req.get(f"{MLB_API}/{endpoint}", params=params, timeout=15)
        return r.json() if r.ok else None
    except Exception:
        return None


def fetch_team_batting(team_id):
    """Fetch team batting stats (season-to-date) for BABIP computation."""
    data = _mlb_fetch(f"teams/{team_id}/stats", {"stats": "season", "group": "hitting", "season": SEASON, "sportId": 1})
    s = data.get("stats", [{}])[0].get("splits", [{}])[0].get("stat") if data else None
    if not s:
        return None
    return {
        "h": int(s.get("hits", 0)),
        "hr": int(s.get("homeRuns", 0)),
        "ab": int(s.get("atBats", 0)),
        "k": int(s.get("strikeOuts", 0)),
        "sf": int(s.get("sacFlies", 0)),
        "hbp": int(s.get("hitByPitch", 0)),
        "bb": int(s.get("baseOnBalls", 0)),
        "gp": int(s.get("gamesPlayed", 0)),
    }


def fetch_team_fielding(team_id):
    """Fetch team fielding stats for defensive quality proxy."""
    data = _mlb_fetch(f"teams/{team_id}/stats", {"stats": "season", "group": "fielding", "season": SEASON, "sportId": 1})
    s = data.get("stats", [{}])[0].get("splits", [{}])[0].get("stat") if data else None
    if not s:
        return None
    return {
        "errors": int(s.get("errors", 0)),
        "dp": int(s.get("doublePlays", 0)),
        "assists": int(s.get("assists", 0)),
        "gp": int(s.get("gamesPlayed", 0)),
    }


def fetch_recent_linescores(team_id, n=20):
    """Fetch recent game linescores for first-inning rate."""
    today = time.strftime("%Y-%m-%d")
    data = _mlb_fetch("schedule", {
        "teamId": team_id, "season": SEASON,
        "startDate": f"{SEASON}-01-01", "endDate": today,
        "hydrate": "linescore", "sportId": 1,
    })
    if not data:
        return []
    
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue
            is_home = g.get("teams", {}).get("home", {}).get("team", {}).get("id") == team_id
            team_side = "home" if is_home else "away"
            
            # First inning runs
            innings = g.get("linescore", {}).get("innings", [])
            first_inn_runs = 0
            if innings:
                first_inn = innings[0]
                first_inn_runs = int(first_inn.get(team_side, {}).get("runs", 0) or 0)
            
            # Total runs
            my_score = g.get("teams", {}).get(team_side, {}).get("score", 0) or 0
            opp_side = "away" if is_home else "home"
            opp_score = g.get("teams", {}).get(opp_side, {}).get("score", 0) or 0
            
            games.append({
                "rs": int(my_score),
                "ra": int(opp_score),
                "first_inn": first_inn_runs,
                "date": g.get("gameDate", d.get("date", "")),
            })
    
    return games[-n:]  # most recent N


def compute_team_rolling(team_id, abbr):
    """Compute all rolling features for a team from MLB Stats API."""
    batting = fetch_team_batting(team_id)
    fielding = fetch_team_fielding(team_id)
    recent = fetch_recent_linescores(team_id, LOOKBACK)
    
    if not recent or len(recent) < MIN_GAMES:
        return None
    
    # ── BABIP luck ──
    babip_luck = 0.0
    if batting and batting["ab"] > 0:
        denom = batting["ab"] - batting["k"] - batting["hr"] + batting["sf"]
        if denom > 0:
            babip = (batting["h"] - batting["hr"]) / denom
            babip_luck = round(babip - 0.300, 4)
    
    # ── First-inning rate ──
    first_inn_runs = [g["first_inn"] for g in recent]
    first_inn_rate = round(np.mean(first_inn_runs), 4)
    
    # ── Defensive quality proxy ──
    # Normalized: (DP*2 + assists*0.5 - errors*3) / games_played
    def_quality = 0.0
    if fielding and fielding["gp"] > 0:
        raw = (fielding["dp"] * 2 + fielding["assists"] * 0.5 - fielding["errors"] * 3) / fielding["gp"]
        # Normalize to roughly [-1, +1] range (mean is ~5-7, so center and scale)
        def_quality = round((raw - 6.0) / 3.0, 4)
    
    # ── Pythagorean residual, entropy, clutch, opp-adj ──
    rs = [g["rs"] for g in recent]
    ra = [g["ra"] for g in recent]
    wins = [1.0 if g["rs"] > g["ra"] else 0.0 for g in recent]
    n = len(recent)
    
    # Pythagorean
    rs_tot, ra_tot = sum(rs), sum(ra)
    if rs_tot + ra_tot > 0:
        exp = max(1.5, ((rs_tot + ra_tot) / n) ** 0.287)
        pyth_wp = rs_tot**exp / (rs_tot**exp + ra_tot**exp)
    else:
        pyth_wp = 0.5
    actual_wp = np.mean(wins)
    pyth_residual = round(actual_wp - pyth_wp, 4)
    
    # Scoring entropy
    hist, _ = np.histogram(rs, bins=8, range=(0, 15))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    entropy = round(float(-np.sum(probs * np.log2(probs))), 4) if len(probs) > 0 else 2.5
    
    # Clutch divergence
    margins = [abs(r - a) for r, a in zip(rs, ra)]
    close_mask = [m <= 2 for m in margins]
    if sum(close_mask) >= 3:
        close_wp = np.mean([w for w, c in zip(wins, close_mask) if c])
        clutch_div = round(close_wp - actual_wp, 4)
    else:
        clutch_div = 0.0
    
    # Opp-adjusted form
    opp_quality = [r / max(np.mean(ra), 1) for r in ra]
    weighted = np.mean([w * q for w, q in zip(wins, opp_quality)])
    opp_adj = round(weighted - actual_wp, 4)
    
    return {
        "team_abbr": abbr,
        "updated_date": time.strftime("%Y-%m-%d"),
        "games_counted": n,
        "roll_pyth_residual": pyth_residual,
        "roll_babip_luck": babip_luck,
        "roll_scoring_entropy": entropy,
        "roll_first_inn_rate": first_inn_rate,
        "roll_clutch_divergence": clutch_div,
        "roll_opp_adj_form": opp_adj,
        "roll_runs_scored_avg": round(np.mean(rs), 3),
        "roll_runs_allowed_avg": round(np.mean(ra), 3),
        "roll_win_pct": round(actual_wp, 4),
        "roll_def_quality": def_quality,
    }


def get_rolling_features(home_team, away_team, ump_name=None):
    """Read rolling features for a prediction. Called by predict_mlb()."""
    result = {
        "pyth_residual_diff": 0.0, "babip_luck_diff": 0.0,
        "scoring_entropy_diff": 0.0, "first_inn_rate_diff": 0.0,
        "clutch_divergence_diff": 0.0, "opp_adj_form_diff": 0.0,
        "ump_run_env": 8.5, "scoring_entropy_combined": 5.0,
        "first_inn_rate_combined": 0.8, "def_oaa_diff": 0.0,
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
            
            # Defensive quality → def_oaa_diff proxy
            hd = float(h.get("roll_def_quality", 0) or 0)
            ad = float(a.get("roll_def_quality", 0) or 0)
            result["def_oaa_diff"] = round(hd - ad, 4)
    except Exception as e:
        print(f"  [mlb_rolling] team read error: {e}")
    
    # Umpire profile
    if ump_name:
        try:
            safe_name = ump_name.replace("'", "''")
            ump_row = sb_get("mlb_ump_profiles", f"ump_name=eq.{safe_name}&limit=1")
            if ump_row:
                u = ump_row[0] if isinstance(ump_row, list) else ump_row
                result["ump_run_env"] = float(u.get("avg_total_runs", 8.5) or 8.5)
        except Exception as e:
            print(f"  [mlb_rolling] ump read error: {e}")
    
    return result


def seed_all():
    """Seed rolling stats from MLB Stats API for all 30 teams."""
    print("\n  Seeding team rolling stats from MLB Stats API...")
    
    seeded = 0
    for team_id, abbr in sorted(MLB_TEAMS.items(), key=lambda x: x[1]):
        try:
            stats = compute_team_rolling(team_id, abbr)
            if stats:
                if sb_upsert("mlb_team_rolling", stats):
                    seeded += 1
                print(f"    {abbr}: {stats['games_counted']}g, pyth={stats['roll_pyth_residual']:+.3f}, "
                      f"babip={stats['roll_babip_luck']:+.3f}, 1st_inn={stats['roll_first_inn_rate']:.2f}, "
                      f"def={stats.get('roll_def_quality', 0):+.3f}")
            else:
                print(f"    {abbr}: insufficient data")
        except Exception as e:
            print(f"    {abbr}: ERROR {e}")
        time.sleep(0.5)  # rate limit
    
    print(f"\n  ✅ Seeded {seeded}/30 teams")
    
    # Seed umpire profiles from mlb_historical
    if "--full" in sys.argv:
        print("\n  Seeding umpire profiles from mlb_historical...")
        umps = sb_get("mlb_historical", "select=umpire,actual_home_runs,actual_away_runs,game_date&limit=50000&umpire=not.is.null")
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
            if sb_upsert("mlb_ump_profiles", {
                "ump_name": ump,
                "games_umped": len(data["runs"]),
                "avg_total_runs": round(np.mean(data["runs"]), 3),
                "updated_date": data["last_date"],
            }):
                ump_seeded += 1
        print(f"  ✅ Seeded {ump_seeded} umpire profiles")


if __name__ == "__main__":
    seed_all()
