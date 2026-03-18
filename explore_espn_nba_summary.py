"""
ESPN NBA Summary API Explorer
Run this locally to see EXACTLY what data is available from the /summary endpoint.

Usage:
  python explore_espn_nba_summary.py
  python explore_espn_nba_summary.py 401810849    # specific game ID

This will:
1. Fetch a completed game summary → show all available fields
2. Fetch an upcoming/today game summary → show what's available PRE-GAME
3. Print a structured report of what we can extract for ML features
"""

import requests, json, sys, os
from datetime import datetime

def fetch_summary(game_id):
    """Fetch ESPN NBA game summary."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    r = requests.get(url, timeout=15)
    if not r.ok:
        print(f"  ERROR: {r.status_code} for game {game_id}")
        return None
    return r.json()

def fetch_today_games():
    """Get today's NBA game IDs from scoreboard."""
    today = datetime.now().strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}&limit=50"
    r = requests.get(url, timeout=15)
    if not r.ok:
        return []
    data = r.json()
    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})
        home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), {})
        away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), {})
        games.append({
            "id": ev.get("id"),
            "home": home.get("team", {}).get("abbreviation", "?"),
            "away": away.get("team", {}).get("abbreviation", "?"),
            "status": "Final" if status.get("completed") else "Preview" if status.get("state") == "pre" else "Live",
        })
    return games

def explore_keys(data, prefix="", depth=0, max_depth=3):
    """Recursively explore JSON structure."""
    if depth > max_depth:
        return []
    results = []
    if isinstance(data, dict):
        for key, val in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                results.append((path, "dict", list(val.keys())[:10]))
                results.extend(explore_keys(val, path, depth + 1, max_depth))
            elif isinstance(val, list):
                results.append((path, f"list[{len(val)}]", type(val[0]).__name__ if val else "empty"))
                if val and isinstance(val[0], dict):
                    results.extend(explore_keys(val[0], f"{path}[0]", depth + 1, max_depth))
            else:
                results.append((path, type(val).__name__, str(val)[:80]))
    return results

def print_section(title, data, depth=3):
    """Print a structured exploration of a data section."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if data is None:
        print("  [not present]")
        return
    keys = explore_keys(data, max_depth=depth)
    for path, typ, preview in keys:
        indent = "  " * (path.count(".") + 1)
        print(f"{indent}{path}: ({typ}) {preview}")


def extract_ml_features(data):
    """Extract all ML-relevant features from a game summary."""
    features = {}
    
    # ── 1. HEADER: Teams, scores, odds ──
    header = data.get("header", {})
    comps = header.get("competitions", [{}])
    if comps:
        comp = comps[0]
        for c in comp.get("competitors", []):
            side = "home" if c.get("homeAway") == "home" else "away"
            team = c.get("team", {})
            features[f"{side}_team_id"] = team.get("id")
            features[f"{side}_team_abbr"] = team.get("abbreviation")
            features[f"{side}_team_name"] = team.get("displayName")
            features[f"{side}_score"] = c.get("score")
            features[f"{side}_record"] = c.get("record", [{}])[0].get("summary") if c.get("record") else None
            # Lineup (starters)
            for j, lineup in enumerate(c.get("lineUp", {}).get("entries", [])):
                athlete = lineup.get("athlete", {})
                features[f"{side}_starter_{j+1}_name"] = athlete.get("displayName")
                features[f"{side}_starter_{j+1}_id"] = athlete.get("id")

    # ── 2. PICKCENTER: Pre-game odds ──
    pickcenter = data.get("pickcenter", [])
    if pickcenter:
        for pc in pickcenter:
            provider = pc.get("provider", {}).get("name", "unknown")
            features[f"odds_{provider}_spread"] = pc.get("details")
            features[f"odds_{provider}_ou"] = pc.get("overUnder")
            features[f"odds_{provider}_home_ml"] = pc.get("homeTeamOdds", {}).get("moneyLine")
            features[f"odds_{provider}_away_ml"] = pc.get("awayTeamOdds", {}).get("moneyLine")
            features[f"odds_{provider}_home_spread_odds"] = pc.get("homeTeamOdds", {}).get("spreadOdds")
            features[f"odds_{provider}_away_spread_odds"] = pc.get("awayTeamOdds", {}).get("spreadOdds")
    
    # ── 3. PREDICTOR: ESPN's BPI/win probability ──
    predictor = data.get("predictor", {})
    if predictor:
        features["espn_home_win_pct"] = predictor.get("homeTeam", {}).get("gameProjection")
        features["espn_away_win_pct"] = predictor.get("awayTeam", {}).get("gameProjection")
    
    # ── 4. WIN PROBABILITY: Pre-game baseline ──
    wp = data.get("winprobability", [])
    if wp:
        features["espn_pregame_wp"] = wp[0].get("homeWinPercentage") if wp else None
    
    # ── 5. BOXSCORE: Player stats ──
    boxscore = data.get("boxscore", {})
    for team_block in boxscore.get("players", []):
        team_id = str(team_block.get("team", {}).get("id", ""))
        # Determine home/away from header
        side = None
        for c in comps[0].get("competitors", []) if comps else []:
            if str(c.get("team", {}).get("id")) == team_id:
                side = "home" if c.get("homeAway") == "home" else "away"
                break
        if not side:
            continue
        
        for stat_group in team_block.get("statistics", []):
            for j, athlete in enumerate(stat_group.get("athletes", [])):
                player = athlete.get("athlete", {})
                stats = athlete.get("stats", [])
                labels = stat_group.get("labels", [])
                starter = athlete.get("starter", False)
                
                if j < 15:  # Cap at 15 players per team
                    features[f"{side}_player_{j+1}_name"] = player.get("displayName")
                    features[f"{side}_player_{j+1}_id"] = player.get("id")
                    features[f"{side}_player_{j+1}_starter"] = starter
                    features[f"{side}_player_{j+1}_status"] = player.get("status", {}).get("type")
                    # Map stats to labels
                    for label, val in zip(labels, stats):
                        features[f"{side}_player_{j+1}_{label}"] = val
    
    # ── 6. TEAM STATS from boxscore ──
    for team_block in boxscore.get("teams", []):
        team_id = str(team_block.get("team", {}).get("id", ""))
        side = None
        for c in comps[0].get("competitors", []) if comps else []:
            if str(c.get("team", {}).get("id")) == team_id:
                side = "home" if c.get("homeAway") == "home" else "away"
                break
        if not side:
            continue
        for stat_group in team_block.get("statistics", []):
            for stat in stat_group.get("stats", []):
                features[f"{side}_team_{stat.get('label', stat.get('name', 'unknown'))}"] = stat.get("displayValue")

    # ── 7. INJURIES ──
    injuries = data.get("injuries", [])
    for inj_block in injuries:
        team_id = str(inj_block.get("team", {}).get("id", ""))
        side = None
        for c in comps[0].get("competitors", []) if comps else []:
            if str(c.get("team", {}).get("id")) == team_id:
                side = "home" if c.get("homeAway") == "home" else "away"
                break
        if not side:
            continue
        for k, inj in enumerate(inj_block.get("injuries", [])):
            features[f"{side}_injury_{k+1}_player"] = inj.get("athlete", {}).get("displayName")
            features[f"{side}_injury_{k+1}_status"] = inj.get("status")
            features[f"{side}_injury_{k+1}_detail"] = inj.get("details", {}).get("detail")

    # ── 8. OFFICIALS ──
    officials = data.get("officials", [])
    if not officials:
        # Try from header → competitions[0] → officials
        officials = comps[0].get("officials", []) if comps else []
    for k, off in enumerate(officials):
        features[f"official_{k+1}_name"] = off.get("displayName") or off.get("athlete", {}).get("displayName")
        features[f"official_{k+1}_id"] = off.get("id") or off.get("athlete", {}).get("id")

    # ── 9. GAME INFO ──
    game_info = data.get("gameInfo", {})
    if game_info:
        venue = game_info.get("venue", {})
        features["venue_name"] = venue.get("fullName")
        features["venue_city"] = venue.get("address", {}).get("city")
        features["venue_state"] = venue.get("address", {}).get("state")
        features["venue_capacity"] = venue.get("capacity")
        features["attendance"] = game_info.get("attendance")
        features["neutral_site"] = game_info.get("neutralSite", False)

    # ── 10. AGAINST THE SPREAD ──
    ats = data.get("againstTheSpread", [])
    for entry in ats:
        team_id = str(entry.get("team", {}).get("id", ""))
        side = None
        for c in comps[0].get("competitors", []) if comps else []:
            if str(c.get("team", {}).get("id")) == team_id:
                side = "home" if c.get("homeAway") == "home" else "away"
                break
        if side:
            features[f"{side}_ats_record"] = entry.get("records", [{}])[0].get("summary") if entry.get("records") else None

    return features


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  ESPN NBA Summary API Explorer")
    print("=" * 70)
    
    # Use provided game ID or default to a known completed game
    game_id = sys.argv[1] if len(sys.argv) > 1 else "401810849"
    
    # ── 1. Fetch completed game ──
    print(f"\n  Fetching completed game {game_id}...")
    data = fetch_summary(game_id)
    if not data:
        print("  FAILED — check game ID")
        sys.exit(1)
    
    # Save raw JSON for reference
    with open("espn_nba_summary_raw.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Raw JSON saved to espn_nba_summary_raw.json ({os.path.getsize('espn_nba_summary_raw.json') / 1024:.0f} KB)")
    
    # Show top-level keys
    print(f"\n  TOP-LEVEL KEYS: {list(data.keys())}")
    
    # Explore each section
    for section in ["header", "predictor", "pickcenter", "officials",
                    "againstTheSpread", "winprobability", "boxscore",
                    "gameInfo", "injuries", "leaders", "seasonseries",
                    "odds", "news", "article", "videos", "plays", "standings"]:
        if section in data:
            val = data[section]
            if isinstance(val, list):
                print_section(f"{section} (list, {len(val)} items)", val[0] if val else None, depth=2)
            else:
                print_section(section, val, depth=2)
    
    # ── 2. Extract ML features ──
    print(f"\n{'='*70}")
    print(f"  ML FEATURE EXTRACTION")
    print(f"{'='*70}")
    features = extract_ml_features(data)
    
    # Group by category
    categories = {
        "Teams & Scores": [k for k in features if any(x in k for x in ["team_id", "team_abbr", "team_name", "score", "record"])],
        "Starters/Lineup": [k for k in features if "starter" in k],
        "Odds/Pickcenter": [k for k in features if "odds_" in k],
        "ESPN Predictions": [k for k in features if "espn_" in k],
        "Officials": [k for k in features if "official" in k],
        "Injuries": [k for k in features if "injury" in k],
        "Venue/Context": [k for k in features if any(x in k for x in ["venue", "attendance", "neutral"])],
        "ATS Records": [k for k in features if "ats" in k],
        "Player Stats": [k for k in features if "player" in k],
        "Team Box Stats": [k for k in features if "team_" in k and "team_id" not in k and "team_abbr" not in k and "team_name" not in k],
    }
    
    for cat_name, keys in categories.items():
        if not keys:
            continue
        print(f"\n  {cat_name} ({len(keys)} fields):")
        for k in sorted(keys)[:20]:  # Cap display at 20 per category
            v = features[k]
            print(f"    {k}: {v}")
        if len(keys) > 20:
            print(f"    ... and {len(keys) - 20} more")
    
    total_features = len(features)
    non_null = sum(1 for v in features.values() if v is not None)
    print(f"\n  TOTAL: {total_features} fields extracted, {non_null} non-null")
    
    # ── 3. Try to find an upcoming game ──
    print(f"\n{'='*70}")
    print(f"  PRE-GAME DATA (upcoming game)")
    print(f"{'='*70}")
    today_games = fetch_today_games()
    preview_game = next((g for g in today_games if g["status"] == "Preview"), None)
    if preview_game:
        print(f"  Found upcoming game: {preview_game['away']} @ {preview_game['home']} (ID: {preview_game['id']})")
        pre_data = fetch_summary(preview_game["id"])
        if pre_data:
            pre_features = extract_ml_features(pre_data)
            pre_non_null = sum(1 for v in pre_features.values() if v is not None)
            print(f"\n  PRE-GAME: {pre_non_null} non-null fields (vs {non_null} post-game)")
            
            # What's available pre-game that isn't post-game?
            pre_cats = {
                "Starters": len([k for k in pre_features if "starter" in k and pre_features[k]]),
                "Odds": len([k for k in pre_features if "odds_" in k and pre_features[k]]),
                "ESPN Predictions": len([k for k in pre_features if "espn_" in k and pre_features[k]]),
                "Officials": len([k for k in pre_features if "official" in k and pre_features[k]]),
                "Injuries": len([k for k in pre_features if "injury" in k and pre_features[k]]),
                "Player Stats": len([k for k in pre_features if "player" in k and pre_features[k]]),
            }
            print(f"\n  Pre-game data availability:")
            for cat, count in pre_cats.items():
                marker = "✅" if count > 0 else "❌"
                print(f"    {marker} {cat}: {count} fields")
            
            with open("espn_nba_summary_pregame_raw.json", "w") as f:
                json.dump(pre_data, f, indent=2)
            print(f"\n  Pre-game raw JSON saved to espn_nba_summary_pregame_raw.json")
    else:
        print("  No upcoming games found today — try running earlier in the day")
    
    # ── 4. Feature wishlist ──
    print(f"\n{'='*70}")
    print(f"  DATA WISHLIST — What We Can Extract for NBA ML")
    print(f"{'='*70}")
    print("""
  FROM ESPN /summary (available NOW):
    ✅ Starting lineups (5 player IDs per team)
    ✅ Officials/referees (names + IDs)
    ✅ ESPN pre-game win probability (predictor section)
    ✅ Pickcenter odds (spread, O/U, moneylines)
    ✅ Injury reports (player, status, detail)
    ✅ Venue info (neutral site, capacity, attendance)
    ✅ ATS records
    ✅ Season series (head-to-head)
    ✅ Player box scores (post-game)
    ✅ Team box scores (post-game)
  
  FROM ESPN /statistics (already using):
    ✅ Team season averages (PPG, FG%, etc.)
  
  FROM ESPN /schedule (already using):
    ✅ W/L record, form score, rest days
  
  NOT AVAILABLE FROM ESPN (need external sources):
    ❌ Player advanced stats (PER, BPM, VORP, WS)
    ❌ Lineup +/- data (which 5-man combos work)
    ❌ Historical referee tendencies
    ❌ Player tracking data (speed, distance)
    ❌ Betting line movement / sharp money
    """)
    
    print("  Done! Check espn_nba_summary_raw.json for full data structure.")
