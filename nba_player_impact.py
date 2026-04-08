#!/usr/bin/env python3
"""
NBA Player Impact Pipeline
===========================
1. Fetches BPM, VORP, minutes for all NBA players (2 sources)
2. Stores in nba_player_impact Supabase table  
3. Computes missing_impact features for game predictions

Sources (in priority order):
  A) nbaapi.com free API (BPM + VORP from Basketball Reference data)
  B) nba_api package → NBA.com leaguedashplayerstats (PIE, +/-, usage as proxy)

Run: python nba_player_impact.py [--upload] [--season 2026]
"""

import requests, json, time, os, sys
import pandas as pd
import numpy as np
from datetime import datetime

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

NBA_TEAMS = [
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS",
]

# ═══════════════════════════════════════════════════════════
# SOURCE A: nbaapi.com (free, has BPM + VORP directly)
# ═══════════════════════════════════════════════════════════

def fetch_from_nbaapi(season=2026):
    """Fetch advanced stats from free nbaapi.com endpoint."""
    print(f"\n  [Source A] Trying nbaapi.com for season {season}...")
    all_players = []
    
    # Try both season conventions (2026 = 2025-26 season)
    for yr in [season, season - 1]:
        try:
            page = 1
            while True:
                url = f"https://api.server.nbaapi.com/api/playeradvancedstats?season={yr}&page={page}&pageSize=100&sortBy=vorp&ascending=false"
                r = requests.get(url, headers={"accept": "application/json"}, timeout=15)
                if not r.ok or not r.text.strip():
                    break
                data = r.json()
                rows = data.get("data", [])
                if not rows:
                    break
                all_players.extend(rows)
                total_pages = data.get("pagination", {}).get("pages", 1)
                if page >= total_pages:
                    break
                page += 1
                time.sleep(0.5)
            
            if all_players:
                print(f"    ✅ Got {len(all_players)} players from season={yr}")
                break
        except Exception as e:
            print(f"    ❌ Failed for season={yr}: {e}")
    
    if not all_players:
        print(f"    ❌ nbaapi.com returned no data")
        return None
    
    df = pd.DataFrame(all_players)
    # Map to our schema
    result = pd.DataFrame({
        "player_name": df["playerName"],
        "player_id_bbref": df.get("playerId", ""),
        "team": df["team"],
        "position": df.get("position", ""),
        "games": pd.to_numeric(df.get("games", 0), errors="coerce").fillna(0).astype(int),
        "minutes": pd.to_numeric(df.get("minutesPlayed", 0), errors="coerce").fillna(0),
        "bpm": pd.to_numeric(df.get("box", 0), errors="coerce").fillna(0),        # BPM = "box" field
        "obpm": pd.to_numeric(df.get("offensiveBox", 0), errors="coerce").fillna(0),
        "dbpm": pd.to_numeric(df.get("defensiveBox", 0), errors="coerce").fillna(0),
        "vorp": pd.to_numeric(df.get("vorp", 0), errors="coerce").fillna(0),
        "per": pd.to_numeric(df.get("per", 0), errors="coerce").fillna(0),
        "usage_pct": pd.to_numeric(df.get("usagePercent", 0), errors="coerce").fillna(0),
        "ws": pd.to_numeric(df.get("winShares", 0), errors="coerce").fillna(0),
        "ts_pct": pd.to_numeric(df.get("tsPercent", 0), errors="coerce").fillna(0),
    })
    result["mpg"] = np.where(result["games"] > 0, result["minutes"] / result["games"], 0)
    result["minutes_share"] = result["minutes"] / (result["games"] * 48).clip(lower=1)  # fraction of team minutes
    result["source"] = "nbaapi"
    return result


# ═══════════════════════════════════════════════════════════
# SOURCE B: nba_api package (NBA.com official stats)
# ═══════════════════════════════════════════════════════════

def fetch_from_nba_api(season="2025-26"):
    """Fetch from NBA.com via nba_api package. No BPM/VORP but has PIE, +/-, usage."""
    print(f"\n  [Source B] Trying nba_api package for {season}...")
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        
        # Advanced stats
        adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        time.sleep(1)
        df_adv = adv.get_data_frames()[0]
        
        # Base stats for minutes
        base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="Totals",
        )
        time.sleep(1)
        df_base = base.get_data_frames()[0]
        
        # Merge
        merged = df_base[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]].merge(
            df_adv[["PLAYER_ID", "PIE", "NET_RATING", "USG_PCT", "TS_PCT"]],
            on="PLAYER_ID", how="left"
        )
        
        # Approximate BPM from available stats:
        # NBA.com doesn't provide BPM directly. Use NET_RATING as proxy.
        # NET_RATING = team pts/100 - opp pts/100 while on floor ≈ BPM conceptually
        result = pd.DataFrame({
            "player_name": merged["PLAYER_NAME"],
            "player_id_nba": merged["PLAYER_ID"].astype(str),
            "team": merged["TEAM_ABBREVIATION"],
            "games": merged["GP"].astype(int),
            "minutes": merged["MIN"].astype(float),
            "bpm": merged["NET_RATING"].fillna(0).astype(float),  # proxy
            "obpm": 0.0,  # not available
            "dbpm": 0.0,  # not available
            "vorp": 0.0,  # will compute below
            "per": 0.0,
            "usage_pct": merged["USG_PCT"].fillna(0).astype(float) * 100,
            "ws": 0.0,
            "ts_pct": merged["TS_PCT"].fillna(0).astype(float),
        })
        result["mpg"] = np.where(result["games"] > 0, result["minutes"] / result["games"], 0)
        result["minutes_share"] = result["minutes"] / (result["games"] * 48).clip(lower=1)
        # Compute VORP from BPM proxy: [BPM - (-2.0)] * pct_possessions * team_games/82
        result["vorp"] = (result["bpm"] + 2.0) * result["minutes_share"] * result["games"] / 82
        result["source"] = "nba_api"
        
        print(f"    ✅ Got {len(result)} players")
        return result
        
    except ImportError:
        print(f"    ❌ nba_api not installed. Run: pip install nba_api")
        return None
    except Exception as e:
        print(f"    ❌ nba_api failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# SOURCE C: Basketball Reference scrape (most accurate BPM/VORP)
# ═══════════════════════════════════════════════════════════

def fetch_from_bbref(season=2026):
    """Scrape Basketball Reference advanced stats page."""
    print(f"\n  [Source C] Trying Basketball Reference for {season}...")
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if not r.ok:
            print(f"    ❌ HTTP {r.status_code}")
            return None
        
        dfs = pd.read_html(r.text, match="Advanced")
        if not dfs:
            print(f"    ❌ No tables found")
            return None
        
        df = dfs[0]
        # Clean header rows that repeat
        df = df[df["Player"] != "Player"].copy()
        
        # Team abbreviation mapping for BBRef quirks
        team_map = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
        
        result = pd.DataFrame({
            "player_name": df["Player"].str.replace("*", "", regex=False),
            "player_id_bbref": "",  # would need href parsing
            "team": df["Tm"].map(lambda x: team_map.get(x, x)),
            "position": df.get("Pos", ""),
            "games": pd.to_numeric(df["G"], errors="coerce").fillna(0).astype(int),
            "minutes": pd.to_numeric(df["MP"], errors="coerce").fillna(0),
            "bpm": pd.to_numeric(df["BPM"], errors="coerce").fillna(0),
            "obpm": pd.to_numeric(df["OBPM"], errors="coerce").fillna(0),
            "dbpm": pd.to_numeric(df["DBPM"], errors="coerce").fillna(0),
            "vorp": pd.to_numeric(df["VORP"], errors="coerce").fillna(0),
            "per": pd.to_numeric(df["PER"], errors="coerce").fillna(0),
            "usage_pct": pd.to_numeric(df["USG%"], errors="coerce").fillna(0),
            "ws": pd.to_numeric(df["WS"], errors="coerce").fillna(0),
            "ts_pct": pd.to_numeric(df["TS%"], errors="coerce").fillna(0),
        })
        result["mpg"] = np.where(result["games"] > 0, result["minutes"] / result["games"], 0)
        result["minutes_share"] = result["minutes"] / (result["games"] * 48).clip(lower=1)
        result["source"] = "bbref"
        
        # Remove "TOT" rows (traded players total) — keep team-specific rows
        result = result[result["team"] != "TOT"].copy()
        
        print(f"    ✅ Got {len(result)} player-team rows")
        return result
        
    except Exception as e:
        print(f"    ❌ BBRef failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# COMPUTED IMPACT SCORE
# ═══════════════════════════════════════════════════════════

def compute_impact_score(df):
    """
    Compute a composite impact score per player.
    
    impact = (BPM * minutes_weight) + VORP_per_game_bonus
    
    This gives us a single number that captures both:
    - How good the player is per minute (BPM)
    - How much they play (minutes weighting)
    - Their cumulative value (VORP bonus)
    """
    df = df.copy()
    
    # Minutes-weighted BPM: a player with +8 BPM playing 36 min is more impactful than +8 BPM at 12 min
    df["bpm_weighted"] = df["bpm"] * (df["mpg"] / 36.0).clip(upper=1.0)
    
    # VORP per game played (normalized for games played)
    df["vorp_pg"] = np.where(df["games"] > 0, df["vorp"] / df["games"], 0)
    
    # Composite impact: primarily BPM-weighted, with VORP as a volume bonus
    # Scale: ~-5 (worst) to ~+12 (MVP-level)
    df["impact_score"] = df["bpm_weighted"] + df["vorp_pg"] * 10
    
    # For missing player estimation: points of margin lost when this player sits
    # Based on BPM definition: BPM ≈ points per 100 possessions above average
    # At ~100 possessions per game and player's minutes share:
    df["margin_impact"] = df["bpm"] * df["minutes_share"]
    
    return df


# ═══════════════════════════════════════════════════════════
# UPLOAD TO SUPABASE
# ═══════════════════════════════════════════════════════════

def upload_to_supabase(df, season=2026):
    """Upload player impact data to nba_player_impact table."""
    if not SUPABASE_KEY:
        print("\n  ❌ No SUPABASE_KEY found in environment")
        return False
    
    print(f"\n  Uploading {len(df)} players to nba_player_impact...")
    
    # Clear existing data for this season
    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/nba_player_impact?season=eq.{season}",
        headers=HEADERS, timeout=15
    )
    print(f"    Cleared existing: HTTP {r.status_code}")
    
    # Upload in batches
    records = []
    for _, row in df.iterrows():
        records.append({
            "player_name": str(row["player_name"]),
            "team": str(row["team"]),
            "position": str(row.get("position", "")),
            "season": season,
            "games": int(row["games"]),
            "minutes": round(float(row["minutes"]), 1),
            "mpg": round(float(row["mpg"]), 1),
            "minutes_share": round(float(row["minutes_share"]), 4),
            "bpm": round(float(row["bpm"]), 2),
            "obpm": round(float(row.get("obpm", 0)), 2),
            "dbpm": round(float(row.get("dbpm", 0)), 2),
            "vorp": round(float(row["vorp"]), 2),
            "per": round(float(row.get("per", 0)), 1),
            "usage_pct": round(float(row.get("usage_pct", 0)), 1),
            "ws": round(float(row.get("ws", 0)), 1),
            "ts_pct": round(float(row.get("ts_pct", 0)), 3),
            "impact_score": round(float(row["impact_score"]), 2),
            "margin_impact": round(float(row["margin_impact"]), 3),
            "bpm_weighted": round(float(row["bpm_weighted"]), 3),
            "source": str(row.get("source", "")),
            "updated_at": datetime.now(datetime.timezone.utc).isoformat(),
        })
    
    # Batch insert (100 at a time)
    batch_size = 100
    success = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/nba_player_impact",
            headers=HEADERS, json=batch, timeout=30
        )
        if r.ok:
            success += len(batch)
        else:
            print(f"    ❌ Batch {i//batch_size + 1} failed: {r.status_code} {r.text[:200]}")
        time.sleep(0.3)
    
    print(f"    ✅ Uploaded {success}/{len(records)} players")
    return True


# ═══════════════════════════════════════════════════════════
# COMPUTE MISSING IMPACT FOR A GAME
# ═══════════════════════════════════════════════════════════

def compute_missing_impact(team_abbr, out_players, impact_df):
    """
    Given a team and list of OUT player names, compute total missing impact.
    
    Returns: {
        missing_bpm_weighted: float,  # sum of BPM*minutes_weight for OUT players
        missing_vorp: float,          # sum of VORP for OUT players
        missing_margin: float,        # estimated point swing from missing players
        missing_minutes: float,       # total minutes/game of OUT players
        n_missing: int,
        missing_players: [str],
    }
    """
    team_players = impact_df[impact_df["team"] == team_abbr].copy()
    if team_players.empty:
        return {"missing_bpm_weighted": 0, "missing_vorp": 0, "missing_margin": 0,
                "missing_minutes": 0, "n_missing": 0, "missing_players": []}
    
    # Fuzzy match OUT player names to impact table
    from difflib import get_close_matches
    matched = []
    for name in out_players:
        # Try exact match first
        exact = team_players[team_players["player_name"].str.lower() == name.lower()]
        if len(exact):
            matched.append(exact.iloc[0])
            continue
        # Fuzzy match
        candidates = team_players["player_name"].tolist()
        close = get_close_matches(name, candidates, n=1, cutoff=0.6)
        if close:
            row = team_players[team_players["player_name"] == close[0]].iloc[0]
            matched.append(row)
    
    if not matched:
        return {"missing_bpm_weighted": 0, "missing_vorp": 0, "missing_margin": 0,
                "missing_minutes": 0, "n_missing": 0, "missing_players": out_players}
    
    mdf = pd.DataFrame(matched)
    return {
        "missing_bpm_weighted": round(float(mdf["bpm_weighted"].sum()), 3),
        "missing_vorp": round(float(mdf["vorp"].sum()), 2),
        "missing_margin": round(float(mdf["margin_impact"].sum()), 2),
        "missing_minutes": round(float(mdf["mpg"].sum()), 1),
        "n_missing": len(mdf),
        "missing_players": mdf["player_name"].tolist(),
    }


# ═══════════════════════════════════════════════════════════
# DEMO: Tonight's LAL vs OKC
# ═══════════════════════════════════════════════════════════

def demo_lal_okc(df):
    """Show impact of tonight's Lakers injuries."""
    print(f"\n{'='*70}")
    print(f"  DEMO: LAL vs OKC — Missing Player Impact")
    print(f"{'='*70}")
    
    lal_out = ["LeBron James", "Luka Doncic", "Austin Reaves", "Marcus Smart"]
    okc_out = []  # Jalen Williams questionable but let's see
    
    lal_impact = compute_missing_impact("LAL", lal_out, df)
    okc_impact = compute_missing_impact("OKC", okc_out, df)
    
    print(f"\n  LAL missing: {lal_impact['missing_players']}")
    print(f"    BPM weighted:  {lal_impact['missing_bpm_weighted']:+.2f}")
    print(f"    VORP sum:      {lal_impact['missing_vorp']:.1f}")
    print(f"    Margin impact: {lal_impact['missing_margin']:+.2f} pts")
    print(f"    Minutes lost:  {lal_impact['missing_minutes']:.1f} mpg")
    
    print(f"\n  OKC missing: {okc_impact['missing_players'] or 'None'}")
    print(f"    Margin impact: {okc_impact['missing_margin']:+.2f} pts")
    
    net_swing = lal_impact["missing_margin"] - okc_impact["missing_margin"]
    print(f"\n  NET SWING: {net_swing:+.1f} pts toward OKC")
    print(f"  → If base spread was LAL +10, injury-adjusted spread = LAL +{10 + abs(net_swing):.1f}")
    
    # Show top LAL players by impact
    print(f"\n  LAL roster by impact:")
    lal = df[df["team"] == "LAL"].sort_values("impact_score", ascending=False)
    for _, p in lal.head(12).iterrows():
        status = "❌ OUT" if p["player_name"] in lal_out else "✅"
        print(f"    {status} {p['player_name']:25s} BPM={p['bpm']:+5.1f}  VORP={p['vorp']:5.1f}  "
              f"MPG={p['mpg']:4.1f}  impact={p['impact_score']:+6.2f}  margin={p['margin_impact']:+5.2f}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    upload = "--upload" in sys.argv
    season = 2026
    for i, arg in enumerate(sys.argv):
        if arg == "--season" and i + 1 < len(sys.argv):
            season = int(sys.argv[i + 1])
    
    print("=" * 70)
    print(f"  NBA PLAYER IMPACT PIPELINE — Season {season}")
    print(f"  Fetching BPM, VORP, minutes for all players")
    print("=" * 70)
    
    # Try sources in order
    df = fetch_from_nbaapi(season)
    if df is None:
        df = fetch_from_bbref(season)
    if df is None:
        df = fetch_from_nba_api(f"{season-1}-{str(season)[2:]}")
    if df is None:
        print("\n  ❌ All sources failed. Check network/dependencies.")
        sys.exit(1)
    
    # Deduplicate: traded players appear multiple times (once per team).
    # Keep the row with most games per (player_name, team) combo.
    # Also remove "2TM"/"3TM" aggregate rows from BBRef — keep team-specific only.
    before = len(df)
    df = df[~df["team"].str.match(r"^\d+TM$", na=False)].copy()
    df = df.sort_values("games", ascending=False).drop_duplicates(subset=["player_name", "team"], keep="first")
    df = df.reset_index(drop=True)
    if len(df) < before:
        print(f"\n  Deduped: {before} → {len(df)} rows ({before - len(df)} duplicates removed)")

    # Compute impact scores
    df = compute_impact_score(df)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {len(df)} players loaded from {df['source'].iloc[0]}")
    print(f"{'='*70}")
    print(f"\n  Top 20 by impact_score:")
    print(f"  {'Player':25s} {'Team':>4s} {'BPM':>6s} {'VORP':>6s} {'MPG':>5s} {'Impact':>8s} {'Margin':>7s}")
    print(f"  {'─'*25} {'─'*4} {'─'*6} {'─'*6} {'─'*5} {'─'*8} {'─'*7}")
    for _, p in df.nlargest(20, "impact_score").iterrows():
        print(f"  {p['player_name']:25s} {p['team']:>4s} {p['bpm']:>+5.1f} {p['vorp']:>6.1f} "
              f"{p['mpg']:>5.1f} {p['impact_score']:>+7.2f} {p['margin_impact']:>+6.2f}")
    
    # Demo
    demo_lal_okc(df)
    
    # Save locally
    df.to_csv(f"nba_player_impact_{season}.csv", index=False)
    print(f"\n  💾 Saved to nba_player_impact_{season}.csv")
    
    # Upload
    if upload:
        upload_to_supabase(df, season)
    else:
        print(f"\n  Add --upload to push to Supabase")
    
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
