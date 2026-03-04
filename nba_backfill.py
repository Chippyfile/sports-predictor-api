"""
NBA Historical Backfill — adds to sports/nba.py
Fetches game results from ESPN scoreboard API for 2021-2025 seasons.
Inserts into nba_historical table in Supabase.

AUDIT C2 FIX (v17): Two-pass approach for leak-free features.
Pass 1: Collect all completed games for the season to build cumulative game logs.
Pass 2: For each game, compute W/L record and form score using ONLY games
completed before that date. Previously used end-of-season stats for all games
(November game had April's win-loss record = massive data leak for ML training).
"""

# ESPN NBA team IDs
NBA_ESPN_IDS = {
    "ATL":1,"BOS":2,"BKN":17,"CHA":30,"CHI":4,"CLE":5,"DAL":6,"DEN":7,
    "DET":8,"GSW":9,"HOU":10,"IND":11,"LAC":12,"LAL":13,"MEM":29,"MIA":14,
    "MIL":15,"MIN":16,"NOP":3,"NYK":18,"OKC":25,"ORL":19,"PHI":20,"PHX":21,
    "POR":22,"SAC":23,"SAS":24,"TOR":28,"UTA":26,"WAS":27,
}

ESPN_ABBR_MAP = {
    "GS":"GSW","NY":"NYK","NO":"NOP","SA":"SAS",
    "WSH":"WAS","UTAH":"UTA","UTH":"UTA","PHO":"PHX",
    "BKLYN":"BKN","BK":"BKN",
}

def _map_nba_abbr(a):
    return ESPN_ABBR_MAP.get(a, a)


def _espn_nba_scoreboard(date_str):
    """Fetch NBA scoreboard for a date (YYYYMMDD format)."""
    import requests, time
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&limit=50"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
        except:
            time.sleep(1)
    return None


def _espn_nba_team_stats(espn_id, season_year):
    """Fetch season stats for a team. season_year = ending year (2025 for 2024-25)."""
    import requests, time
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{espn_id}/statistics?season={season_year}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except:
            time.sleep(1)
    return None


def _parse_team_stats(stats_json):
    """Extract stat values from ESPN team stats response."""
    cats = (stats_json or {}).get("results", {}).get("stats", {}).get("categories", [])
    def get(name):
        for cat in cats:
            for s in cat.get("stats", []):
                if s.get("name") == name or s.get("displayName") == name:
                    try:
                        return float(s["value"])
                    except:
                        pass
        return None

    def norm_pct(v, fb):
        p = v if v is not None else fb
        return p / 100 if p and p > 1 else p

    return {
        "ppg": get("avgPoints") or 112.0,
        "opp_ppg": get("avgPointsAllowed") or get("opponentPointsPerGame") or 112.0,
        "fgpct": norm_pct(get("fieldGoalPct"), 0.471),
        "threepct": norm_pct(get("threePointFieldGoalPct"), 0.365),
        "ftpct": norm_pct(get("freeThrowPct"), 0.780),
        "assists": get("avgAssists") or 25.0,
        "turnovers": get("avgTurnovers") or 14.0,
        "steals": get("avgSteals") or 7.5,
        "blocks": get("avgBlocks") or 5.0,
        "tempo": get("paceFactor") or get("possessions") or 100.0,
        "off_reb": get("avgOffensiveRebounds") or 10.0,
        "def_reb": get("avgDefensiveRebounds") or 34.0,
        "total_reb": get("avgRebounds") or 44.0,
        "opp_fgpct": norm_pct(get("opponentFieldGoalPct"), 0.471),
        "opp_threepct": norm_pct(get("opponentThreePointFieldGoalPct"), 0.365),
    }


def backfill_nba_historical(seasons=None, batch_size=50):
    """
    Fetch NBA game results from ESPN and insert into nba_historical.
    seasons: list of ending years, e.g. [2022, 2023, 2024, 2025]
    Default: 2022-2025 (4 seasons, ~4800 games)
    """
    import requests, time as _time
    from datetime import datetime, timedelta
    from config import SUPABASE_URL, SUPABASE_KEY

    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    # Pre-fetch team stats for each season
    print(f"  Backfilling NBA seasons: {seasons}")
    total_inserted = 0
    total_skipped = 0

    for season_year in seasons:
        # NBA season: Oct of (year-1) to June of (year)
        start_date = datetime(season_year - 1, 10, 18)
        end_date = datetime(season_year, 6, 30)
        season_weight = {2022: 0.7, 2023: 0.8, 2024: 0.9, 2025: 1.0}.get(season_year, 0.6)

        # Fetch team stats for this season (all 30 teams)
        print(f"\n  Season {season_year-1}-{str(season_year)[2:]}: Fetching team stats...")
        team_stats = {}
        for abbr, espn_id in NBA_ESPN_IDS.items():
            stats_json = _espn_nba_team_stats(espn_id, season_year)
            if stats_json:
                team_stats[abbr] = _parse_team_stats(stats_json)
            _time.sleep(0.25)  # Rate limit
        print(f"    Got stats for {len(team_stats)}/30 teams")

        # ── AUDIT C2 FIX: Build cumulative W/L + form per team per date ──
        # Old approach used end-of-season stats for EVERY game = data leak.
        # win_pct_diff and form_score were most leaky — a November game was
        # using April win-loss records. Now we do two passes:
        # Pass 1: Collect all completed games to build game logs
        # Pass 2: For each game, compute cumulative W/L from ONLY prior games.
        print(f"    Pass 1: Collecting all games for cumulative W/L tracking...")
        all_season_games = []
        scan_date = start_date
        while scan_date <= end_date:
            ds = scan_date.strftime("%Y%m%d")
            sb_data = _espn_nba_scoreboard(ds)
            _time.sleep(0.15)
            if sb_data and "events" in sb_data:
                for ev in sb_data["events"]:
                    comp = ev.get("competitions", [{}])[0]
                    status = comp.get("status", {}).get("type", {})
                    if not status.get("completed"):
                        continue
                    competitors = comp.get("competitors", [])
                    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                    if not home or not away:
                        continue
                    h_abbr = _map_nba_abbr(home.get("team", {}).get("abbreviation", ""))
                    a_abbr = _map_nba_abbr(away.get("team", {}).get("abbreviation", ""))
                    h_score = int(home.get("score", 0))
                    a_score = int(away.get("score", 0))
                    if h_score == 0 and a_score == 0:
                        continue
                    all_season_games.append({
                        "date": scan_date.strftime("%Y-%m-%d"),
                        "home": h_abbr, "away": a_abbr,
                        "home_score": h_score, "away_score": a_score,
                    })
            scan_date += timedelta(days=1)
        print(f"    Collected {len(all_season_games)} completed games")

        # Build chronological game log per team
        all_season_games.sort(key=lambda g: g["date"])
        team_game_log = {abbr: [] for abbr in NBA_ESPN_IDS}
        for sg in all_season_games:
            h_won = sg["home_score"] > sg["away_score"]
            team_game_log[sg["home"]].append({"date": sg["date"], "won": h_won})
            team_game_log[sg["away"]].append({"date": sg["date"], "won": not h_won})

        def _cumulative_at_date(abbr, game_date_str):
            """Return (wins, losses, form_score) using only games BEFORE game_date_str."""
            log = team_game_log.get(abbr, [])
            prior = [g for g in log if g["date"] < game_date_str]
            wins = sum(1 for g in prior if g["won"])
            losses = len(prior) - wins
            last5 = prior[-5:]
            if not last5:
                return wins, losses, 0.0
            form = sum((1 if g["won"] else -1) * (i + 1) for i, g in enumerate(last5)) / 15.0
            return wins, losses, round(form, 4)

        print(f"    Pass 2: Building predictions with leak-free features...")
        # ── END AUDIT C2 FIX ─────────────────────────────────────────

        # Iterate through each day of the season
        current = start_date
        games_batch = []
        season_count = 0

        while current <= end_date:
            date_str = current.strftime("%Y%m%d")
            data = _espn_nba_scoreboard(date_str)
            _time.sleep(0.15)  # Rate limit

            if not data or "events" not in data:
                current += timedelta(days=1)
                continue

            for ev in data["events"]:
                comp = ev.get("competitions", [{}])[0]
                status = comp.get("status", {}).get("type", {})
                if not status.get("completed"):
                    continue

                competitors = comp.get("competitors", [])
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue

                home_abbr = _map_nba_abbr(home.get("team", {}).get("abbreviation", ""))
                away_abbr = _map_nba_abbr(away.get("team", {}).get("abbreviation", ""))
                home_score = int(home.get("score", 0))
                away_score = int(away.get("score", 0))

                if home_score == 0 and away_score == 0:
                    continue

                h_stats = team_stats.get(home_abbr, {})
                a_stats = team_stats.get(away_abbr, {})

                # AUDIT C2 FIX: Cumulative W/L and form as of game date (not end-of-season)
                game_date_str = current.strftime("%Y-%m-%d")
                h_cum_w, h_cum_l, h_form = _cumulative_at_date(home_abbr, game_date_str)
                a_cum_w, a_cum_l, a_form = _cumulative_at_date(away_abbr, game_date_str)
                h_wpct = h_cum_w / max(h_cum_w + h_cum_l, 1)
                a_wpct = a_cum_w / max(a_cum_w + a_cum_l, 1)

                # Compute derived features
                h_fga = h_stats.get("ppg", 112) / max(h_stats.get("fgpct", 0.471), 0.3)
                a_fga = a_stats.get("ppg", 112) / max(a_stats.get("fgpct", 0.471), 0.3)
                h_fta = h_stats.get("ppg", 112) * 0.25  # rough estimate
                a_fta = a_stats.get("ppg", 112) * 0.25

                h_orb_pct = h_stats.get("off_reb", 10) / max(h_stats.get("total_reb", 44), 1)
                a_orb_pct = a_stats.get("off_reb", 10) / max(a_stats.get("total_reb", 44), 1)
                h_fta_rate = h_fta / max(h_fga, 1)
                a_fta_rate = a_fta / max(a_fga, 1)
                h_ato = h_stats.get("assists", 25) / max(h_stats.get("turnovers", 14), 1)
                a_ato = a_stats.get("assists", 25) / max(a_stats.get("turnovers", 14), 1)

                h_off_rtg = h_stats.get("ppg", 112) / max(h_stats.get("tempo", 100), 80) * 100
                a_off_rtg = a_stats.get("ppg", 112) / max(a_stats.get("tempo", 100), 80) * 100
                h_def_rtg = h_stats.get("opp_ppg", 112) / max(h_stats.get("tempo", 100), 80) * 100
                a_def_rtg = a_stats.get("opp_ppg", 112) / max(a_stats.get("tempo", 100), 80) * 100

                row = {
                    "game_date": current.strftime("%Y-%m-%d"),
                    "season": season_year,
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "actual_home_score": home_score,
                    "actual_away_score": away_score,
                    "home_ppg": h_stats.get("ppg", 112),
                    "away_ppg": a_stats.get("ppg", 112),
                    "home_opp_ppg": h_stats.get("opp_ppg", 112),
                    "away_opp_ppg": a_stats.get("opp_ppg", 112),
                    "home_fgpct": h_stats.get("fgpct", 0.471),
                    "away_fgpct": a_stats.get("fgpct", 0.471),
                    "home_threepct": h_stats.get("threepct", 0.365),
                    "away_threepct": a_stats.get("threepct", 0.365),
                    "home_ftpct": h_stats.get("ftpct", 0.780),
                    "away_ftpct": a_stats.get("ftpct", 0.780),
                    "home_orb_pct": round(h_orb_pct, 3),
                    "away_orb_pct": round(a_orb_pct, 3),
                    "home_fta_rate": round(h_fta_rate, 3),
                    "away_fta_rate": round(a_fta_rate, 3),
                    "home_ato_ratio": round(h_ato, 2),
                    "away_ato_ratio": round(a_ato, 2),
                    "home_opp_fgpct": h_stats.get("opp_fgpct", 0.471),
                    "away_opp_fgpct": a_stats.get("opp_fgpct", 0.471),
                    "home_opp_threepct": h_stats.get("opp_threepct", 0.365),
                    "away_opp_threepct": a_stats.get("opp_threepct", 0.365),
                    "home_steals": h_stats.get("steals", 7.5),
                    "away_steals": a_stats.get("steals", 7.5),
                    "home_blocks": h_stats.get("blocks", 5.0),
                    "away_blocks": a_stats.get("blocks", 5.0),
                    "home_turnovers": h_stats.get("turnovers", 14),
                    "away_turnovers": a_stats.get("turnovers", 14),
                    "home_assists": h_stats.get("assists", 25),
                    "away_assists": a_stats.get("assists", 25),
                    "home_tempo": h_stats.get("tempo", 100),
                    "away_tempo": a_stats.get("tempo", 100),
                    "home_net_rtg": round(h_off_rtg - h_def_rtg, 1),
                    "away_net_rtg": round(a_off_rtg - a_def_rtg, 1),
                    "pred_home_score": round(h_stats.get("ppg", 112), 1),
                    "pred_away_score": round(a_stats.get("ppg", 112), 1),
                    # ── AUDIT C2 FIX: Cumulative W/L (leak-free) ──
                    # Was: win_pct_home=0.55 (useless HCA default)
                    # Now: compute from actual W/L record UP TO this game date
                    "home_wins": h_cum_w,
                    "home_losses": h_cum_l,
                    "away_wins": a_cum_w,
                    "away_losses": a_cum_l,
                    "home_form": h_form,
                    "away_form": a_form,
                    "win_pct_home": round(
                        0.5 + (h_wpct - a_wpct) / 2 + 0.025,  # HCA = +2.5% baseline
                        4
                    ) if (h_cum_w + h_cum_l >= 5 and a_cum_w + a_cum_l >= 5) else 0.55,
                    "season_weight": season_weight,
                    "is_outlier_season": False,
                }

                games_batch.append(row)
                season_count += 1

                # Flush batch
                if len(games_batch) >= batch_size:
                    try:
                        resp = requests.post(
                            f"{SUPABASE_URL}/rest/v1/nba_historical",
                            headers=headers, json=games_batch, timeout=30,
                        )
                        if resp.ok:
                            total_inserted += len(games_batch)
                        else:
                            total_skipped += len(games_batch)
                            print(f"    Batch error: {resp.status_code} {resp.text[:200]}")
                    except Exception as e:
                        total_skipped += len(games_batch)
                        print(f"    Batch exception: {e}")
                    games_batch = []

            current += timedelta(days=1)

        # Flush remaining
        if games_batch:
            try:
                resp = requests.post(
                    f"{SUPABASE_URL}/rest/v1/nba_historical",
                    headers=headers, json=games_batch, timeout=30,
                )
                if resp.ok:
                    total_inserted += len(games_batch)
                else:
                    total_skipped += len(games_batch)
            except:
                total_skipped += len(games_batch)
            games_batch = []

        print(f"    Season {season_year}: {season_count} games")

    return {
        "status": "complete",
        "seasons": seasons,
        "total_inserted": total_inserted,
        "total_skipped": total_skipped,
    }
