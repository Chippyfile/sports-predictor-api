"""
ESPN NBA Historical Summary Scraper v2 - COMPLETE EXTRACTION
Run: python3 scrape_nba_summaries.py --test 5
"""
import os, sys, time, json, requests, numpy as np, pandas as pd
from datetime import datetime

ESPN_ABBR_MAP = {"GS":"GSW","NY":"NYK","NO":"NOP","SA":"SAS","WSH":"WAS","UTAH":"UTA","UTH":"UTA","PHO":"PHX","BKLYN":"BKN","BK":"BKN"}
def _abbr(a): return ESPN_ABBR_MAP.get(a, a)

CHECKPOINT = "nba_scrape_checkpoint.json"
OUT_SUMMARY = "nba_summary_data.parquet"
OUT_PLAYERS = "nba_player_boxscores.parquet"
OUT_REFS = "nba_referee_log.parquet"
OUT_PBP = "nba_pbp_features.parquet"
DELAY = 0.35
BATCH_SAVE = 200

def _fetch(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.ok: return r.json()
            if r.status_code == 429: time.sleep(2**(attempt+1)); continue
            if r.status_code == 404: return None
        except: time.sleep(1)
    return None

def _si(v, d=0):
    try: return int(v) if v is not None else d
    except: return d

def _sf(v, d=None):
    try: return float(v) if v is not None else d
    except: return d

def _parse_fg(s):
    if not s or not isinstance(s, str) or "-" not in s: return None, None
    try:
        p = s.split("-"); return int(p[0]), int(p[1])
    except: return None, None

def extract_game(data, game_id):
    if not data: return None, [], [], None
    g = {"game_id": str(game_id)}
    players, refs = [], []
    header = data.get("header", {})
    comps = header.get("competitions", [{}])
    comp = comps[0] if comps else {}
    competitors = comp.get("competitors", [])
    home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})
    h_team, a_team = home_c.get("team", {}), away_c.get("team", {})
    g["home_team"] = _abbr(h_team.get("abbreviation", ""))
    g["away_team"] = _abbr(a_team.get("abbreviation", ""))
    g["home_espn_id"] = h_team.get("id", "")
    g["away_espn_id"] = a_team.get("id", "")
    g["game_date"] = comp.get("date", "")[:10]
    g["home_score"] = _si(home_c.get("score"))
    g["away_score"] = _si(away_c.get("score"))
    g["neutral_site"] = comp.get("neutralSite", False)
    g["conference_game"] = comp.get("conferenceCompetition", False)
    # Records
    hr = home_c.get("record", [])
    ar = away_c.get("record", [])
    g["home_record"] = hr[0].get("summary") if hr else None
    g["away_record"] = ar[0].get("summary") if ar else None
    g["home_split_record"] = hr[1].get("summary") if len(hr) > 1 else None
    g["away_split_record"] = ar[1].get("summary") if len(ar) > 1 else None
    # Quarter scores
    for side_c, prefix in [(home_c, "home"), (away_c, "away")]:
        ls = side_c.get("linescores", [])
        for qi, l in enumerate(ls):
            label = f"ot{qi-3}" if qi > 3 else f"q{qi+1}"
            g[f"{prefix}_{label}"] = _si(l.get("value"))
        g[f"{prefix}_n_periods"] = len(ls)
    g["went_to_ot"] = max(g.get("home_n_periods", 4), g.get("away_n_periods", 4)) > 4
    # Season series
    series = comp.get("series", [])
    if series:
        s = series[0]
        g["h2h_summary"] = s.get("summary", "")
        g["h2h_total_games"] = s.get("totalCompetitions", 0)
        h2h_margins = []
        for evt in s.get("events", []):
            ec = evt.get("competitors", [])
            he = next((c for c in ec if c.get("homeAway") == "home"), {})
            ae = next((c for c in ec if c.get("homeAway") == "away"), {})
            hs, as_ = _si(he.get("score")), _si(ae.get("score"))
            if hs and as_:
                hid = str(he.get("team", {}).get("id", ""))
                h2h_margins.append(hs - as_ if hid == str(g["home_espn_id"]) else as_ - hs)
        g["h2h_avg_margin"] = round(np.mean(h2h_margins), 1) if h2h_margins else None
    # Game info
    gi = data.get("gameInfo", {})
    venue = gi.get("venue", {})
    g["venue_name"] = venue.get("fullName")
    g["venue_id"] = venue.get("id")
    g["venue_city"] = venue.get("address", {}).get("city")
    g["venue_state"] = venue.get("address", {}).get("state")
    g["venue_capacity"] = venue.get("capacity")
    g["attendance"] = gi.get("attendance")
    # Officials
    for i, off in enumerate(gi.get("officials", [])[:3]):
        name = off.get("displayName") or off.get("fullName", "")
        pos = off.get("position", {}).get("name", "")
        g[f"ref_{i+1}_name"] = name
        g[f"ref_{i+1}_position"] = pos
        if name:
            refs.append({"game_id": str(game_id), "game_date": g["game_date"],
                         "home_team": g["home_team"], "away_team": g["away_team"],
                         "home_score": g["home_score"], "away_score": g["away_score"],
                         "ref_name": name, "ref_slot": i+1, "ref_position": pos})
    # Pickcenter
    pc_list = data.get("pickcenter", [])
    if pc_list:
        pc = pc_list[0]
        g["dk_provider"] = pc.get("provider", {}).get("name", "")
        g["dk_spread"] = _sf(pc.get("spread"))
        g["dk_ou"] = _sf(pc.get("overUnder"))
        g["dk_home_ml"] = _si(pc.get("homeTeamOdds", {}).get("moneyLine"))
        g["dk_away_ml"] = _si(pc.get("awayTeamOdds", {}).get("moneyLine"))
        g["dk_over_odds"] = _sf(pc.get("overOdds"))
        g["dk_under_odds"] = _sf(pc.get("underOdds"))
        g["dk_home_spread_odds"] = _sf(pc.get("homeTeamOdds", {}).get("spreadOdds"))
        g["dk_away_spread_odds"] = _sf(pc.get("awayTeamOdds", {}).get("spreadOdds"))
        g["dk_home_fav_at_open"] = pc.get("homeTeamOdds", {}).get("favoriteAtOpen")
        ps = pc.get("pointSpread", {})
        if ps:
            g["spread_open"] = _sf(ps.get("home", {}).get("open", {}).get("line"))
            g["spread_close"] = _sf(ps.get("home", {}).get("close", {}).get("line"))
            g["spread_open_odds"] = _sf(ps.get("home", {}).get("open", {}).get("odds"))
            g["spread_close_odds"] = _sf(ps.get("home", {}).get("close", {}).get("odds"))
        tl = pc.get("total", {})
        if tl:
            g["ou_open"] = _sf(tl.get("over", {}).get("open", {}).get("line"))
            g["ou_close"] = _sf(tl.get("over", {}).get("close", {}).get("line"))
            g["ou_open_odds"] = _sf(tl.get("over", {}).get("open", {}).get("odds"))
            g["ou_close_odds"] = _sf(tl.get("over", {}).get("close", {}).get("odds"))
        ml = pc.get("moneyline", {})
        if ml:
            g["home_ml_open"] = _sf(ml.get("home", {}).get("open", {}).get("odds"))
            g["home_ml_close"] = _sf(ml.get("home", {}).get("close", {}).get("odds"))
            g["away_ml_open"] = _sf(ml.get("away", {}).get("open", {}).get("odds"))
            g["away_ml_close"] = _sf(ml.get("away", {}).get("close", {}).get("odds"))
        if g.get("spread_open") is not None and g.get("spread_close") is not None:
            g["spread_movement"] = round(g["spread_close"] - g["spread_open"], 1)
        if g.get("ou_open") is not None and g.get("ou_close") is not None:
            g["ou_movement"] = round(g["ou_close"] - g["ou_open"], 1)
        if g.get("home_ml_open") is not None and g.get("home_ml_close") is not None:
            g["ml_movement"] = round(g["home_ml_close"] - g["home_ml_open"], 1)
        if g.get("dk_home_fav_at_open") is not None and g.get("spread_close") is not None:
            g["line_reversal"] = int(g["dk_home_fav_at_open"] != (g["spread_close"] < 0))
    # ESPN predictions
    pred = data.get("predictor", {})
    if pred:
        g["espn_home_win_pct"] = _sf(pred.get("homeTeam", {}).get("gameProjection"))
        g["espn_away_win_pct"] = _sf(pred.get("awayTeam", {}).get("gameProjection"))
    wp = data.get("winprobability", [])
    if wp:
        g["espn_pregame_wp"] = _sf(wp[0].get("homeWinPercentage"))
    # Injuries
    hid, aid = str(g["home_espn_id"]), str(g["away_espn_id"])
    for inj_block in data.get("injuries", []):
        tid = str(inj_block.get("team", {}).get("id", ""))
        side = "home" if tid == hid else "away" if tid == aid else None
        if not side: continue
        out_l, dtd_l, out_pos = [], [], []
        for inj in inj_block.get("injuries", []):
            st = inj.get("status", "")
            nm = inj.get("athlete", {}).get("displayName", "")
            ps = inj.get("athlete", {}).get("position", {}).get("abbreviation", "")
            if st in ("Out", "out", "Suspended"): out_l.append(nm); out_pos.append(ps) if ps else None
            elif st in ("Day-To-Day", "day-to-day", "Doubtful", "Questionable"): dtd_l.append(nm)
        g[f"{side}_players_out"] = len(out_l)
        g[f"{side}_players_dtd"] = len(dtd_l)
        g[f"{side}_out_names"] = ",".join(out_l) if out_l else None
        g[f"{side}_dtd_names"] = ",".join(dtd_l) if dtd_l else None
        g[f"{side}_out_positions"] = ",".join(out_pos) if out_pos else None
    # Standings
    for grp in data.get("standings", {}).get("groups", []):
        conf = grp.get("conferenceHeader", "")
        div = grp.get("divisionHeader", "")
        for entry in grp.get("standings", {}).get("entries", []):
            ta = _abbr(entry.get("team", {}).get("abbreviation", ""))
            if ta not in (g["home_team"], g["away_team"]): continue
            side = "home" if ta == g["home_team"] else "away"
            for st in entry.get("stats", []):
                sn = st.get("name", "") or st.get("abbreviation", "")
                sv = st.get("value")
                if sn in ("wins", "W"): g[f"{side}_standings_wins"] = _si(sv)
                elif sn in ("losses", "L"): g[f"{side}_standings_losses"] = _si(sv)
                elif sn in ("gamesBack", "GB"): g[f"{side}_games_back"] = _sf(sv)
                elif sn in ("playoffSeed", "SEED"): g[f"{side}_seed"] = _si(sv)
            g[f"{side}_conference"] = conf
            g[f"{side}_division"] = div
    # Team box stats
    TMAP = {"fieldGoalsMade-fieldGoalsAttempted":"fg","threePointFieldGoalsMade-threePointFieldGoalsAttempted":"3pt",
            "freeThrowsMade-freeThrowsAttempted":"ft","totalRebounds":"total_reb","offensiveRebounds":"oreb",
            "defensiveRebounds":"dreb","assists":"game_ast","steals":"game_stl","blocks":"game_blk",
            "turnovers":"game_to","fouls":"game_pf","technicalFouls":"tech_fouls","flagrantFouls":"flagrant_fouls",
            "fastBreakPoints":"fast_break_pts","pointsInPaint":"paint_pts","pointsOffTurnovers":"pts_off_to",
            "secondChancePoints":"second_chance_pts","benchPoints":"bench_pts","largestLead":"largest_lead"}
    box = data.get("boxscore", {})
    for tb in box.get("teams", []):
        tid = str(tb.get("team", {}).get("id", ""))
        side = "home" if tid == hid else "away" if tid == aid else None
        if not side: continue
        for sg in tb.get("statistics", []):
            for st in sg.get("stats", []):
                rn = st.get("name", "") or st.get("label", "")
                dv = st.get("displayValue", "")
                mp = TMAP.get(rn)
                if mp:
                    if mp in ("fg", "3pt", "ft"):
                        m, a = _parse_fg(dv); g[f"{side}_{mp}m"] = m; g[f"{side}_{mp}a"] = a
                    else: g[f"{side}_{mp}"] = _si(dv) if dv else None
    # Player box scores
    for tb in box.get("players", []):
        tid = str(tb.get("team", {}).get("id", ""))
        side = "home" if tid == hid else "away" if tid == aid else None
        if not side: continue
        ta = g[f"{side}_team"]
        s_ids, s_names, s_pos = [], [], []
        for sg in tb.get("statistics", []):
            labels = sg.get("labels", [])
            for ath in sg.get("athletes", []):
                pl = ath.get("athlete", {})
                starter = ath.get("starter", False)
                stats = ath.get("stats", [])
                pid = str(pl.get("id", ""))
                pn = pl.get("displayName", "")
                pos = pl.get("position", {}).get("abbreviation", "")
                if starter: s_ids.append(pid); s_names.append(pn); s_pos.append(pos)
                sd = {l: v for l, v in zip(labels, stats)}
                players.append({"game_id": str(game_id), "game_date": g["game_date"],
                    "team": ta, "side": side, "player_id": pid, "player_name": pn,
                    "jersey": pl.get("jersey", ""), "position": pos,
                    "starter": starter, "status": pl.get("status", {}).get("type", ""),
                    "MIN": sd.get("MIN", "0"), "PTS": sd.get("PTS", "0"),
                    "REB": sd.get("REB", "0"), "OREB": sd.get("OREB", "0"),
                    "DREB": sd.get("DREB", "0"), "AST": sd.get("AST", "0"),
                    "STL": sd.get("STL", "0"), "BLK": sd.get("BLK", "0"),
                    "TO": sd.get("TO", "0"), "PF": sd.get("PF", "0"),
                    "FG": sd.get("FG", "0-0"), "3PT": sd.get("3PT", "0-0"),
                    "FT": sd.get("FT", "0-0"), "PLUSMINUS": sd.get("+/-", "0")})
        g[f"{side}_starter_ids"] = ",".join(s_ids[:5])
        g[f"{side}_starter_names"] = ",".join(s_names[:5])
        g[f"{side}_starter_positions"] = ",".join(s_pos[:5])
    # PBP features
    pbp = None
    plays = data.get("plays", [])
    if plays:
        pbp = {"game_id": str(game_id), "total_plays": len(plays)}
        lc, h_lead, a_lead, prev_l = 0, 0, 0, None
        h_run, a_run, h_max, a_max, h_8, a_8 = 0, 0, 0, 0, 0, 0
        h_last5, a_last5, h_clutch, a_clutch = 0, 0, 0, 0
        for play in plays:
            hs, as_ = _si(play.get("homeScore")), _si(play.get("awayScore"))
            period = play.get("period", {}).get("number", 1)
            clock = play.get("clock", {}).get("displayValue", "12:00")
            scoring = play.get("scoringPlay", False)
            sv = _si(play.get("scoreValue"))
            ptid = str(play.get("team", {}).get("id", ""))
            leader = "home" if hs > as_ else "away" if as_ > hs else "tie"
            if leader == "home": h_lead += 1
            elif leader == "away": a_lead += 1
            if prev_l and leader != prev_l and leader != "tie" and prev_l != "tie": lc += 1
            prev_l = leader
            if scoring and sv > 0:
                if ptid == hid:
                    h_run += sv; a_run = 0; h_max = max(h_max, h_run)
                    if h_run >= 8: h_8 += 1; h_run = 0
                elif ptid == aid:
                    a_run += sv; h_run = 0; a_max = max(a_max, a_run)
                    if a_run >= 8: a_8 += 1; a_run = 0
                try:
                    mins = int(clock.split(":")[0])
                    if period == 4 and mins < 5:
                        if ptid == hid: h_last5 += sv
                        elif ptid == aid: a_last5 += sv
                        if abs(hs - as_) <= 5:
                            if ptid == hid: h_clutch += sv
                            elif ptid == aid: a_clutch += sv
                except: pass
        tot = max(h_lead + a_lead, 1)
        pbp.update({"lead_changes": lc, "home_lead_pct": round(h_lead/tot, 3),
                     "away_lead_pct": round(a_lead/tot, 3), "home_max_run": h_max,
                     "away_max_run": a_max, "home_runs_8plus": h_8, "away_runs_8plus": a_8,
                     "home_last5_pts": h_last5, "away_last5_pts": a_last5,
                     "home_clutch_pts": h_clutch, "away_clutch_pts": a_clutch})
    return g, players, refs, pbp

def discover_ids(tdf, sf=None):
    pred_ids = set(tdf["game_id"].dropna().astype(str).tolist()) if "game_id" in tdf.columns else set()
    print(f"  IDs from predictions: {len(pred_ids)}")
    hist = tdf[tdf.get("game_id", pd.Series(dtype=str)).isna()].copy() if "game_id" in tdf.columns else tdf.copy()
    dates = sorted(hist["game_date"].dropna().unique()) if len(hist) > 0 else []
    if sf:
        dates = [d for d in dates if f"{sf-1}-10-01" <= d <= f"{sf}-06-30"]
    print(f"  Scanning {len(dates)} dates...")
    disc = {}
    for i, d in enumerate(dates):
        data = _fetch(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d.replace('-','')}&limit=50")
        time.sleep(0.15)
        if data:
            for ev in data.get("events", []):
                comp = ev.get("competitions", [{}])[0]
                if comp.get("status", {}).get("type", {}).get("completed"):
                    disc[str(ev.get("id"))] = d
        if (i+1) % 50 == 0: print(f"    {i+1}/{len(dates)} dates, {len(disc)} IDs")
    all_ids = pred_ids | set(disc.keys())
    print(f"  Total: {len(all_ids)} game IDs")
    return sorted(all_ids)

if __name__ == "__main__":
    print("=" * 70)
    print("  ESPN NBA Summary Scraper v2 — COMPLETE EXTRACTION")
    print("=" * 70)
    test_limit, sf, resume = None, None, "--resume" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--test" and i+1 < len(sys.argv): test_limit = int(sys.argv[i+1])
        if a == "--season" and i+1 < len(sys.argv): sf = int(sys.argv[i+1])
    ckpt = {"done": [], "ts": None}
    if resume and os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: ckpt = json.load(f)
    done = set(ckpt.get("done", []))
    print(f"  Checkpoint: {len(done)} done")
    ex_s = pd.read_parquet(OUT_SUMMARY) if resume and os.path.exists(OUT_SUMMARY) else pd.DataFrame()
    ex_p = pd.read_parquet(OUT_PLAYERS) if resume and os.path.exists(OUT_PLAYERS) else pd.DataFrame()
    ex_r = pd.read_parquet(OUT_REFS) if resume and os.path.exists(OUT_REFS) else pd.DataFrame()
    ex_b = pd.read_parquet(OUT_PBP) if resume and os.path.exists(OUT_PBP) else pd.DataFrame()
    tdf = pd.read_parquet("nba_training_data.parquet")
    all_ids = discover_ids(tdf, sf)
    to_do = [g for g in all_ids if g not in done]
    if test_limit: to_do = to_do[:test_limit]
    print(f"\n  Scraping {len(to_do)} games (~{len(to_do)*DELAY/60:.0f} min)...")
    sr, pr, rr, br = [], [], [], []
    errs, t0 = 0, time.time()
    for i, gid in enumerate(to_do):
        try:
            data = _fetch(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}")
            time.sleep(DELAY)
            if data:
                g, pls, rfs, pbp = extract_game(data, gid)
                if g: sr.append(g)
                pr.extend(pls); rr.extend(rfs)
                if pbp: br.append(pbp)
            else: errs += 1
        except Exception as e:
            errs += 1
            if errs <= 5: print(f"    ERR {gid}: {e}")
        done.add(gid)
        if (i+1) % BATCH_SAVE == 0 or i == len(to_do)-1:
            el = time.time()-t0; rate = (i+1)/el*60 if el > 0 else 0
            rem = (len(to_do)-i-1)/rate if rate > 0 else 0
            print(f"  [{i+1}/{len(to_do)}] {len(sr)} games, {len(pr)} players, {len(rr)} refs, {errs} err | {rate:.0f}/min ~{rem:.0f}m left")
            ckpt["done"] = list(done); ckpt["ts"] = datetime.now().isoformat()
            with open(CHECKPOINT, "w") as f: json.dump(ckpt, f)
            def _sv(rows, ex, path):
                if not rows: return ex
                ndf = pd.DataFrame(rows)
                c = pd.concat([ex, ndf], ignore_index=True) if len(ex) > 0 else ndf
                if "game_id" in c.columns: c = c.drop_duplicates(subset=["game_id"], keep="last") if path != OUT_PLAYERS and path != OUT_REFS else c
                c.to_parquet(path, index=False); return c
            ex_s = _sv(sr, ex_s, OUT_SUMMARY); ex_p = _sv(pr, ex_p, OUT_PLAYERS)
            ex_r = _sv(rr, ex_r, OUT_REFS); ex_b = _sv(br, ex_b, OUT_PBP)
            sr, pr, rr, br = [], [], [], []
    print(f"\n{'='*70}\n  DONE in {(time.time()-t0)/60:.1f} min, {errs} errors")
    for n, p in [(OUT_SUMMARY, OUT_SUMMARY), (OUT_PLAYERS, OUT_PLAYERS), (OUT_REFS, OUT_REFS), (OUT_PBP, OUT_PBP)]:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            print(f"  {n}: {len(df)} rows, {len(df.columns)} cols, {os.path.getsize(p)/1024:.0f} KB")
    if os.path.exists(OUT_SUMMARY):
        df = pd.read_parquet(OUT_SUMMARY)
        print(f"\n  Coverage ({len(df)} games):")
        for nm, col in [("Officials","ref_1_name"),("DK odds","dk_spread"),("Open/close","spread_open"),
                        ("ESPN BPI","espn_pregame_wp"),("Lineups","home_starter_ids"),
                        ("Standings","home_seed"),("Team box","home_fast_break_pts"),
                        ("Attendance","attendance"),("Quarters","home_q1"),("H2H","h2h_summary"),
                        ("PBP","total_plays")]:
            ct = df[col].notna().sum() if col in df.columns else 0
            pct = ct/len(df)*100
            ic = "✅" if pct > 50 else "⚠️" if pct > 10 else "❌"
            print(f"    {ic} {nm:16s} {ct:>5d}/{len(df)} ({pct:.0f}%)")
