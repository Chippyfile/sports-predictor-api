import requests, time as _time, traceback, json
import numpy as np, pandas as pd
from datetime import datetime
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get

def _espn_cbb_get(path, retries=2):
    """Fetch from ESPN CBB API with retries."""
    url = f"{ESPN_CBB_BASE}/{path}"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                return r.json()
            if r.status_code == 429:
                _time.sleep(2 ** attempt)
                continue
        except Exception as e:
            if attempt == retries:
                print(f"  ESPN CBB fetch failed: {path} — {e}")
    return None


def _fetch_all_d1_teams():
    """Fetch all D1 basketball team IDs from ESPN using pagination."""
    team_ids = set()

    # Method 1: Paginate through groups=50 (all D1)
    # ESPN returns max 50 per page, ~363 D1 teams total = ~8 pages
    print("  Fetching D1 teams (paginated)...")
    page = 1
    while page <= 10:
        data = _espn_cbb_get(f"teams?limit=50&groups=50&page={page}")
        if not data:
            break
        teams_list = []
        if "sports" in data:
            for sport in data["sports"]:
                for league in sport.get("leagues", []):
                    teams_list.extend(league.get("teams", []))
        elif "teams" in data:
            teams_list = data.get("teams", [])

        if not teams_list:
            break

        for team_obj in teams_list:
            t = team_obj.get("team", team_obj)
            tid = t.get("id")
            if tid:
                team_ids.add(str(tid))

        print(f"    Page {page}: +{len(teams_list)} teams (total: {len(team_ids)})")
        if len(teams_list) < 50:
            break  # Last page
        page += 1
        _time.sleep(0.3)

    print(f"  Total from pagination: {len(team_ids)} teams")

    # Method 2: Scoreboard fallback if pagination didn't get enough
    if len(team_ids) < 300:
        print("  Scoreboard fallback...")
        for days_back in range(0, 60, 2):
            dt = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
            try:
                data = _espn_cbb_get(f"scoreboard?dates={dt}&limit=200")
                if data and "events" in data:
                    for ev in data["events"]:
                        for comp in ev.get("competitions", []):
                            for c in comp.get("competitors", []):
                                tid = c.get("team", {}).get("id")
                                if tid:
                                    team_ids.add(str(tid))
            except Exception:
                pass
            if days_back % 20 == 0 and days_back > 0:
                _time.sleep(0.5)
        print(f"  After scoreboard: {len(team_ids)} teams")

    return list(team_ids)


def _fetch_team_data_for_ratings(team_id):
    """Fetch team info, season stats, schedule, and record for ratings."""
    team_data = _espn_cbb_get(f"teams/{team_id}")
    if not team_data:
        return None

    stats_data = _espn_cbb_get(f"teams/{team_id}/statistics")
    sched_data = _espn_cbb_get(f"teams/{team_id}/schedule")
    record_data = _espn_cbb_get(f"teams/{team_id}/record")

    # Handle both {"team": {...}} and direct {...} response formats
    team = team_data.get("team", team_data)
    cats = (stats_data or {}).get("results", {}).get("stats", {}).get("categories", [])
    # Fallback: some ESPN endpoints nest stats differently
    if not cats:
        cats = (stats_data or {}).get("statistics", {}).get("categories", [])
        if not cats:
            # Try splits format
            splits = (stats_data or {}).get("results", {}).get("splits", {}).get("categories", [])
            if splits:
                cats = splits
    def get_stat(name):
        for cat in cats:
            for s in cat.get("stats", []):
                if s.get("name") == name or s.get("displayName") == name:
                    try:
                        return float(s["value"])
                    except (ValueError, TypeError):
                        return None
        return None

    ppg = get_stat("avgPoints") or get_stat("pointsPerGame") or 75.0
    opp_ppg = get_stat("avgPointsAllowed") or get_stat("opponentPointsPerGame") or 75.0
    fga = get_stat("avgFieldGoalsAttempted") or get_stat("fieldGoalsAttempted") or 55.0
    fta = get_stat("avgFreeThrowsAttempted") or get_stat("freeThrowsAttempted") or 18.0
    off_reb = get_stat("avgOffensiveRebounds") or get_stat("offensiveReboundsPerGame") or 10.0
    turnovers = get_stat("avgTurnovers") or 12.0

    # Detect season totals vs per-game
    wins, losses = 0, 0
    # Method 1: record endpoint items
    if record_data and record_data.get("items"):
        for item in record_data["items"]:
            if item.get("type", "") == "total" or item.get("description", "") == "Overall":
                for st in item.get("stats", []):
                    if st.get("name") == "wins":
                        wins = int(st.get("value", 0))
                    if st.get("name") == "losses":
                        losses = int(st.get("value", 0))
    # Method 2: direct wins/losses in record
    if wins == 0 and record_data:
        for item in (record_data.get("items") or []):
            summary = item.get("summary", "")
            if "-" in summary and item.get("type") in ("total", None, ""):
                parts = summary.split("-")
                if len(parts) == 2 and parts[0].strip().isdigit():
                    wins = int(parts[0].strip())
                    losses = int(parts[1].strip())
                    break
    # Method 3: count from game log
    if wins == 0 and sched_data and sched_data.get("events"):
        for ev in sched_data["events"]:
            comp = (ev.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            for c in comp.get("competitors", []):
                if str(c.get("team", {}).get("id")) == str(team_id):
                    if c.get("winner"):
                        wins += 1
                    else:
                        losses += 1

    # Conference: try multiple paths
    conference = ""
    conf_obj = team.get("conference") or team.get("groups", {})
    if isinstance(conf_obj, dict):
        conference = conf_obj.get("name", conf_obj.get("shortName", ""))
    if not conference and team.get("groups", {}).get("parent"):
        conference = team["groups"]["parent"].get("name", "")

    gp = wins + losses or 30
    if fga > 200:
        fga /= gp
        fta /= gp
        off_reb /= gp

    # Tempo (Dean Oliver)
    off_poss = fga - off_reb + turnovers + 0.475 * fta
    tempo = max(55, min(80, off_poss))

    # SOS
    sos = None
    if record_data and record_data.get("items"):
        sos_item = next((i for i in record_data["items"] if i.get("type") == "sos"), None)
        if sos_item:
            sos_stat = next((s for s in sos_item.get("stats", []) if s.get("name") == "opponentWinPercent"), None)
            if sos_stat:
                try:
                    sos = float(sos_stat["value"])
                except (ValueError, TypeError):
                    pass

    # Game log
    game_log = []
    if sched_data and sched_data.get("events"):
        for ev in sched_data["events"]:
            comp = (ev.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            team_comp, opp_comp = None, None
            for c in comp.get("competitors", []):
                if str(c.get("team", {}).get("id")) == str(team_id):
                    team_comp = c
                else:
                    opp_comp = c
            if team_comp and opp_comp:
                try:
                    # ESPN score can be int, str, or dict {"displayValue":"75","value":75.0}
                    def _parse_score(s):
                        if isinstance(s, dict):
                            return int(float(s.get("value", s.get("displayValue", 0))))
                        return int(s)
                    game_log.append({
                        "opp_id": str(opp_comp["team"]["id"]),
                        "my_score": _parse_score(team_comp.get("score", 0)),
                        "opp_score": _parse_score(opp_comp.get("score", 0)),
                        "is_home": team_comp.get("homeAway") == "home",
                        "is_neutral": comp.get("neutralSite", False),
                    })
                except (ValueError, TypeError):
                    continue

    raw_oe = (ppg / tempo * 100) if tempo > 0 else 107.0
    raw_de = (opp_ppg / tempo * 100) if tempo > 0 else 107.0

    return {
        "team_id": str(team_id),
        "name": team.get("displayName", ""),
        "abbr": team.get("abbreviation", ""),
        "conference": conference,
        "ppg": ppg, "opp_ppg": opp_ppg, "tempo": tempo,
        "raw_oe": raw_oe, "raw_de": raw_de,
        "sos": sos, "wins": wins, "losses": losses,
        "game_log": game_log,
    }


def compute_kenpom_ratings(teams_data, max_iterations=8, convergence_threshold=0.01):
    """
    Iterative KenPom-style efficiency computation with 5 enhancements:
    1. Game recency weighting (recent games weighted more)
    2. Opponent rank weighting (top-100 games count more)
    3. Conference strength prior (WCC penalized, B10/SEC boosted)
    4. Margin capping (blowouts capped at ±2 std devs)
    5. Home/away efficiency splits (for spread prediction)
    Plus: post-iteration normalization + dynamic Bayesian shrinkage
    """
    import math

    lookup = {t["team_id"]: t for t in teams_data}
    team_ids = list(lookup.keys())
    n_teams = len(team_ids)

    # ── Conference strength mapping ──
    # Historical conference quality tiers (adj_em shift applied post-convergence)
    # Positive = boost, negative = penalty
    # These are mild priors — they shift EM by 0.5-1.5 pts max
    CONF_PRIOR = {
        # Power conferences (slight boost for depth)
        "SEC": 0.4, "Big Ten": 0.4, "Big 12": 0.3, "ACC": 0.2,
        "Big East": 0.1,
        # Strong mid-majors (no adjustment)
        "American Athletic": 0.0, "Mountain West": 0.0, "Atlantic 10": 0.0,
        "Missouri Valley": 0.0,
        # Weaker conferences (penalty for inflated stats)
        "West Coast": -0.8, "Colonial": -0.5, "Conference USA": -0.5,
        "Sun Belt": -0.5, "Mid-American": -0.6, "Ohio Valley": -0.7,
        "Big South": -0.8, "Southland": -0.8, "MEAC": -1.0, "SWAC": -1.0,
        "Northeast": -0.9, "Patriot": -0.6, "Ivy": -0.4,
        "Horizon": -0.5, "Summit": -0.7, "Big Sky": -0.6,
        "WAC": -0.7, "ASUN": -0.5, "Big West": -0.6,
        "CAA": -0.4, "Southern": -0.6, "America East": -0.7,
        "Atlantic Sun": -0.5, "Metro Atlantic": -0.7, "MAAC": -0.7,
    }

    # ── Compute per-game efficiency std dev for margin capping ──
    all_game_oes = []
    for t in teams_data:
        for g in t["game_log"]:
            if g["opp_id"] in lookup:
                opp = lookup[g["opp_id"]]
                poss = (t["tempo"] + opp["tempo"]) / 2
                if poss > 0:
                    all_game_oes.append(g["my_score"] / poss * 100)
    if all_game_oes:
        oe_global_mean = sum(all_game_oes) / len(all_game_oes)
        oe_global_std = (sum((x - oe_global_mean)**2 for x in all_game_oes) / len(all_game_oes)) ** 0.5
    else:
        oe_global_mean, oe_global_std = 109.7, 15.0
    OE_CAP_LOW = oe_global_mean - 2.0 * oe_global_std
    OE_CAP_HIGH = oe_global_mean + 2.0 * oe_global_std
    print(f"  Margin cap: OE range [{OE_CAP_LOW:.1f}, {OE_CAP_HIGH:.1f}] (mean={oe_global_mean:.1f}, std={oe_global_std:.1f})")

    # ── Recency weights: exponential decay ──
    # Most recent game = 1.0, oldest game ≈ 0.7
    # decay = 0.7^(1/n_games) per game from newest
    RECENCY_FLOOR = 0.70  # oldest game gets 70% weight of newest

    # Iteration 0: raw values
    adj_oe = {tid: lookup[tid]["raw_oe"] for tid in team_ids}
    adj_de = {tid: lookup[tid]["raw_de"] for tid in team_ids}
    adj_ppg = {tid: lookup[tid]["ppg"] for tid in team_ids}
    adj_opp_ppg = {tid: lookup[tid]["opp_ppg"] for tid in team_ids}

    # Home/away tracking (Fix #5)
    home_oes = {tid: [] for tid in team_ids}
    away_oes = {tid: [] for tid in team_ids}
    home_des = {tid: [] for tid in team_ids}
    away_des = {tid: [] for tid in team_ids}

    n_iters = 0
    for iteration in range(max_iterations):
        n_iters = iteration + 1

        all_oe_v = [adj_oe[t] for t in team_ids if adj_oe[t] is not None]
        all_de_v = [adj_de[t] for t in team_ids if adj_de[t] is not None]
        lg_oe = sum(all_oe_v) / len(all_oe_v) if all_oe_v else 109.7
        lg_de = sum(all_de_v) / len(all_de_v) if all_de_v else 109.7
        lg_ppg = sum(lookup[t]["ppg"] for t in team_ids) / len(team_ids)

        # Build current rankings for opponent-rank weighting
        cur_em = {t: adj_oe[t] - adj_de[t] for t in team_ids}
        sorted_by_em = sorted(team_ids, key=lambda t: cur_em[t], reverse=True)
        rank_map = {t: i + 1 for i, t in enumerate(sorted_by_em)}

        new_oe, new_de, new_ppg, new_opp_ppg = {}, {}, {}, {}
        max_delta = 0.0

        # Reset home/away tracking on last iteration
        if iteration == max_iterations - 1 or iteration >= 6:
            home_oes = {tid: [] for tid in team_ids}
            away_oes = {tid: [] for tid in team_ids}
            home_des = {tid: [] for tid in team_ids}
            away_des = {tid: [] for tid in team_ids}

        for tid in team_ids:
            team = lookup[tid]
            if not team["game_log"]:
                new_oe[tid] = adj_oe[tid]
                new_de[tid] = adj_de[tid]
                new_ppg[tid] = adj_ppg[tid]
                new_opp_ppg[tid] = adj_opp_ppg[tid]
                continue

            g_oes, g_des, g_ppgs, g_opps, g_weights = [], [], [], [], []
            n_games = len(team["game_log"])

            for game_idx, game in enumerate(team["game_log"]):
                opp_id = game["opp_id"]
                if opp_id not in lookup:
                    continue

                opp = lookup[opp_id]
                opp_tempo = opp["tempo"]
                opp_de_r = adj_de.get(opp_id, lg_de)
                opp_oe_r = adj_oe.get(opp_id, lg_oe)
                opp_def_ppg = adj_opp_ppg.get(opp_id, opp["opp_ppg"])
                opp_off_ppg = adj_ppg.get(opp_id, opp["ppg"])

                my_score = game["my_score"]
                opp_score = game["opp_score"]
                game_poss = (team["tempo"] + opp_tempo) / 2
                if game_poss <= 0:
                    continue

                # Per-game raw efficiency
                game_oe_raw = my_score / game_poss * 100
                game_de_raw = opp_score / game_poss * 100

                # ── Fix #4: Margin capping ──
                # Cap extreme efficiencies at ±2 std devs
                game_oe_raw = max(OE_CAP_LOW, min(OE_CAP_HIGH, game_oe_raw))
                game_de_raw = max(OE_CAP_LOW, min(OE_CAP_HIGH, game_de_raw))

                # Opponent-adjust
                adj_game_oe = game_oe_raw * (lg_de / opp_de_r) if opp_de_r > 0 else game_oe_raw
                adj_game_de = game_de_raw * (lg_oe / opp_oe_r) if opp_oe_r > 0 else game_de_raw

                # ── Fix #1: Recency weighting ──
                # Games are in chronological order; later index = more recent
                if n_games > 1:
                    recency = RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * (game_idx / (n_games - 1))
                else:
                    recency = 1.0

                # ── Fix #2: Opponent rank weighting ──
                # Top-50 opponents: weight 1.3x, 50-100: 1.1x, 100-200: 1.0x, 200+: 0.85x
                opp_rank = rank_map.get(opp_id, n_teams)
                if opp_rank <= 50:
                    rank_weight = 1.30
                elif opp_rank <= 100:
                    rank_weight = 1.15
                elif opp_rank <= 200:
                    rank_weight = 1.00
                else:
                    rank_weight = 0.85

                # Combined weight
                weight = recency * rank_weight

                g_oes.append(adj_game_oe)
                g_des.append(adj_game_de)
                g_weights.append(weight)

                # PPG-level adjustment
                g_ppgs.append(my_score * (lg_ppg / opp_def_ppg) if opp_def_ppg > 0 else my_score)
                g_opps.append(opp_score * (lg_ppg / opp_off_ppg) if opp_off_ppg > 0 else opp_score)

                # ── Fix #5: Home/away tracking (last iteration only) ──
                if iteration == max_iterations - 1 or iteration >= 6:
                    is_home = game.get("is_home", False)
                    if is_home:
                        home_oes[tid].append(adj_game_oe)
                        home_des[tid].append(adj_game_de)
                    else:
                        away_oes[tid].append(adj_game_oe)
                        away_des[tid].append(adj_game_de)

            if g_oes and sum(g_weights) > 0:
                total_w = sum(g_weights)
                nv_oe = sum(o * w for o, w in zip(g_oes, g_weights)) / total_w
                nv_de = sum(d * w for d, w in zip(g_des, g_weights)) / total_w
                max_delta = max(max_delta, abs(nv_oe - adj_oe[tid]), abs(nv_de - adj_de[tid]))
                new_oe[tid] = nv_oe
                new_de[tid] = nv_de
                # PPG uses simple average (weights less critical for totals)
                new_ppg[tid] = sum(g_ppgs) / len(g_ppgs)
                new_opp_ppg[tid] = sum(g_opps) / len(g_opps)
            else:
                new_oe[tid] = adj_oe[tid]
                new_de[tid] = adj_de[tid]
                new_ppg[tid] = adj_ppg[tid]
                new_opp_ppg[tid] = adj_opp_ppg[tid]

        adj_oe, adj_de = new_oe, new_de
        adj_ppg, adj_opp_ppg = new_ppg, new_opp_ppg

        # ── NORMALIZATION ──
        target_avg = 109.7
        cur_oe_vals = [adj_oe[t] for t in team_ids]
        cur_de_vals = [adj_de[t] for t in team_ids]
        cur_ppg_vals = [adj_ppg[t] for t in team_ids]
        cur_opp_vals = [adj_opp_ppg[t] for t in team_ids]
        oe_mean = sum(cur_oe_vals) / len(cur_oe_vals)
        de_mean = sum(cur_de_vals) / len(cur_de_vals)
        ppg_mean = sum(cur_ppg_vals) / len(cur_ppg_vals)
        opp_mean = sum(cur_opp_vals) / len(cur_opp_vals)
        oe_shift = target_avg - oe_mean
        de_shift = target_avg - de_mean
        ppg_shift = lg_ppg - ppg_mean
        opp_shift = lg_ppg - opp_mean
        for tid in team_ids:
            adj_oe[tid] += oe_shift
            adj_de[tid] += de_shift
            adj_ppg[tid] += ppg_shift
            adj_opp_ppg[tid] += opp_shift

        print(f"  Iteration {n_iters}: max_delta={max_delta:.4f}, oe_mean={oe_mean:.1f}→{target_avg:.1f}")
        if max_delta < convergence_threshold:
            print(f"  Converged after {n_iters} iterations")
            break

    # ── POST-CONVERGENCE: Bayesian shrinkage ──
    avg_games = sum(len(lookup[t]["game_log"]) for t in team_ids) / len(team_ids)
    # Base 0.63 (recalibrated for recency + opponent-rank weighting which
    # compresses distribution ~3 pts vs unweighted iteration)
    SHRINK = 0.63 + 0.35 * avg_games / (avg_games + 8)
    SHRINK = max(0.70, min(0.95, SHRINK))
    # Winsorized league averages for shrinkage targets (5th/95th percentile)
    # Prevents extreme outlier teams from distorting the shrinkage anchor
    def _winsorized_avg(vals):
        arr = np.array(vals)
        if len(arr) < 20:
            return float(arr.mean())
        lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
        return float(np.clip(arr, lo, hi).mean())

    final_oe_vals = [adj_oe[t] for t in team_ids]
    final_de_vals = [adj_de[t] for t in team_ids]
    oe_avg = _winsorized_avg(final_oe_vals)
    de_avg = _winsorized_avg(final_de_vals)
    ppg_vals = [adj_ppg[t] for t in team_ids]
    opp_vals = [adj_opp_ppg[t] for t in team_ids]
    ppg_avg = _winsorized_avg(ppg_vals)
    opp_avg = _winsorized_avg(opp_vals)

    # FIX 2: Non-linear shrinkage for extreme teams (top/bottom 5%).
    # Standard shrinkage compresses all teams equally toward the mean,
    # which systematically underestimates blowout spreads by 5-6 pts.
    # For teams in the top/bottom 5% of EM distribution, apply reduced
    # shrinkage so their extreme ratings are preserved. This only affects
    # ~18 teams on each tail (360 * 0.05) and leaves the middle 90% unchanged.
    pre_shrink_em = {t: adj_oe[t] - adj_de[t] for t in team_ids}
    em_values = sorted(pre_shrink_em.values())
    n_teams_local = len(em_values)
    em_p5 = em_values[max(0, int(n_teams_local * 0.05))]     # bottom 5% threshold
    em_p95 = em_values[min(n_teams_local - 1, int(n_teams_local * 0.95))]  # top 5% threshold
    extreme_count = 0

    for tid in team_ids:
        team_em = pre_shrink_em[tid]
        # Determine per-team shrinkage factor
        if team_em >= em_p95 or team_em <= em_p5:
            # Extreme teams: blend toward full SHRINK but retain more signal.
            # effective_shrink = SHRINK + (1 - SHRINK) * 0.6 → keeps 60% of
            # the gap that would otherwise be shrunk away.
            # Example: SHRINK=0.92, extreme team gets 0.92 + 0.08*0.6 = 0.968
            effective_shrink = SHRINK + (1 - SHRINK) * 0.6
            extreme_count += 1
        else:
            effective_shrink = SHRINK
        adj_oe[tid] = oe_avg + (adj_oe[tid] - oe_avg) * effective_shrink
        adj_de[tid] = de_avg + (adj_de[tid] - de_avg) * effective_shrink
        adj_ppg[tid] = ppg_avg + (adj_ppg[tid] - ppg_avg) * effective_shrink
        adj_opp_ppg[tid] = opp_avg + (adj_opp_ppg[tid] - opp_avg) * effective_shrink
    print(f"  Bayesian shrinkage applied (base={SHRINK:.4f}, avg_games={avg_games:.1f}, extreme_teams={extreme_count})")

    # ── Fix #3: Conference strength prior ──
    # Apply AFTER shrinkage so it doesn't get compressed.
    # Mild shift: only affects EM by adjusting OE up and DE down (or vice versa)
    conf_applied = 0
    for tid in team_ids:
        conf = lookup[tid].get("conference", "")
        prior = CONF_PRIOR.get(conf, 0.0)
        if prior != 0.0:
            adj_oe[tid] += prior / 2   # half to offense boost
            adj_de[tid] -= prior / 2   # half to defense boost (lower = better)
            adj_ppg[tid] += prior / 4
            adj_opp_ppg[tid] -= prior / 4
            conf_applied += 1
    print(f"  Conference priors applied to {conf_applied}/{n_teams} teams")

    # Build results with rankings and home/away splits
    results = []
    for tid in team_ids:
        t = lookup[tid]
        em = adj_oe[tid] - adj_de[tid]

        # Home/away splits (Fix #5)
        h_oe = sum(home_oes[tid]) / len(home_oes[tid]) if home_oes[tid] else None
        a_oe = sum(away_oes[tid]) / len(away_oes[tid]) if away_oes[tid] else None
        h_de = sum(home_des[tid]) / len(home_des[tid]) if home_des[tid] else None
        a_de = sum(away_des[tid]) / len(away_des[tid]) if away_des[tid] else None

        result = {
            "team_id": tid, "team_name": t["name"], "team_abbr": t["abbr"],
            "conference": t["conference"],
            "adj_oe": round(adj_oe[tid], 2), "adj_de": round(adj_de[tid], 2),
            "adj_em": round(em, 2),
            "adj_ppg": round(adj_ppg[tid], 2), "adj_opp_ppg": round(adj_opp_ppg[tid], 2),
            "adj_tempo": round(t["tempo"], 1),
            "raw_oe": round(t["raw_oe"], 2), "raw_de": round(t["raw_de"], 2),
            "raw_ppg": round(t["ppg"], 2), "raw_opp_ppg": round(t["opp_ppg"], 2),
            "sos": round(t["sos"], 4) if t["sos"] is not None else None,
            "wins": t["wins"], "losses": t["losses"],
            "games_used": len(t["game_log"]), "iterations": n_iters,
        }
        # Add home/away splits if available
        if h_oe is not None:
            result["home_oe"] = round(h_oe, 2)
            result["home_de"] = round(h_de, 2) if h_de else None
        if a_oe is not None:
            result["away_oe"] = round(a_oe, 2)
            result["away_de"] = round(a_de, 2) if a_de else None

        results.append(result)

    results.sort(key=lambda x: x["adj_em"], reverse=True)
    for i, r in enumerate(results):
        r["rank_adj_em"] = i + 1

    return results


def run_ncaa_efficiency_computation():
    """Full pipeline: fetch teams → iterate → store in Supabase."""
    start = _time.time()
    print("\n" + "=" * 60)
    print("NCAA EFFICIENCY RATINGS — KenPom Replication")
    print("=" * 60)

    print("\n[1/4] Fetching D1 team IDs...")
    team_ids = _fetch_all_d1_teams()
    print(f"  Found {len(team_ids)} teams")
    if len(team_ids) < 100:
        return {"error": f"Only found {len(team_ids)} teams", "teams_found": len(team_ids)}

    print(f"\n[2/4] Fetching team data ({len(team_ids)} teams)...")
    teams_data = []
    failed = 0
    for i, tid in enumerate(team_ids):
        if i > 0 and i % 50 == 0:
            print(f"  ... {i}/{len(team_ids)} teams ({failed} failed)")
            _time.sleep(1)
        if i > 0 and i % 10 == 0:
            _time.sleep(0.2)

        data = _fetch_team_data_for_ratings(tid)
        if data and data["game_log"]:
            teams_data.append(data)
        else:
            failed += 1

    print(f"  Loaded {len(teams_data)} teams ({failed} failed)")
    if len(teams_data) < 100:
        return {"error": f"Only {len(teams_data)} teams with data", "teams_loaded": len(teams_data)}

    print(f"\n[3/4] Computing ratings (iterative)...")
    ratings = compute_kenpom_ratings(teams_data)

    print(f"\n[4/4] Storing {len(ratings)} ratings in Supabase...")
    stored = _store_ncaa_ratings(ratings)

    elapsed = _time.time() - start
    top5 = ratings[:5]
    print(f"\nCOMPLETE: {len(ratings)} teams in {elapsed:.1f}s")
    for r in top5:
        print(f"  #{r['rank_adj_em']} {r['team_abbr']:6s} EM={r['adj_em']:+.1f} OE={r['adj_oe']:.1f} DE={r['adj_de']:.1f}")

    return {
        "status": "ok",
        "teams_rated": len(ratings),
        "teams_fetched": len(team_ids),
        "iterations": ratings[0]["iterations"] if ratings else 0,
        "elapsed_sec": round(elapsed, 1),
        "stored_to_supabase": stored,
        "top_10": ratings[:10],
    }


def _store_ncaa_ratings(ratings):
    """Upsert ratings to ncaa_team_ratings in Supabase."""
    if not SUPABASE_KEY:
        print("  ⚠️ No Supabase key — skipping storage")
        return False

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    success = 0
    batch_size = 50
    for i in range(0, len(ratings), batch_size):
        batch = ratings[i:i + batch_size]
        for r in batch:
            r["updated_at"] = datetime.utcnow().isoformat()
        try:
            resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/ncaa_team_ratings",
                headers=headers, json=batch, timeout=15,
            )
            if resp.ok:
                success += len(batch)
            else:
                print(f"  Upsert error (batch {i}): {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"  Upsert exception (batch {i}): {e}")

    print(f"  Stored {success}/{len(ratings)} ratings")
    return success == len(ratings)


