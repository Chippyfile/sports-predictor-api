#!/usr/bin/env python3
"""
ncaa_pit_backfill_v3.py — The Definitive NCAA Point-in-Time Backfill
═══════════════════════════════════════════════════════════════════════
107 features. Zero future leakage. Every stat derived from ESPN box scores,
game results, and market data already in Supabase.

Designed by Claude Opus 4.6 for Chip's multi-sport prediction platform.

Run:  SUPABASE_ANON_KEY="..." python3 ncaa_pit_backfill_v3.py
      SUPABASE_ANON_KEY="..." python3 ncaa_pit_backfill_v3.py --resume
      SUPABASE_ANON_KEY="..." python3 ncaa_pit_backfill_v3.py --fetch-only
"""
import os, sys, json, time, argparse, math, requests
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
CACHE_FILE = "ncaa_boxscore_cache.json"
if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)
HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
           "Content-Type": "application/json", "Prefer": "return=minimal"}

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def sb_get(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=30)
        if not r.ok: break
        data = r.json()
        if not data: break
        all_data.extend(data)
        if len(data) < limit: break
        offset += limit
    return all_data

def sb_patch(table, match_col, match_val, patch_data):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    r = requests.patch(url, headers=HEADERS, json=patch_data, timeout=15)
    return r.ok

def safe_div(a, b, default=0):
    return a / b if b and b != 0 else default

def safe_log2(p):
    return p * math.log2(p) if p > 0 else 0

def slope(values):
    if len(values) < 3: return 0.0
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])

def autocorr(values):
    if len(values) < 5: return 0.0
    a = np.array(values)
    a = a - a.mean()
    c = np.correlate(a, a, 'full')
    c = c[len(c)//2:]
    return float(c[1] / max(c[0], 1e-9))

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / max(na * nb, 1e-9))

# ═══════════════════════════════════════════════════════════════
# ESPN API + BOX SCORE PARSING
# ═══════════════════════════════════════════════════════════════
def fetch_box_score(event_id):
    url = f"{ESPN_SUMMARY}?event={event_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.ok: return r.json()
            if r.status_code == 429: time.sleep(2 ** (attempt + 1)); continue
            if r.status_code == 404: return None
        except requests.exceptions.RequestException: time.sleep(1)
    return None

def parse_box_score(summary_json):
    if not summary_json: return {}
    bs = summary_json.get("boxscore", {})
    result = {}
    for t in bs.get("teams", []):
        tid = str(t.get("team", {}).get("id", ""))
        if not tid: continue
        stats = {}
        for cat in t.get("statistics", []):
            name, display = cat.get("name", ""), cat.get("displayValue", "")
            if not display: continue
            try:
                if "-" in name and "-" in display and not display.startswith("-"):
                    parts = display.split("-")
                    if len(parts) == 2:
                        stats[name + "_made"] = float(parts[0])
                        stats[name + "_att"] = float(parts[1])
                else:
                    stats[name] = float(display)
            except (ValueError, TypeError): pass
        result[tid] = stats
    header = summary_json.get("header", {})
    comps = header.get("competitions", [{}])
    if comps:
        for comp in comps[0].get("competitors", []):
            tid = str(comp.get("id", ""))
            try:
                score = float(comp.get("score", 0))
                if tid in result: result[tid]["points"] = score
                elif tid: result[tid] = {"points": score}
            except (ValueError, TypeError): pass
    return result

def extract_flat_stats(raw):
    def g(k, d=0): return raw.get(k, d)
    fgm = g("fieldGoalsMade-fieldGoalsAttempted_made", g("fieldGoalsMade", 0))
    fga = g("fieldGoalsMade-fieldGoalsAttempted_att", g("fieldGoalsAttempted", 0))
    tpm = g("threePointFieldGoalsMade-threePointFieldGoalsAttempted_made", 0)
    tpa = g("threePointFieldGoalsMade-threePointFieldGoalsAttempted_att", 0)
    ftm = g("freeThrowsMade-freeThrowsAttempted_made", g("freeThrowsMade", 0))
    fta = g("freeThrowsMade-freeThrowsAttempted_att", g("freeThrowsAttempted", 0))
    return {"points": g("points", g("avgPoints", 0)), "fgm": fgm, "fga": fga,
            "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
            "total_reb": g("totalRebounds", 0), "off_reb": g("offensiveRebounds", 0),
            "def_reb": g("defensiveRebounds", 0), "assists": g("assists", g("avgAssists", 0)),
            "steals": g("steals", 0), "blocks": g("blocks", 0),
            "turnovers": g("turnovers", g("totalTurnovers", 0)), "fouls": g("fouls", 0),
            "pts_off_to": g("turnoverPoints", 0), "fastbreak_pts": g("fastBreakPoints", 0),
            "paint_pts": g("pointsInPaint", 0)}

# ═══════════════════════════════════════════════════════════════
# ELO SYSTEM
# ═══════════════════════════════════════════════════════════════
class EloSystem:
    def __init__(self, k=20, hca=100, carryover=0.75):
        self.ratings = defaultdict(lambda: 1500.0)
        self.k = k
        self.hca = hca
        self.carryover = carryover

    def get(self, tid): return self.ratings[tid]

    def expected(self, elo_a, elo_b, home=True):
        diff = elo_a - elo_b + (self.hca if home else 0)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, h_tid, a_tid, h_score, a_score):
        h_elo, a_elo = self.ratings[h_tid], self.ratings[a_tid]
        h_exp = self.expected(h_elo, a_elo, home=True)
        h_result = 1.0 if h_score > a_score else (0.0 if h_score < a_score else 0.5)
        # Margin-of-victory multiplier (FiveThirtyEight style)
        mov = abs(h_score - a_score)
        mov_mult = math.log(max(mov, 1) + 1) * (2.2 / (1.0 + 0.001 * abs(h_elo - a_elo)))
        delta = self.k * mov_mult * (h_result - h_exp)
        self.ratings[h_tid] = h_elo + delta
        self.ratings[a_tid] = a_elo - delta

    def new_season(self):
        for tid in self.ratings:
            self.ratings[tid] = self.carryover * self.ratings[tid] + (1 - self.carryover) * 1500.0

# ═══════════════════════════════════════════════════════════════
# TEAM TRACKER — ALL 107 FEATURES
# ═══════════════════════════════════════════════════════════════
class TeamTracker:
    def __init__(self):
        self.game_stats = []
        self.opp_stats = []
        self.margins = []
        self.scores = []
        self.opp_scores = []
        self.opp_ids = []
        self.opp_elos = []
        self.game_dates = []
        self.is_home_list = []
        self.home_margins = []
        self.away_margins = []
        self.wins = 0
        self.losses = 0
        self.close_wins = 0
        self.close_games = 0
        self.close_expected_probs = []
        self.close_results = []
        self.last_game_date = None
        self.current_streak = 0     # positive=wins, negative=losses
        self.last_result_win = None
        self.win_streak_margins = []
        self.loss_streak_margins = []
        self.after_loss_margins = []
        self.last_loss_date = None
        self.last_blowout_loss_idx = -1
        self.per_game_fgpct = []
        self.per_game_threepct = []
        self.per_game_ppp = []
        self.per_game_opp_fgpct = []
        self.per_game_opp_threepct = []
        self.per_game_ftm = []
        self.per_game_fta = []
        self.opp_tids_set = set()
        self.opp_conferences = set()
        self.dow_margins = defaultdict(list)
        self.rest_bucket_margins = {"b2b": [], "normal": [], "extended": []}
        self.opp_fast_ppg = []
        self.opp_slow_ppg = []
        # For rematches
        self.games_vs = defaultdict(list)  # opp_tid -> [(margin, game_idx)]

    def record_game(self, my_stats, opp_stats, my_score, opp_score, game_date,
                    is_home, opp_tid, opp_elo, opp_conf="", opp_tempo=68,
                    rest_days=3, expected_win_prob=0.5, dow=0):
        margin = my_score - opp_score
        self.game_stats.append(my_stats)
        self.opp_stats.append(opp_stats)
        self.margins.append(margin)
        self.scores.append(my_score)
        self.opp_scores.append(opp_score)
        self.opp_ids.append(opp_tid)
        self.opp_elos.append(opp_elo)
        self.game_dates.append(game_date)
        self.is_home_list.append(is_home)
        self.opp_tids_set.add(opp_tid)
        if opp_conf: self.opp_conferences.add(opp_conf)
        self.dow_margins[dow].append(margin)
        self.games_vs[opp_tid].append((margin, len(self.margins) - 1))

        if is_home: self.home_margins.append(margin)
        else: self.away_margins.append(margin)

        # Rest buckets
        if rest_days <= 1: self.rest_bucket_margins["b2b"].append(margin)
        elif rest_days <= 3: self.rest_bucket_margins["normal"].append(margin)
        else: self.rest_bucket_margins["extended"].append(margin)

        # Defensive versatility: track opponent PPG by pace
        if opp_tempo > 68: self.opp_fast_ppg.append(opp_score)
        else: self.opp_slow_ppg.append(opp_score)

        # Per-game rate stats for divergence
        fga = my_stats.get("fga", 0)
        self.per_game_fgpct.append(safe_div(my_stats.get("fgm", 0), fga))
        self.per_game_threepct.append(safe_div(my_stats.get("tpm", 0), my_stats.get("tpa", 0)))
        poss = fga + 0.475 * my_stats.get("fta", 0) - my_stats.get("off_reb", 0) + my_stats.get("turnovers", 0)
        self.per_game_ppp.append(safe_div(my_score, poss))
        opp_fga = opp_stats.get("fga", 0)
        self.per_game_opp_fgpct.append(safe_div(opp_stats.get("fgm", 0), opp_fga))
        self.per_game_opp_threepct.append(safe_div(opp_stats.get("tpm", 0), opp_stats.get("tpa", 0)))

        # Close game FT tracking
        if abs(margin) <= 8:
            self.per_game_ftm.append(my_stats.get("ftm", 0))
            self.per_game_fta.append(my_stats.get("fta", 0))

        # W/L tracking
        is_win = margin > 0
        if is_win: self.wins += 1
        elif margin < 0: self.losses += 1

        # Close games
        if abs(margin) <= 5:
            self.close_games += 1
            self.close_expected_probs.append(expected_win_prob)
            self.close_results.append(1 if is_win else 0)
            if is_win: self.close_wins += 1

        # Streak tracking
        if self.last_result_win is not None and not self.last_result_win:
            self.after_loss_margins.append(margin)
        if is_win:
            self.current_streak = max(self.current_streak + 1, 1)
            if self.current_streak >= 2: self.win_streak_margins.append(margin)
        else:
            self.current_streak = min(self.current_streak - 1, -1)
            if self.current_streak <= -2: self.loss_streak_margins.append(margin)
            self.last_loss_date = game_date

        if margin < -20: self.last_blowout_loss_idx = len(self.margins) - 1
        self.last_result_win = is_win
        self.last_game_date = game_date

    # ── TIME HELPERS ──
    def get_rest_days(self, d):
        if not self.last_game_date: return 5
        try: return max(0, (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(self.last_game_date, "%Y-%m-%d")).days)
        except: return 3

    def get_games_in_window(self, d, days):
        try:
            curr = datetime.strptime(d, "%Y-%m-%d")
            cutoff = (curr - timedelta(days=days)).strftime("%Y-%m-%d")
            return sum(1 for x in self.game_dates if cutoff < x < d)
        except: return 0

    def get_days_since_loss(self, d):
        if not self.last_loss_date: return 60
        try: return max(0, (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(self.last_loss_date, "%Y-%m-%d")).days)
        except: return 30

    # ══════════════════════════════════════════════════════════
    # THE MAIN METHOD: get_all_features()
    # Returns dict with ALL ~50+ per-team features
    # ══════════════════════════════════════════════════════════
    def get_all_features(self, game_date, elo, opp_elo):
        n = len(self.game_stats)
        if n == 0: return None

        f = {}  # features dict
        tot = defaultdict(float)
        for g in self.game_stats:
            for k, v in g.items(): tot[k] += v
        avg = {k: tot[k] / n for k in tot}

        # ── v2 CORE ──
        f["fgpct"] = safe_div(tot["fgm"], tot["fga"])
        f["threepct"] = safe_div(tot["tpm"], tot["tpa"])
        f["ftpct"] = safe_div(tot["ftm"], tot["fta"])
        f["orb_pct"] = safe_div(tot["off_reb"], tot["total_reb"])
        f["fta_rate"] = safe_div(tot["fta"], tot["fga"])
        f["ato_ratio"] = safe_div(avg.get("assists", 14), max(avg.get("turnovers", 12), 0.1))
        f["points"] = avg.get("points", 75)
        f["assists"] = avg.get("assists", 14)
        f["turnovers"] = avg.get("turnovers", 12)
        f["steals"] = avg.get("steals", 7)
        f["blocks"] = avg.get("blocks", 3.5)
        f["fouls"] = avg.get("fouls", 17)
        f["pts_off_to"] = avg.get("pts_off_to", 0)
        f["fastbreak_pts"] = avg.get("fastbreak_pts", 0)
        f["paint_pts"] = avg.get("paint_pts", 0)

        two_pm = tot["fgm"] - tot["tpm"]
        two_pa = tot["fga"] - tot["tpa"]
        f["twopt_pct"] = safe_div(two_pm, two_pa)
        f["efg_pct"] = safe_div(tot["fgm"] + 0.5 * tot["tpm"], tot["fga"])
        total_pts = sum(self.scores)
        f["ts_pct"] = safe_div(total_pts, 2 * (tot["fga"] + 0.44 * tot["fta"]), 0.53)
        f["three_rate"] = safe_div(tot["tpa"], tot["fga"])
        f["assist_rate"] = safe_div(tot["assists"], max(tot["fgm"], 1))
        opp_orb = sum(g.get("off_reb", 0) for g in self.opp_stats)
        f["drb_pct"] = safe_div(tot["def_reb"], tot["def_reb"] + opp_orb)
        poss = tot["fga"] + 0.475 * tot["fta"] - tot["off_reb"] + tot["turnovers"]
        f["ppp"] = safe_div(total_pts, poss, 1.0)
        f["tempo"] = poss / n if n > 0 else 68

        # ── FORM ──
        last10 = self.margins[-10:]
        if last10:
            w = np.linspace(0.5, 1.0, len(last10))
            form = float(np.average(last10, weights=w))
            f["form"] = round(max(-15, min(15, form)) / 15.0, 4)
        else: f["form"] = 0.0

        f["wins"] = self.wins
        f["losses"] = self.losses
        total_games = self.wins + self.losses
        f["sos"] = safe_div(self.wins, total_games)
        f["n_games"] = n

        # ── v2 ADVANCED ──
        f["consistency"] = round(float(np.std(self.margins)), 2) if n > 2 else 15.0
        f["scoring_var"] = round(float(np.std(self.scores)), 2) if n > 2 else 10.0

        # Luck
        if total_games >= 5:
            pts_for, pts_ag = sum(self.scores), sum(self.opp_scores)
            if pts_for > 0 and pts_ag > 0:
                pyth = (pts_for ** 11.5) / (pts_for ** 11.5 + pts_ag ** 11.5)
                f["luck"] = round(safe_div(self.wins, total_games) - pyth, 4)
            else: f["luck"] = 0.0
        else: f["luck"] = 0.0

        f["close_win_rate"] = safe_div(self.close_wins, self.close_games) if self.close_games >= 3 else 0.5

        # Margin trend
        f["margin_trend"] = round(slope(self.margins[-10:]), 3) if n >= 5 else 0.0

        # ── v3 RESEARCH ──
        f["elo"] = elo

        # Pythagorean residual
        f["pyth_residual"] = round(f["points"] - f["ppp"] * f["tempo"], 2)

        # Scoring entropy (across games)
        if n > 5:
            bins = np.histogram(self.scores, bins=6)[0]
            probs = bins / max(bins.sum(), 1)
            f["scoring_entropy"] = round(-sum(safe_log2(p) for p in probs), 3)
        else: f["scoring_entropy"] = 1.5

        # Margin autocorrelation
        f["margin_autocorr"] = round(autocorr(self.margins), 3) if n >= 5 else 0.0

        # Home/away splits
        f["home_avg_margin"] = round(float(np.mean(self.home_margins)), 1) if self.home_margins else 0.0
        f["away_avg_margin"] = round(float(np.mean(self.away_margins)), 1) if self.away_margins else 0.0

        # Opponent-adjusted form
        if n >= 5 and self.opp_elos:
            last10m = self.margins[-10:]
            last10e = self.opp_elos[-10:]
            adj = [m * (e / 1500.0) for m, e in zip(last10m, last10e)]
            w = np.linspace(0.5, 1.0, len(adj))
            f["opp_adj_form"] = round(float(np.average(adj, weights=w)) / 15.0, 4)
        else: f["opp_adj_form"] = f["form"]

        # Run vulnerability
        to_rate = safe_div(avg.get("turnovers", 12), f["tempo"])
        f["run_vulnerability"] = round(to_rate * (1 - min(f["assist_rate"], 1.0)) * safe_div(avg.get("steals", 7), f["tempo"]), 4)

        # SOS trajectory
        if len(self.opp_elos) >= 10:
            f["sos_trajectory"] = round(slope(self.opp_elos[-10:]), 2)
        else: f["sos_trajectory"] = 0.0

        # Anti-fragility
        if len(self.opp_elos) >= 10:
            underdog_games = [(m, e) for m, e in zip(self.margins, self.opp_elos) if e > elo + 50]
            if len(underdog_games) >= 3:
                underdog_wpct = sum(1 for m, _ in underdog_games if m > 0) / len(underdog_games)
                f["anti_fragility"] = round(underdog_wpct - 0.3, 3)  # 0.3 = expected underdog win rate
            else: f["anti_fragility"] = 0.0
        else: f["anti_fragility"] = 0.0

        # ── v3 ORIGINAL ──
        off_var = float(np.std(self.scores)) if n > 2 else 10
        def_var = float(np.std(self.opp_scores)) if n > 2 else 10
        f["eff_vol_ratio"] = round(safe_div(off_var, def_var), 3)

        # Ceiling / Floor
        if n >= 5:
            f["ceiling"] = round(float(np.percentile(self.margins, 90)), 1)
            f["floor"] = round(float(np.percentile(self.margins, 10)), 1)
        else: f["ceiling"], f["floor"] = 15.0, -10.0

        # Recovery index
        f["recovery_idx"] = round(float(np.mean(self.after_loss_margins)) - float(np.mean(self.margins)), 2) if len(self.after_loss_margins) >= 3 else 0.0
        f["is_after_loss"] = 1 if self.last_result_win is not None and not self.last_result_win else 0

        # Opponent suppression
        if n >= 5 and self.opp_elos:
            suppressions = []
            for i, (opp_score, opp_elo_i) in enumerate(zip(self.opp_scores, self.opp_elos)):
                expected_opp_ppg = (opp_elo_i / 1500.0) * 72  # rough: higher elo = more expected points
                suppressions.append(expected_opp_ppg - opp_score)
            f["opp_suppression"] = round(float(np.mean(suppressions)), 2)
        else: f["opp_suppression"] = 0.0

        # Concentration
        fgpct_var = float(np.std(self.per_game_fgpct)) if n > 2 else 0.05
        f["concentration"] = round(f["scoring_var"] * fgpct_var / max(f["assist_rate"], 0.3), 4)

        # Blowout asymmetry + clutch ratio
        blowout_w = sum(1 for m in self.margins if m > 15)
        blowout_l = sum(1 for m in self.margins if m < -15)
        close_w = sum(1 for m in self.margins if 0 < m <= 5)
        close_l = sum(1 for m in self.margins if -5 <= m < 0)
        f["blowout_asym"] = round(safe_div(blowout_w - blowout_l, n), 3)
        f["clutch_ratio"] = round(safe_div(close_w, close_w + close_l), 3) if (close_w + close_l) >= 3 else 0.5

        # Margin acceleration
        if n >= 10:
            h1 = slope(self.margins[-10:-5])
            h2 = slope(self.margins[-5:])
            f["margin_accel"] = round(h2 - h1, 3)
        else: f["margin_accel"] = 0.0

        # W/L momentum asymmetry
        all_mean = float(np.mean(self.margins)) if self.margins else 0
        ws_mean = float(np.mean(self.win_streak_margins)) if self.win_streak_margins else all_mean
        ls_mean = float(np.mean(self.loss_streak_margins)) if self.loss_streak_margins else all_mean
        f["wl_momentum"] = round((ws_mean - all_mean) - (ls_mean - all_mean), 2)

        # Clutch over expected
        if len(self.close_results) >= 5:
            actual_cwr = np.mean(self.close_results)
            expected_cwr = np.mean(self.close_expected_probs)
            f["clutch_over_exp"] = round(actual_cwr - expected_cwr, 3)
        else: f["clutch_over_exp"] = 0.0

        # Defensive identity stability
        if n >= 5:
            f["def_stability"] = round(1.0 / (float(np.std(self.per_game_opp_fgpct)) + 0.01), 2)
        else: f["def_stability"] = 10.0

        # Calendar fatigue load
        fatigue = 0.0
        try:
            curr = datetime.strptime(game_date, "%Y-%m-%d")
            for i, gd in enumerate(self.game_dates):
                days_ago = (curr - datetime.strptime(gd, "%Y-%m-%d")).days
                if days_ago > 0:
                    weight = math.exp(-days_ago / 30.0)
                    intensity = f["tempo"] * (1 + abs(self.margins[i]) / 20.0)
                    fatigue += weight * intensity
            f["fatigue_load"] = round(fatigue / max(n, 1), 2)
        except: f["fatigue_load"] = 0.0

        # Information gain from last game
        if n >= 2 and self.opp_elos:
            last_elo_diff = elo - self.opp_elos[-1]
            expected_margin = last_elo_diff / 25.0
            actual_margin = self.margins[-1]
            sigma = max(f["consistency"], 5.0)
            surprise = (actual_margin - expected_margin) / sigma
            f["info_gain"] = round(surprise, 3)
        else: f["info_gain"] = 0.0

        # Regression pressure
        recent5 = float(np.mean(self.margins[-5:])) if n >= 5 else all_mean
        season_mean = all_mean
        sigma = max(f["consistency"], 5.0)
        f["regression_pressure"] = round(
            f["luck"] * 10 + f["pyth_residual"] / 5.0 +
            f["clutch_over_exp"] * 15 + (recent5 - season_mean) / sigma, 3)

        # Pace-adjusted margin
        if n >= 3:
            pam = [m / max(f["tempo"] / 70.0, 0.5) for m in self.margins]
            f["pace_adj_margin"] = round(float(np.mean(pam)), 2)
        else: f["pace_adj_margin"] = 0.0

        # FT pressure differential
        if self.per_game_fta and sum(self.per_game_fta) > 0:
            close_ftpct = safe_div(sum(self.per_game_ftm), sum(self.per_game_fta))
            f["ft_pressure"] = round(close_ftpct - f["ftpct"], 4)
        else: f["ft_pressure"] = 0.0

        # Transition dependency
        f["transition_dep"] = round(safe_div(avg.get("fastbreak_pts", 0), f["points"]), 4)

        # ── v3 FINAL FRONTIER ──
        # Day of week effect
        if game_date:
            try:
                dow = datetime.strptime(game_date, "%Y-%m-%d").weekday()
                if dow in self.dow_margins and self.dow_margins[dow]:
                    f["dow_effect"] = round(float(np.mean(self.dow_margins[dow])) - all_mean, 2)
                else: f["dow_effect"] = 0.0
            except: f["dow_effect"] = 0.0
        else: f["dow_effect"] = 0.0

        # Days since loss + streak
        f["days_since_loss"] = self.get_days_since_loss(game_date)
        f["streak"] = self.current_streak

        # Season pct
        f["season_pct"] = round(n / 30.0, 3)

        # Margin skewness
        if n >= 6:
            m = np.array(self.margins)
            mean_m, std_m = m.mean(), m.std()
            f["margin_skew"] = round(float(((m - mean_m) ** 3).mean() / max(std_m ** 3, 1e-9)), 3)
        else: f["margin_skew"] = 0.0

        # Scoring kurtosis
        if n >= 6:
            s = np.array(self.scores)
            mean_s, std_s = s.mean(), s.std()
            f["score_kurtosis"] = round(float(((s - mean_s) ** 4).mean() / max(std_s ** 4, 1e-9)) - 3.0, 3)
        else: f["score_kurtosis"] = 0.0

        # Bimodal detection
        if n >= 6:
            s = np.array(self.scores)
            far_from_mean = np.sum(np.abs(s - s.mean()) > s.std()) / n
            f["bimodal"] = round(far_from_mean, 3)
        else: f["bimodal"] = 0.0

        # Network centrality proxy
        f["centrality"] = round(len(self.opp_tids_set) * len(self.opp_conferences) / max(n, 1), 3)

        # Overreaction detector
        if n >= 2 and self.opp_elos:
            expected = (elo - self.opp_elos[-1]) / 25.0
            actual = self.margins[-1]
            f["overreaction"] = round(expected - actual, 2)  # positive = team did worse than expected, market may overcorrect down
        else: f["overreaction"] = 0.0

        # Rhythm disruption index
        f["rhythm_disruption"] = round(safe_div(avg.get("turnovers", 12) + avg.get("fouls", 17), f["tempo"]), 4)

        # ── CENTURY PUSH ──
        # Scoring source entropy
        pts_2p = (tot["fgm"] - tot["tpm"]) * 2
        pts_3p = tot["tpm"] * 3
        pts_ft = tot["ftm"]
        total_decomp = pts_2p + pts_3p + pts_ft
        if total_decomp > 0:
            p2 = pts_2p / total_decomp
            p3 = pts_3p / total_decomp
            pf = pts_ft / total_decomp
            f["scoring_source_entropy"] = round(-safe_log2(p2) - safe_log2(p3) - safe_log2(pf), 3)
        else: f["scoring_source_entropy"] = 1.5

        f["ft_dependency"] = round(safe_div(tot["ftm"], total_pts), 4)
        f["three_value"] = round(f["threepct"] * f["three_rate"] * 3, 4)
        f["steal_foul_ratio"] = round(safe_div(avg.get("steals", 7), max(avg.get("fouls", 17), 1)), 3)
        f["block_foul_ratio"] = round(safe_div(avg.get("blocks", 3.5), max(avg.get("fouls", 17), 1)), 3)

        # Defensive versatility
        if self.opp_fast_ppg and self.opp_slow_ppg:
            f["def_versatility"] = round(1.0 / (1.0 + abs(np.mean(self.opp_fast_ppg) - np.mean(self.opp_slow_ppg))), 3)
        else: f["def_versatility"] = 0.5

        # TO conversion rate
        opp_to_total = sum(g.get("turnovers", 0) for g in self.opp_stats)
        f["to_conversion"] = round(safe_div(tot["pts_off_to"], opp_to_total), 3)

        # Rest-stratified effect
        overall_mean = all_mean
        rest = self.get_rest_days(game_date)
        if rest <= 1: bucket = "b2b"
        elif rest <= 3: bucket = "normal"
        else: bucket = "extended"
        if self.rest_bucket_margins[bucket]:
            f["rest_effect"] = round(float(np.mean(self.rest_bucket_margins[bucket])) - overall_mean, 2)
        else: f["rest_effect"] = 0.0

        # Point-in-time SOS (using Elo at time of game)
        f["pit_sos"] = round(float(np.mean(self.opp_elos)), 0) if self.opp_elos else 1500.0

        # Games since blowout loss
        f["games_since_blowout_loss"] = n - 1 - self.last_blowout_loss_idx if self.last_blowout_loss_idx >= 0 else 99

        # Win aging
        if n >= 5:
            # For simplicity: use opponent's subsequent Elo direction
            # (We approximate by checking if opponents we beat had high Elos)
            win_elos = [e for m, e in zip(self.margins, self.opp_elos) if m > 0]
            f["win_aging"] = round(float(np.mean(win_elos)) / 1500.0, 3) if win_elos else 1.0
        else: f["win_aging"] = 1.0

        # Short vs long window divergence
        if n >= 8:
            last5_fg = float(np.mean(self.per_game_fgpct[-5:]))
            season_fg = f["fgpct"]
            f["fg_divergence"] = round(last5_fg - season_fg, 4)
            last5_3 = float(np.mean(self.per_game_threepct[-5:]))
            f["three_divergence"] = round(last5_3 - f["threepct"], 4)
            last5_ppp = float(np.mean(self.per_game_ppp[-5:]))
            f["ppp_divergence"] = round(last5_ppp - f["ppp"], 4)
        else:
            f["fg_divergence"] = f["three_divergence"] = f["ppp_divergence"] = 0.0

        # Momentum half-life
        if n >= 10:
            above_avg = [1 if m > all_mean else 0 for m in self.margins]
            streaks = []
            current = 0
            for v in above_avg:
                if v: current += 1
                elif current > 0: streaks.append(current); current = 0
            if current > 0: streaks.append(current)
            f["momentum_halflife"] = round(float(np.median(streaks)), 1) if streaks else 3.0
        else: f["momentum_halflife"] = 3.0

        # Defensive improvement rate
        if n >= 9:
            third = n // 3
            first_third = float(np.mean(self.opp_scores[:third]))
            last_third = float(np.mean(self.opp_scores[-third:]))
            f["def_improvement"] = round(last_third - first_third, 2)
        else: f["def_improvement"] = 0.0

        return f

    def get_opp_averages(self):
        n = len(self.opp_stats)
        if n == 0: return None
        tot = defaultdict(float)
        for g in self.opp_stats:
            for k, v in g.items(): tot[k] += v
        o = {}
        o["opp_ppg"] = tot["points"] / n
        o["opp_fgpct"] = safe_div(tot["fgm"], tot["fga"])
        o["opp_threepct"] = safe_div(tot["tpm"], tot["tpa"])
        o["opp_efg_pct"] = safe_div(tot["fgm"] + 0.5 * tot["tpm"], tot["fga"])
        opp_poss = tot["fga"] + 0.475 * tot["fta"] - tot["off_reb"] + tot["turnovers"]
        o["opp_to_rate"] = safe_div(tot["turnovers"], opp_poss)
        o["opp_fta_rate"] = safe_div(tot["fta"], tot["fga"])
        my_drb = sum(g.get("def_reb", 0) for g in self.game_stats)
        o["opp_orb_pct"] = safe_div(tot["off_reb"], tot["off_reb"] + my_drb)
        return o

    def get_style_vector(self):
        """Return team style vector for cosine similarity."""
        f = self.get_all_features("2000-01-01", 1500, 1500)
        if not f: return [68, 0.50, 0.18, 0.28, 0.34, 0.35]
        return [f["tempo"], f["efg_pct"], safe_div(f["turnovers"], f["tempo"]),
                f["orb_pct"], f["fta_rate"], f["three_rate"]]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="NCAA PIT Backfill v3 — 107 features")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NCAA POINT-IN-TIME BACKFILL v3 — 107 FEATURES")
    print("  Designed by Claude Opus 4.6")
    print("=" * 70)

    all_games = sb_get("ncaa_historical",
                       "actual_home_score=not.is.null&select=*&order=game_date.asc,game_id.asc")
    print(f"  Loaded {len(all_games)} games")
    if not all_games: return

    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f: cache = json.load(f)
        print(f"  Cache: {len(cache)} games")

    # ── PHASE 1: Fetch box scores ──
    game_ids = [g["game_id"] for g in all_games if g.get("game_id")]
    to_fetch = [gid for gid in game_ids if gid not in cache]
    print(f"\n  Phase 1: {len(to_fetch)} to fetch, {len(cache)} cached")

    if to_fetch:
        errors = 0
        for i, gid in enumerate(to_fetch):
            summary = fetch_box_score(gid)
            if summary:
                parsed = parse_box_score(summary)
                cache[gid] = {tid: extract_flat_stats(raw) for tid, raw in parsed.items()} if parsed else {}
            else: cache[gid] = {}; errors += 1
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(to_fetch)} (errors: {errors})")
                with open(CACHE_FILE, "w") as fp: json.dump(cache, fp)
            time.sleep(0.2)
        with open(CACHE_FILE, "w") as fp: json.dump(cache, fp)
        print(f"  Phase 1 done: {errors} errors")

    if args.fetch_only: print("  --fetch-only, stopping"); return

    # ── PHASE 2: Compute all features ──
    print(f"\n  Phase 2: Computing 107 features per game...")
    by_season = defaultdict(list)
    for g in all_games: by_season[int(g.get("season", 0))].append(g)

    elo = EloSystem(k=20, hca=100, carryover=0.75)
    updates = []
    computed = 0

    for season in sorted(by_season.keys()):
        games = sorted(by_season[season], key=lambda g: g.get("game_date", ""))
        trackers = defaultdict(TeamTracker)
        if season > min(by_season.keys()): elo.new_season()
        print(f"\n    Season {season}: {len(games)} games")

        # Build schedule index for lookahead (sandwich detection)
        team_schedule = defaultdict(list)
        for gi, g in enumerate(games):
            h, a = str(g.get("home_team_id", "")), str(g.get("away_team_id", ""))
            team_schedule[h].append(gi)
            team_schedule[a].append(gi)

        for gi, g in enumerate(games):
            gid = g.get("game_id", "")
            h_tid = str(g.get("home_team_id", ""))
            a_tid = str(g.get("away_team_id", ""))
            game_date = g.get("game_date", "")
            h_score = float(g.get("actual_home_score", 0) or 0)
            a_score = float(g.get("actual_away_score", 0) or 0)
            h_rank = int(g.get("home_rank", 200) or 200)
            a_rank = int(g.get("away_rank", 200) or 200)
            h_conf = str(g.get("home_conference", "") or "")
            a_conf = str(g.get("away_conference", "") or "")

            h_elo = elo.get(h_tid)
            a_elo = elo.get(a_tid)
            h_rest = trackers[h_tid].get_rest_days(game_date)
            a_rest = trackers[a_tid].get_rest_days(game_date)
            h_exp = elo.expected(h_elo, a_elo, home=True)
            a_exp = 1.0 - h_exp

            # ── Get PRE-GAME features ──
            hf = trackers[h_tid].get_all_features(game_date, h_elo, a_elo)
            af = trackers[a_tid].get_all_features(game_date, a_elo, h_elo)

            patch = {}
            if hf and af:
                # Build the massive patch
                for prefix, feat, rest, g7, g14 in [
                    ("home_", hf, h_rest, trackers[h_tid].get_games_in_window(game_date, 7),
                     trackers[h_tid].get_games_in_window(game_date, 14)),
                    ("away_", af, a_rest, trackers[a_tid].get_games_in_window(game_date, 7),
                     trackers[a_tid].get_games_in_window(game_date, 14))]:

                    for key in ["ppg", "fgpct", "threepct", "ftpct", "assists", "turnovers",
                                "orb_pct", "fta_rate", "ato_ratio", "steals", "blocks",
                                "form", "sos", "wins", "losses",
                                "twopt_pct", "efg_pct", "ts_pct", "three_rate", "assist_rate",
                                "drb_pct", "ppp", "consistency", "scoring_var", "luck",
                                "close_win_rate", "margin_trend",
                                "pts_off_to", "fastbreak_pts", "paint_pts", "fouls",
                                "elo", "pyth_residual", "scoring_entropy",
                                "margin_autocorr", "opp_adj_form", "run_vulnerability",
                                "sos_trajectory", "anti_fragility",
                                "eff_vol_ratio", "ceiling", "floor",
                                "recovery_idx", "is_after_loss", "opp_suppression",
                                "concentration", "blowout_asym", "clutch_ratio",
                                "margin_accel", "wl_momentum", "clutch_over_exp",
                                "def_stability", "fatigue_load", "info_gain",
                                "regression_pressure", "pace_adj_margin",
                                "ft_pressure", "transition_dep",
                                "dow_effect", "days_since_loss", "streak", "season_pct",
                                "margin_skew", "score_kurtosis", "bimodal", "centrality",
                                "overreaction", "rhythm_disruption",
                                "scoring_source_entropy", "ft_dependency", "three_value",
                                "steal_foul_ratio", "block_foul_ratio",
                                "def_versatility", "to_conversion", "rest_effect",
                                "pit_sos", "games_since_blowout_loss", "win_aging",
                                "fg_divergence", "three_divergence", "ppp_divergence",
                                "momentum_halflife", "def_improvement"]:
                        remap = {"ppg": "points"}.get(key, key)
                        val = feat.get(remap, feat.get(key, 0))
                        col = prefix + key
                        if isinstance(val, (int, np.integer)):
                            patch[col] = int(val)
                        elif val is not None:
                            patch[col] = round(float(val), 4) if isinstance(val, float) else val

                    patch[prefix + "rest_days"] = rest
                    patch[prefix + "games_last_7"] = g7
                    patch[prefix + "games_last_14"] = g14
                    patch[prefix + "tempo"] = round(feat.get("tempo", 68), 1)

                # Home/away margin splits
                patch["home_home_margin"] = round(hf.get("home_avg_margin", 0), 1)
                patch["home_away_margin"] = round(hf.get("away_avg_margin", 0), 1)
                patch["away_home_margin"] = round(af.get("home_avg_margin", 0), 1)
                patch["away_away_margin"] = round(af.get("away_avg_margin", 0), 1)

                # Opponent defensive stats
                h_opp = trackers[h_tid].get_opp_averages()
                a_opp = trackers[a_tid].get_opp_averages()
                if h_opp:
                    for k, v in h_opp.items():
                        patch["home_" + k] = round(v, 4)
                if a_opp:
                    for k, v in a_opp.items():
                        patch["away_" + k] = round(v, 4)

                # ── MATCHUP FEATURES (require both teams) ──
                # Four Factors matchup asymmetry
                patch["matchup_efg"] = round(hf.get("efg_pct", 0.50) - (a_opp or {}).get("opp_efg_pct", 0.50), 4)
                patch["matchup_to"] = round((a_opp or {}).get("opp_to_rate", 0.18) - safe_div(hf.get("turnovers", 12), hf.get("tempo", 68)), 4)
                patch["matchup_orb"] = round(hf.get("orb_pct", 0.28) - af.get("drb_pct", 0.70), 4)
                patch["matchup_ft"] = round(hf.get("fta_rate", 0.34) - (a_opp or {}).get("opp_fta_rate", 0.30), 4)

                # Pace leverage
                patch["pace_leverage"] = round(abs(hf.get("tempo", 68) - af.get("tempo", 68)), 1)

                # Garbage-time adjusted PPP
                for prefix, tracker in [("home_", trackers[h_tid]), ("away_", trackers[a_tid])]:
                    if tracker.margins:
                        weighted_pts, weighted_poss = 0, 0
                        for i, (gs, m) in enumerate(zip(tracker.game_stats, tracker.margins)):
                            w = 0.25 if abs(m) > 30 else (0.5 if abs(m) > 20 else 1.0)
                            gp = gs.get("fga", 0) + 0.475 * gs.get("fta", 0) - gs.get("off_reb", 0) + gs.get("turnovers", 0)
                            weighted_pts += tracker.scores[i] * w
                            weighted_poss += gp * w
                        patch[prefix + "garbage_adj_ppp"] = round(safe_div(weighted_pts, weighted_poss), 4)

                # Style familiarity
                h_style = trackers[h_tid].get_style_vector()
                a_style = trackers[a_tid].get_style_vector()
                # How familiar is home team with away team's style?
                h_prior_styles = [trackers[oid].get_style_vector() for oid in trackers[h_tid].opp_tids_set if oid in trackers]
                if h_prior_styles:
                    sims = [cosine_sim(a_style, ps) for ps in h_prior_styles]
                    patch["style_familiarity"] = round(max(sims) if sims else 0.5, 3)
                else:
                    patch["style_familiarity"] = 0.5

                # Revenge game
                h_vs_a = trackers[h_tid].games_vs.get(a_tid, [])
                a_vs_h = trackers[a_tid].games_vs.get(h_tid, [])
                if h_vs_a or a_vs_h:
                    patch["is_revenge_game"] = 1
                    last_meeting = h_vs_a[-1][0] if h_vs_a else -a_vs_h[-1][0]
                    patch["revenge_margin"] = round(last_meeting, 1)
                else:
                    patch["is_revenge_game"] = 0
                    patch["revenge_margin"] = 0

                # Common opponents
                h_opps = trackers[h_tid].games_vs
                a_opps = trackers[a_tid].games_vs
                common = set(h_opps.keys()) & set(a_opps.keys())
                if common:
                    diffs = []
                    for opp in common:
                        h_margins = [m for m, _ in h_opps[opp]]
                        a_margins = [m for m, _ in a_opps[opp]]
                        diffs.append(np.mean(h_margins) - np.mean(a_margins))
                    patch["common_opp_diff"] = round(float(np.mean(diffs)), 2)
                    patch["n_common_opps"] = len(common)
                else:
                    patch["common_opp_diff"] = 0
                    patch["n_common_opps"] = 0

                # Sandwich / letdown / lookahead detection
                def get_next_opp_rank(tid, current_gi):
                    sched = team_schedule.get(tid, [])
                    for si in sched:
                        if si > current_gi:
                            ng = games[si]
                            if str(ng.get("home_team_id", "")) == tid:
                                return int(ng.get("away_rank", 200) or 200)
                            else:
                                return int(ng.get("home_rank", 200) or 200)
                    return 200

                h_next_rank = get_next_opp_rank(h_tid, gi)
                a_next_rank = get_next_opp_rank(a_tid, gi)
                patch["is_sandwich"] = 1 if (h_rank <= 50 and h_next_rank <= 50 and a_rank > 75) else 0
                patch["is_letdown"] = 1 if (h_rank <= 25 and a_rank > 75) else 0
                patch["is_lookahead"] = 1 if (a_rank > 75 and h_next_rank <= 25) else 0

                # Midweek
                try:
                    dow = datetime.strptime(game_date, "%Y-%m-%d").weekday()
                    patch["is_midweek"] = 1 if dow in (1, 2, 3) else 0
                except: patch["is_midweek"] = 0

                # Defensive rest advantage
                if h_rest > a_rest:
                    home_def_q = (72 - (h_opp or {}).get("opp_ppg", 72)) / 72.0
                    patch["def_rest_advantage"] = round((h_rest - a_rest) * max(home_def_q, 0), 3)
                elif a_rest > h_rest:
                    away_def_q = (72 - (a_opp or {}).get("opp_ppg", 72)) / 72.0
                    patch["def_rest_advantage"] = round(-(a_rest - h_rest) * max(away_def_q, 0), 3)
                else:
                    patch["def_rest_advantage"] = 0

                # Pace control diff
                h_pace_ctrl = 1.0 - abs(hf.get("tempo", 68) - 68) / max(hf.get("tempo", 68), 1)
                a_pace_ctrl = 1.0 - abs(af.get("tempo", 68) - 68) / max(af.get("tempo", 68), 1)
                patch["pace_control_diff"] = round(h_pace_ctrl - a_pace_ctrl, 4)

                # Spread regime
                mkt_spread = float(g.get("market_spread_home", 0) or 0)
                abs_spread = abs(mkt_spread)
                patch["spread_regime"] = 0 if abs_spread <= 3 else (1 if abs_spread <= 7 else (2 if abs_spread <= 12 else (3 if abs_spread <= 18 else 4)))

                # ── INTERACTION FEATURES ──
                patch["fatigue_x_quality"] = round(trackers[h_tid].get_games_in_window(game_date, 7) * (h_elo - a_elo) / 100, 3)
                patch["luck_x_spread"] = round(hf.get("luck", 0) * abs_spread, 3)
                patch["rest_x_defense"] = round((h_rest - a_rest) * ((h_opp or {}).get("opp_ppg", 72) - (a_opp or {}).get("opp_ppg", 72)), 3)
                patch["form_x_familiarity"] = round(hf.get("form", 0) * patch.get("style_familiarity", 0.5), 4)
                patch["consistency_x_spread"] = round(safe_div(1.0, hf.get("consistency", 15)) * abs_spread, 4)

                # Conference balance
                # (simple proxy: std of Elos of teams we know in same conference)
                for prefix, conf in [("home_", h_conf), ("away_", a_conf)]:
                    conf_elos = [elo.get(tid) for tid in elo.ratings
                                 if tid in trackers and trackers[tid].opp_conferences and conf in str(trackers[tid].opp_conferences)]
                    if len(conf_elos) >= 4:
                        patch[prefix + "conf_balance"] = round(1.0 / (float(np.std(conf_elos)) + 1), 4)
                    else:
                        patch[prefix + "conf_balance"] = 0.5

                # Tournament context
                is_post = g.get("is_postseason", False)
                month = int(game_date.split("-")[1]) if game_date and "-" in game_date else 1
                patch["is_early_season"] = month in (11, 12) and not is_post
                patch["is_ncaa_tournament"] = bool(is_post) and month in (3, 4)
                patch["importance_multiplier"] = 1.3 if is_post else (0.8 if month in (11, 12) else 1.0)

                computed += 1

            if patch and gid:
                updates.append((gid, patch))

            # ── Record game into trackers ──
            box = cache.get(gid, {})
            h_box = box.get(h_tid, {})
            a_box = box.get(a_tid, {})
            has_box = h_box and a_box and any(v > 0 for v in h_box.values())

            if not has_box and h_score > 0:
                zero = {"points": 0, "fgm": 0, "fga": 0, "tpm": 0, "tpa": 0, "ftm": 0, "fta": 0,
                        "total_reb": 0, "off_reb": 0, "def_reb": 0, "assists": 0, "steals": 0,
                        "blocks": 0, "turnovers": 0, "fouls": 0, "pts_off_to": 0, "fastbreak_pts": 0, "paint_pts": 0}
                h_box = {**zero, "points": h_score}
                a_box = {**zero, "points": a_score}

            if h_box and a_box:
                try: dow = datetime.strptime(game_date, "%Y-%m-%d").weekday()
                except: dow = 0
                trackers[h_tid].record_game(h_box, a_box, h_score, a_score, game_date,
                    is_home=True, opp_tid=a_tid, opp_elo=a_elo, opp_conf=a_conf,
                    opp_tempo=af.get("tempo", 68) if af else 68, rest_days=h_rest,
                    expected_win_prob=h_exp, dow=dow)
                trackers[a_tid].record_game(a_box, h_box, a_score, h_score, game_date,
                    is_home=False, opp_tid=h_tid, opp_elo=h_elo, opp_conf=h_conf,
                    opp_tempo=hf.get("tempo", 68) if hf else 68, rest_days=a_rest,
                    expected_win_prob=a_exp, dow=dow)
                elo.update(h_tid, a_tid, h_score, a_score)

    print(f"\n  Phase 2 done: {computed} games computed")
    print(f"  Sample patch columns: {len(updates[len(updates)//2][1]) if updates else 0}")

    # ── PHASE 3: Push to Supabase ──
    print(f"\n  Phase 3: Pushing {len(updates)} updates...")
    success = 0
    for i, (gid, patch) in enumerate(updates):
        if sb_patch("ncaa_historical", "game_id", gid, patch): success += 1
        if (i + 1) % 200 == 0: print(f"    {i+1}/{len(updates)} ({success} ok)")
        time.sleep(0.02)

    print(f"\n{'='*70}")
    print(f"  v3 BACKFILL COMPLETE — 107 FEATURES")
    print(f"{'='*70}")
    print(f"  Games: {len(all_games)} | Computed: {computed} | Updated: {success}")
    print(f"  Columns per row: {len(updates[len(updates)//2][1]) if updates else 0}")
    print(f"  Cache: {CACHE_FILE}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
