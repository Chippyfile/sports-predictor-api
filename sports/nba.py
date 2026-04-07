import numpy as np, pandas as pd, traceback as _tb, shap
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from db import sb_get, save_model, load_model
from dynamic_constants import compute_nba_league_averages, NBA_DEFAULT_AVERAGES
from ml_utils import HAS_XGB, _time_series_oof, StackedRegressor, StackedClassifier
if HAS_XGB:
    from xgboost import XGBRegressor, XGBClassifier
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

def nba_build_features(df):
    """
    NBA feature builder v20 — expanded to use 30+ raw stat columns from nbaSync.
    Mirrors the NCAA pattern: differential features + context + heuristic signal.

    v20 audit fixes:
      CRIT-1:  spread_vs_market zeroed when has_market=0 (was: = score_diff_pred, circular leak)
      CRIT-1b: market_total zeroed when no real market O/U (was: fell back to model's own ou_total)
      MED-1:   ato_ratio default 1.7 → 1.8 (matches JS/backfill baseline)
      MED-2:   opp_threepct fallback 0.35 → 0.365, threepct 0.36 → 0.365 (matches lg average)
    """
    df = df.copy()

    # ── Core heuristic outputs ──
    df["score_diff_pred"] = df["pred_home_score"].fillna(0) - df["pred_away_score"].fillna(0)
    df["total_pred"]      = df["pred_home_score"].fillna(0) + df["pred_away_score"].fillna(0)
    df["home_fav"]        = (pd.to_numeric(df.get("model_ml_home", 0), errors="coerce").fillna(0) < 0).astype(int)
    df["win_pct_home"]    = pd.to_numeric(df.get("win_pct_home", 0.5), errors="coerce").fillna(0.5)

    # ── NEW-H1 FIX: ou_gap NULL handling for historical rows ──
    _market_ou = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else pd.Series(dtype=float, index=df.index),
        errors="coerce"
    )
    _has_market_ou = _market_ou.notna() & (_market_ou != 0)
    df["ou_gap"] = np.where(_has_market_ou, df["total_pred"] - _market_ou.fillna(228), 0.0)

    # ── Net rating ──
    df["home_net_rtg"] = pd.to_numeric(df.get("home_net_rtg", 0), errors="coerce").fillna(0)
    df["away_net_rtg"] = pd.to_numeric(df.get("away_net_rtg", 0), errors="coerce").fillna(0)
    df["net_rtg_diff"] = df["home_net_rtg"] - df["away_net_rtg"]

    # ── Raw stat differentials (from nbaSync 30+ columns) ──
    # Dynamic league averages (derived from nba_historical, falls back to static)
    _nba_avgs = getattr(nba_build_features, "_league_averages", NBA_DEFAULT_AVERAGES)
    for col_base, default in [
        ("ppg", _nba_avgs.get("ppg", 110)),
        ("opp_ppg", _nba_avgs.get("opp_ppg", 110)),
        ("fgpct", _nba_avgs.get("fgpct", 0.46)),
        ("threepct", _nba_avgs.get("threepct", 0.365)),    # v20 MED-2 FIX: was 0.36
        ("ftpct", _nba_avgs.get("ftpct", 0.77)),
        ("assists", _nba_avgs.get("assists", 25)),
        ("turnovers", _nba_avgs.get("turnovers", 14)),
        ("tempo", _nba_avgs.get("tempo", 100)),
        ("orb_pct", _nba_avgs.get("orb_pct", 0.25)),
        ("fta_rate", _nba_avgs.get("fta_rate", 0.28)),
        ("ato_ratio", _nba_avgs.get("ato_ratio", 1.8)),    # v20 MED-1 FIX: was 1.7
        ("opp_fgpct", _nba_avgs.get("fgpct", 0.46)),
        ("opp_threepct", _nba_avgs.get("threepct", 0.365)), # v20 MED-2 FIX: was 0.35
        ("steals", _nba_avgs.get("steals", 7.5)),
        ("blocks", _nba_avgs.get("blocks", 5.0)),
    ]:
        h_col = f"home_{col_base}"
        a_col = f"away_{col_base}"
        df[h_col] = pd.to_numeric(df.get(h_col, default), errors="coerce").fillna(default)
        df[a_col] = pd.to_numeric(df.get(a_col, default), errors="coerce").fillna(default)
        df[f"{col_base}_diff"] = df[h_col] - df[a_col]

    # ── Win % differential ──
    h_wins = pd.to_numeric(df.get("home_wins", 0), errors="coerce").fillna(0)
    h_losses = pd.to_numeric(df.get("home_losses", 0), errors="coerce").fillna(0)
    a_wins = pd.to_numeric(df.get("away_wins", 0), errors="coerce").fillna(0)
    a_losses = pd.to_numeric(df.get("away_losses", 0), errors="coerce").fillna(0)
    df["win_pct_diff"] = (h_wins / (h_wins + h_losses).clip(1)) - (a_wins / (a_wins + a_losses).clip(1))

    # ── Form differential ──
    df["form_diff"] = (
        pd.to_numeric(df.get("home_form", 0), errors="coerce").fillna(0) -
        pd.to_numeric(df.get("away_form", 0), errors="coerce").fillna(0)
    )

    # ── Tempo average ──
    df["tempo_avg"] = (df["home_tempo"] + df["away_tempo"]) / 2

    # ── Rest & travel ──
    df["rest_diff"] = (
        pd.to_numeric(df.get("home_days_rest", 2), errors="coerce").fillna(2) -
        pd.to_numeric(df.get("away_days_rest", 2), errors="coerce").fillna(2)
    )
    df["away_travel"] = pd.to_numeric(df.get("away_travel_dist", 0), errors="coerce").fillna(0)

    # ── Turnover quality ──
    df["to_margin_diff"] = df["away_turnovers"] - df["home_turnovers"]
    df["steals_to_h"] = df["home_steals"] / df["home_turnovers"].clip(0.5)
    df["steals_to_a"] = df["away_steals"] / df["away_turnovers"].clip(0.5)
    df["steals_to_diff"] = df["steals_to_h"] - df["steals_to_a"]

    # ── Market line features (strongest public predictor) ──
    # market_spread: prefer market_spread_home, fall back to dk_spread (from summary parquet)
    _ms_home = pd.to_numeric(df["market_spread_home"] if "market_spread_home" in df.columns else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    _ms_dk   = pd.to_numeric(df["dk_spread"] if "dk_spread" in df.columns else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    df["market_spread"] = np.where(_ms_home != 0, _ms_home, _ms_dk)

    # ═══ v20 CRIT-1b FIX: market_total = 0 when no real market data ═══
    # Before: fell back to model's own ou_total, creating circular dependency
    _raw_market_total = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else pd.Series(dtype=float, index=df.index),
        errors="coerce"
    )
    _mkt_total_real = _raw_market_total.notna() & (_raw_market_total != 0)
    df["market_total"] = np.where(_mkt_total_real, _raw_market_total.fillna(0), 0.0)

    # has_market: detect if real market data exists (not just zero defaults)
    _mkt_spread_real = df["market_spread"].notna() & (df["market_spread"] != 0)
    df["has_market"] = (_mkt_spread_real | _mkt_total_real).astype(int)

    # ═══ v20 CRIT-1 FIX: Zero spread_vs_market when no market data ═══
    # Before: spread_vs_market = score_diff_pred - 0 = score_diff_pred (LEAKED)
    # After: 0 when has_market=0, actual model-vs-market diff when has_market=1
    df["spread_vs_market"] = np.where(
        df["has_market"] == 1,
        df["score_diff_pred"] - df["market_spread"],
        0.0
    )

    feature_cols = [
        # Heuristic signal
        "score_diff_pred", "win_pct_home", "home_fav", "ou_gap",
        # Net rating (primary signal)
        "net_rtg_diff",
        # Offensive differentials
        "ppg_diff", "fgpct_diff", "threepct_diff", "ftpct_diff",
        # Four factors
        "orb_pct_diff", "fta_rate_diff", "ato_ratio_diff",
        # Defensive differentials
        "opp_ppg_diff", "opp_fgpct_diff", "opp_threepct_diff",
        "steals_diff", "blocks_diff",
        # Turnover quality
        "to_margin_diff", "steals_to_diff",
        # Context
        "win_pct_diff", "form_diff", "tempo_avg",
        "rest_diff", "away_travel",
        # Market line signal (Vegas spread is strongest public predictor)
        "market_spread", "market_total", "spread_vs_market", "has_market",
    ]

    return df[feature_cols].fillna(0)


def _nba_season_weight(season):
    # v20 MED-5 FIX: dynamic year (was hardcoded 2026)
    current = datetime.now().year
    age = current - season
    if age <= 0: return 1.0
    if age == 1: return 1.0
    if age == 2: return 0.9
    if age == 3: return 0.8
    if age == 4: return 0.7
    if age == 5: return 0.6
    return 0.5


# ═══════════════════════════════════════════════════════════════
# NBA v20 Backfill Heuristic — Full JS Signal Chain in Python
#
# v19 fixes retained:
#   - CRIT-1: pred_home_score was raw PPG (no matchup adjustment)
#   - CRIT-2: win_pct_home was hardcoded 0.55 for all games
#   - CRIT-3: model_ml_home was never written → home_fav = 0 always
#   - NEW-CRIT-1: Backfill missing ALL supplementary signals
#
# v20 audit fixes:
#   CRIT-2: three_rate derived from data (was hardcoded 0.40)
#   CRIT-3: ALIGN-5 spread-preserving total cap added
#   CRIT-4: O/U computed BEFORE score clamping (matches JS order)
#   HIGH-1: True Shooting % signal added (weight 0.05)
#   HIGH-4: Rim protection uses actual foulsPerGame (was default 20.0)
#   HIGH-5: Moneyline cap ±500 removed (matches JS)
#   MED-3:  Neutral site check added
# ═══════════════════════════════════════════════════════════════
def _nba_backfill_heuristic(df):
    """
    Compute realistic heuristic predictions for historical NBA rows.
    Mirrors JS nbaPredictGame() so ML trains on the same signal distribution
    as live predictions.
    """
    df = df.copy()

    # ── League averages (static defaults; dynamic will override in train_nba) ──
    LG = {
        "offRtg": 113.5, "pace": 99.5, "fgPct": 0.471, "threePct": 0.365,
        "eFGpct": 0.543, "toPct": 14.5, "orbPct": 0.245, "ftaRate": 0.270,
        "steals": 7.5, "blocks": 5.0, "ts": 0.578,
    }
    # Use dynamic averages if available (set by train_nba before calling merge)
    _dyn = getattr(nba_build_features, "_league_averages", None)
    if _dyn:
        for k, v in _dyn.items():
            if k in LG and v is not None:
                LG[k] = v

    def _safe(val, default):
        """Safely convert to float, return default if NaN/None."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    pred_home_scores = []
    pred_away_scores = []
    win_pcts = []
    model_mls = []
    ou_totals = []  # v20 CRIT-4: track O/U separately (computed before clamping)

    for idx, row in df.iterrows():
        # ── Extract raw stats with safe defaults ──
        h_ppg    = _safe(row.get("home_ppg"), 110)
        a_ppg    = _safe(row.get("away_ppg"), 110)
        h_oppPpg = _safe(row.get("home_opp_ppg"), 110)
        a_oppPpg = _safe(row.get("away_opp_ppg"), 110)
        h_pace   = _safe(row.get("home_tempo"), 99.5)
        a_pace   = _safe(row.get("away_tempo"), 99.5)
        h_fgPct  = _safe(row.get("home_fgpct"), LG["fgPct"])
        a_fgPct  = _safe(row.get("away_fgpct"), LG["fgPct"])
        h_3pct   = _safe(row.get("home_threepct"), LG["threePct"])
        a_3pct   = _safe(row.get("away_threepct"), LG["threePct"])
        h_assists   = _safe(row.get("home_assists"), 25)
        a_assists   = _safe(row.get("away_assists"), 25)
        h_turnovers = _safe(row.get("home_turnovers"), 14)
        a_turnovers = _safe(row.get("away_turnovers"), 14)
        h_orb_pct   = _safe(row.get("home_orb_pct"), LG["orbPct"])
        a_orb_pct   = _safe(row.get("away_orb_pct"), LG["orbPct"])
        h_fta_rate  = _safe(row.get("home_fta_rate"), LG["ftaRate"])
        a_fta_rate  = _safe(row.get("away_fta_rate"), LG["ftaRate"])
        h_ato       = _safe(row.get("home_ato_ratio"), 1.8)
        a_ato       = _safe(row.get("away_ato_ratio"), 1.8)
        h_steals    = _safe(row.get("home_steals"), LG["steals"])
        a_steals    = _safe(row.get("away_steals"), LG["steals"])
        h_blocks    = _safe(row.get("home_blocks"), LG["blocks"])
        a_blocks    = _safe(row.get("away_blocks"), LG["blocks"])
        h_netRtg    = _safe(row.get("home_net_rtg"), 0)
        a_netRtg    = _safe(row.get("away_net_rtg"), 0)
        h_form      = _safe(row.get("home_form"), 0)
        a_form      = _safe(row.get("away_form"), 0)
        h_wins      = _safe(row.get("home_wins"), 20)
        h_losses    = _safe(row.get("home_losses"), 20)
        a_wins      = _safe(row.get("away_wins"), 20)
        a_losses    = _safe(row.get("away_losses"), 20)
        h_rest      = _safe(row.get("home_days_rest"), 2)
        a_rest      = _safe(row.get("away_days_rest"), 2)

        # v20 HIGH-4 FIX: Use actual foulsPerGame (was default 20.0)
        h_fouls     = _safe(row.get("home_fouls_per_game"), 20.0)
        a_fouls     = _safe(row.get("away_fouls_per_game"), 20.0)

        # v20 CRIT-2 FIX: Derive threeAttRate from data (was hardcoded 0.40)
        h_fga       = _safe(row.get("home_fga"), 88.0)
        a_fga       = _safe(row.get("away_fga"), 88.0)
        h_threeAtt  = _safe(row.get("home_three_att"), h_fga * 0.40)
        a_threeAtt  = _safe(row.get("away_three_att"), a_fga * 0.40)
        h_fta       = _safe(row.get("home_fta"), 24.0)
        a_fta       = _safe(row.get("away_fta"), 24.0)
        h_threeRate = h_threeAtt / max(h_fga, 1)
        a_threeRate = a_threeAtt / max(a_fga, 1)

        # ── Efficiency ratings (derive from PPG/pace if not directly available) ──
        h_adjOE = (h_ppg / max(h_pace, 85)) * 100
        h_adjDE = (h_oppPpg / max(h_pace, 85)) * 100
        a_adjOE = (a_ppg / max(a_pace, 85)) * 100
        a_adjDE = (a_oppPpg / max(a_pace, 85)) * 100

        # ── KenPom Additive Core (mirrors JS AUDIT-C1) ──
        poss = (h_pace + a_pace) / 2
        lgAvg = LG["offRtg"]
        homeScore = ((h_adjOE + a_adjDE - lgAvg) / 100) * poss
        awayScore = ((a_adjOE + h_adjDE - lgAvg) / 100) * poss

        # ── Four Factors (Dean Oliver weights, mirrors JS AUDIT-M2) ──
        # v20 CRIT-2 FIX: uses derived three_rate instead of hardcoded 0.40
        def four_factors(fg_pct, three_pct, three_rate, to_count, team_pace, orb_pct, fta_rate):
            eFG = fg_pct + 0.5 * three_rate * three_pct
            eFG_boost = (eFG - LG["eFGpct"]) * 12.0

            to_pct = (to_count / max(team_pace, 85)) * 100 if team_pace > 0 else LG["toPct"]
            to_boost = (LG["toPct"] - to_pct) * 0.07
            to_boost = max(-2.5, min(2.5, to_boost))

            orb_boost = (orb_pct - LG["orbPct"]) * 4.0
            ftr_boost = (fta_rate - LG["ftaRate"]) * 2.2

            return eFG_boost + to_boost + orb_boost + ftr_boost

        h_ff = four_factors(h_fgPct, h_3pct, h_threeRate, h_turnovers, h_pace, h_orb_pct, h_fta_rate)
        a_ff = four_factors(a_fgPct, a_3pct, a_threeRate, a_turnovers, a_pace, a_orb_pct, a_fta_rate)

        # ── Tempo scaling (ALIGN-3) ──
        nba_avg_pace = LG["pace"]
        tempo_scale = poss / nba_avg_pace if nba_avg_pace > 0 else 1.0

        # ── Defensive disruption (steals/blocks only — no oppFgPct double-count) ──
        def def_boost(steals_val, blocks_val):
            return ((steals_val - LG["steals"]) * 0.085 +
                    (blocks_val - LG["blocks"]) * 0.065)

        # ── Blowout scaling (ALIGN-2: net rating gap >= 12) ──
        net_rtg_gap = abs(h_netRtg - a_netRtg)
        blowout_scale = min(1.4, 1.0 + (net_rtg_gap - 12) / 45) if net_rtg_gap >= 12 else 1.0

        # Apply Four Factors + defensive disruption with blowout scaling
        homeScore += h_ff * tempo_scale * 0.30 * blowout_scale
        awayScore += a_ff * tempo_scale * 0.30 * blowout_scale
        homeScore += def_boost(h_steals, h_blocks) * 0.22 * blowout_scale
        awayScore += def_boost(a_steals, a_blocks) * 0.22 * blowout_scale

        # ── ATO ball control differential ──
        h_ato_diff = (h_ato - 1.8)
        a_ato_diff = (a_ato - 1.8)
        ato_boost = (h_ato_diff - a_ato_diff) * 0.4
        homeScore += ato_boost * 0.5
        awayScore -= ato_boost * 0.5

        # ── Turnover margin signal (ALIGN-4) ──
        to_margin_h = h_steals - h_turnovers
        to_margin_a = a_steals - a_turnovers
        to_margin_boost = (to_margin_h - to_margin_a) * 0.08
        homeScore += to_margin_boost * 0.5
        awayScore -= to_margin_boost * 0.5

        # ── v20 HIGH-1 FIX: True Shooting % (was missing in v19 backfill) ──
        def ts_boost(ppg_val, fga_val, fta_val):
            if fga_val <= 0 or fta_val <= 0:
                return 0
            tsa = fga_val + 0.44 * fta_val
            if tsa <= 0:
                return 0
            ts = ppg_val / (2 * tsa)
            return max(-2.5, min(2.5, (ts - LG["ts"]) * 15))
        homeScore += ts_boost(h_ppg, h_fga, h_fta) * 0.05
        awayScore += ts_boost(a_ppg, a_fga, a_fta) * 0.05

        # ── Rim protection (v19-H1 FIX: clamp to >= 0) ──
        # v20 HIGH-4 FIX: uses actual foulsPerGame (was default 20.0)
        def rim_protection(blk, opp_fouls):
            blk_bonus = (blk - LG["blocks"]) * 0.18
            foul_penalty = (opp_fouls - 20) * -0.06
            return max(0, blk_bonus + foul_penalty)

        awayScore -= rim_protection(h_blocks, a_fouls) * 0.15
        homeScore -= rim_protection(a_blocks, h_fouls) * 0.15

        # ── v20 MED-3 FIX: Check neutral_site (was always applying HCA) ──
        _is_neutral = row.get("neutral_site", False)
        if _is_neutral is True or _is_neutral == 1 or str(_is_neutral).lower() == "true":
            pass  # No HCA for neutral site games
        else:
            homeScore += 2.4 / 2
            awayScore -= 2.4 / 2

        # ── B2B rest penalties ──
        if h_rest == 0:
            homeScore -= 3.0
        elif a_rest == 0:
            awayScore -= 3.6
        elif h_rest - a_rest >= 3:
            homeScore += 1.2
        elif a_rest - h_rest >= 3:
            awayScore += 1.2

        # ── Form adjustment (per-team weight, mirrors JS AUDIT-L1) ──
        h_total_games = h_wins + h_losses
        a_total_games = a_wins + a_losses
        h_fw = min(0.10, 0.10 * (min(h_total_games, 30) / 30) ** 0.5)
        a_fw = min(0.10, 0.10 * (min(a_total_games, 30) / 30) ** 0.5)
        homeScore += h_form * h_fw * 3
        awayScore += a_form * a_fw * 3

        # ── v20 CRIT-4 FIX: Compute O/U BEFORE clamping (matches JS order) ──
        ou_total = round((homeScore + awayScore) * 0.992, 1)

        # ── v20 CRIT-3 FIX: ALIGN-5 spread-preserving total cap ──
        # Was missing in v19 — JS applies this before individual clamping
        raw_total = homeScore + awayScore
        if raw_total > 260:
            current_spread = homeScore - awayScore
            mid = 260 / 2
            homeScore = mid + current_spread / 2
            awayScore = mid - current_spread / 2

        # ── Score clamping (mirrors JS NBA-16) ──
        homeScore = max(85, min(155, homeScore))
        awayScore = max(85, min(155, awayScore))

        # ── Win probability (base-e sigmoid, validated σ=7.0) ──
        # AUDIT FIX: was base-10 with dynamic σ=12.0 — matched JS but wrong for ML training.
        # Base-e with σ=7.0 validated via Brier sweep in v27 retrain.
        spread = homeScore - awayScore
        hwp = 1 / (1 + np.exp(-spread / 7.0))
        hwp = max(0.05, min(0.95, hwp))

        # ── Moneyline (mirrors JS conversion) ──
        # v20 HIGH-5 FIX: removed ±500 cap (JS has no cap)
        if hwp >= 0.5:
            mml = -round((hwp / (1 - hwp)) * 100)
        else:
            mml = round(((1 - hwp) / hwp) * 100)

        pred_home_scores.append(round(homeScore, 1))
        pred_away_scores.append(round(awayScore, 1))
        win_pcts.append(round(hwp, 4))
        model_mls.append(mml)
        ou_totals.append(ou_total)

    df["pred_home_score"] = pred_home_scores
    df["pred_away_score"] = pred_away_scores
    df["win_pct_home"] = win_pcts
    df["model_ml_home"] = model_mls
    df["ou_total"] = ou_totals  # v20 CRIT-4: pre-clamp O/U

    # Log diagnostics
    spread_pred = df["pred_home_score"] - df["pred_away_score"]
    wp = df["win_pct_home"]
    print(f"  NBA heuristic backfill v20: {len(df)} rows | "
          f"spread std={spread_pred.std():.2f}, range=[{spread_pred.min():.1f}, {spread_pred.max():.1f}] | "
          f"win_pct std={wp.std():.3f}, range=[{wp.min():.3f}, {wp.max():.3f}]")

    return df


def _nba_merge_historical(current_df):
    hist_rows = sb_get(
        "nba_historical",
        "is_outlier_season=eq.false&actual_home_score=not.is.null&select=*&order=season.desc&limit=50000"
    )
    if not hist_rows:
        print("  WARNING: nba_historical empty -- current season only")
        if current_df is not None and len(current_df) > 0:
            return current_df, None, 0
        return pd.DataFrame(), None, 0
    hist_df = pd.DataFrame(hist_rows)
    for col in hist_df.columns:
        if col not in ['home_team', 'away_team', 'game_date', 'id', 'season', 'is_outlier_season', 'home_win', 'result_entered']:
            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
    if "season" in hist_df.columns:
        hist_df["season_weight"] = hist_df["season"].apply(
            lambda s: _nba_season_weight(int(s)) if pd.notna(s) else 0.5
        )

    # ── v19 FIX: Backfill heuristic predictions on historical rows ──
    print("  Backfilling NBA heuristic predictions on historical rows...")
    hist_df = _nba_backfill_heuristic(hist_df)

    if current_df is not None and len(current_df) > 0:
        combined = pd.concat([hist_df, current_df], ignore_index=True)
    else:
        combined = hist_df
    weights = combined["season_weight"].fillna(1.0).astype(float).values if "season_weight" in combined.columns else None
    n_hist = len(hist_df)
    n_curr = len(current_df) if current_df is not None else 0
    print(f"  NBA corpus: {n_hist} historical + {n_curr} current = {n_hist + n_curr}")
    return combined, weights, n_hist

def train_nba():
    """NBA model training — stacking ensemble with isotonic calibration."""
    import traceback as _tb
    try:
        rows = sb_get("nba_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        # ═══ v20 MED-6 FIX: Derive league averages BEFORE merge ═══
        # Before: averages were set at line 442 AFTER merge at line 436,
        # so backfill ALWAYS used static defaults, never dynamic averages.
        try:
            _nba_lg = compute_nba_league_averages()
            if _nba_lg:
                nba_build_features._league_averages = _nba_lg
                print(f"  Using dynamic NBA averages ({len(_nba_lg)} stats)")
        except Exception as e:
            print(f"  Dynamic NBA averages failed ({e}), using static")

        # Merge with historical corpus (2021-2025)
        # Now backfill will pick up dynamic averages via getattr
        df, sample_weights, n_historical = _nba_merge_historical(current_df)
        if len(df) < 10:
            return {"error": "Not enough NBA data", "n": len(df), "n_current": len(current_df)}

        X  = nba_build_features(df)
        y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
        y_win    = (y_margin > 0).astype(int)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n = len(df)
        cv_folds = min(10, n)  # 5 for Railway; set 10 for local

        if n >= 200:
            # ── Stacking ensemble: XGB+CAT+RF at 50 est (sweep-optimized) ──
            rf_reg = RandomForestRegressor(
                n_estimators=50, max_depth=6,
                min_samples_leaf=15, max_features=0.7,
                random_state=42, n_jobs=1,  # Railway: single core; set -1 for local
            )
            if HAS_XGB:
                xgb_reg = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, random_state=42, tree_method="hist", verbosity=0)
            if HAS_CAT:
                cat_reg = CatBoostRegressor(iterations=50, depth=4, learning_rate=0.06, subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0)

            ensemble_parts = []
            if HAS_XGB: ensemble_parts.append('XGB')
            if HAS_CAT: ensemble_parts.append('CAT')
            ensemble_parts.append('RF')
            ensemble_label = '+'.join(ensemble_parts)
            print(f"  NBA: Training stacking ensemble on {n} rows (ts-cv, {ensemble_label}, 50 est)...")

            reg_models = {"rf": rf_reg}
            if HAS_XGB: reg_models["xgb"] = xgb_reg
            if HAS_CAT: reg_models["cat"] = cat_reg

            oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=sample_weights)
            oof_rf = oof["rf"]

            rf_reg.fit(X_scaled, y_margin)
            if HAS_XGB: xgb_reg.fit(X_scaled, y_margin)
            if HAS_CAT: cat_reg.fit(X_scaled, y_margin)

            oof_cols = [oof_rf]
            if HAS_XGB: oof_cols.append(oof["xgb"])
            if HAS_CAT: oof_cols.append(oof["cat"])
            meta_X = np.column_stack(oof_cols)
            meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            meta_reg.fit(meta_X, y_margin)

            base_regs = [rf_reg]
            if HAS_XGB: base_regs.append(xgb_reg)
            if HAS_CAT: base_regs.append(cat_reg)
            reg = StackedRegressor(base_regs, meta_reg, scaler)

            # True stacked OOF MAE
            oof_meta = meta_reg.predict(meta_X)
            reg_cv_mae = float(np.mean(np.abs(oof_meta - y_margin.values)))
            print(f"  NBA stacked OOF MAE: {reg_cv_mae:.3f}")

            try:
                explainer = shap.TreeExplainer(xgb_reg if HAS_XGB else rf_reg)
            except Exception as _shap_err:
                print(f'  SHAP explainer failed: {_shap_err}')
                explainer = None
            model_type = "StackedEnsemble_v4_TSCV"
            meta_weights = meta_reg.coef_.round(4).tolist()
            print(f"  NBA meta weights: {meta_weights}")

            # ── Bias correction ──
            bias_correction = float(np.mean(oof_meta - y_margin.values))
            print(f"  NBA bias correction: {bias_correction:+.3f} pts")

            # ── Stacked classifier ──
            gbm_clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.06, subsample=0.8,
                min_samples_leaf=20, random_state=42,
            )
            rf_clf = RandomForestClassifier(
                n_estimators=100, max_depth=6,
                min_samples_leaf=15, max_features=0.7,
                random_state=42, n_jobs=1,
            )
            lr_clf = LogisticRegression(max_iter=1000, C=1.0)

            oof_gbm_p = cross_val_predict(gbm_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
            oof_rf_p  = cross_val_predict(rf_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
            oof_lr_p  = cross_val_predict(lr_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]

            gbm_clf.fit(X_scaled, y_win)
            rf_clf.fit(X_scaled, y_win)
            lr_clf.fit(X_scaled, y_win)

            meta_clf_X = np.column_stack([oof_gbm_p, oof_rf_p, oof_lr_p])
            meta_lr = LogisticRegression(max_iter=1000, C=1.0)
            meta_lr.fit(meta_clf_X, y_win)
            clf = StackedClassifier([gbm_clf, rf_clf, lr_clf], meta_lr)

            # ── Isotonic calibration on OOF classifier probs ──
            oof_stacked_probs = meta_lr.predict_proba(meta_clf_X)[:, 1]
            isotonic = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            isotonic.fit(oof_stacked_probs, y_win.values)
            print(f"  NBA isotonic calibration fitted on {len(oof_stacked_probs)} OOF samples")

        else:
            # Simple models for small data
            reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                             learning_rate=0.1, random_state=42)
            reg.fit(X_scaled, y_margin)
            reg_cv = cross_val_score(reg, X_scaled, y_margin,
                                      cv=min(5, n), scoring="neg_mean_absolute_error")
            reg_cv_mae = float(-reg_cv.mean())
            clf = CalibratedClassifierCV(
                LogisticRegression(max_iter=1000), cv=min(5, n)
            )
            clf.fit(X_scaled, y_win)
            try:
                explainer = shap.TreeExplainer(reg)
            except Exception as _shap_err:
                print(f'  SHAP explainer failed: {_shap_err}')
                explainer = None
            model_type = "GBM"
            bias_correction = 0.0
            isotonic = None
            meta_weights = []

        bundle = {
            "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
            "feature_cols": list(X.columns), "n_train": n,
            "mae_cv": reg_cv_mae, "model_type": model_type,
            "trained_at": datetime.utcnow().isoformat(),
            "bias_correction": bias_correction,
            "isotonic": isotonic,
            "meta_weights": meta_weights if n >= 200 else [],
        }
        save_model("nba", bundle)
        return {"status": "trained", "n_train": n, "model_type": model_type,
                "mae_cv": round(reg_cv_mae, 3), "features": list(X.columns),
                "bias_correction": round(bias_correction, 3),
                "meta_weights": meta_weights if n >= 200 else []}

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__,
                "traceback": _tb.format_exc()}

def _load_nba_v26():
    """Load Lasso v26 model from local pkl file (git-deployed)."""
    import pickle, os
    for path in ["nba_model_local.pkl", "models/nba_model_local.pkl"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("architecture", "").startswith("Lasso"):
                return bundle
    return None


def predict_nba(game: dict):
    # Try v26 Lasso first, fall back to Supabase stacked ensemble
    bundle = _load_nba_v26()
    is_v26 = bundle is not None

    if not is_v26:
        bundle = load_model("nba")

    if not bundle:
        return {"error": "NBA model not trained — call /train/nba first"}

    # Build a single-row DataFrame with all features the model might need.
    _RAW_DEFAULTS = {
        "pred_home_score": 110, "pred_away_score": 110,
        "home_net_rtg": 0, "away_net_rtg": 0,
        "win_pct_home": 0.5, "ou_total": 228,
        "model_ml_home": 0, "market_ou_total": 228,
        "market_spread_home": 0,
        "home_ppg": 110, "away_ppg": 110,
        "home_opp_ppg": 110, "away_opp_ppg": 110,
        "home_fgpct": 0.46, "away_fgpct": 0.46,
        "home_threepct": 0.365, "away_threepct": 0.365,
        "home_ftpct": 0.77, "away_ftpct": 0.77,
        "home_assists": 25, "away_assists": 25,
        "home_turnovers": 14, "away_turnovers": 14,
        "home_tempo": 100, "away_tempo": 100,
        "home_orb_pct": 0.25, "away_orb_pct": 0.25,
        "home_fta_rate": 0.28, "away_fta_rate": 0.28,
        "home_ato_ratio": 1.8, "away_ato_ratio": 1.8,
        "home_opp_fgpct": 0.46, "away_opp_fgpct": 0.46,
        "home_opp_threepct": 0.365, "away_opp_threepct": 0.365,
        "home_steals": 7.5, "away_steals": 7.5,
        "home_blocks": 5.0, "away_blocks": 5.0,
        "home_wins": 20, "away_wins": 20,
        "home_losses": 20, "away_losses": 20,
        "home_form": 0, "away_form": 0,
        "home_days_rest": 2, "away_days_rest": 2,
        "away_travel_dist": 0,
    }
    # Merge: game values override defaults
    merged = {**_RAW_DEFAULTS, **game}
    row = pd.DataFrame([merged])

    # ── v26 Lasso prediction path ──
    if is_v26:
        try:
            from nba_build_features_v25 import nba_build_features as build_v25
            X = build_v25(row)
        except ImportError:
            X = nba_build_features(row)

        feature_list = bundle["feature_list"]
        # Ensure all features exist (fill missing with 0)
        for f in feature_list:
            if f not in X.columns:
                X[f] = 0.0
        X_slim = X[feature_list]

        X_s = bundle["scaler"].transform(X_slim)
        margin = float(bundle["model"].predict(X_s)[0])

        # Win probability: sigmoid → isotonic calibration
        # AUDIT FIX: read σ from bundle (v27=7.0), not hardcoded 8.0
        _sigma = bundle.get("sigma", 7.0)
        raw_prob = 1.0 / (1.0 + np.exp(-margin / _sigma))
        calibrator = bundle.get("calibrator")
        if calibrator is not None:
            try:
                win_prob = float(calibrator.predict([raw_prob])[0])
            except Exception:
                win_prob = raw_prob
        else:
            win_prob = raw_prob

        # Feature explanation via Lasso coefficients (replaces SHAP)
        coefs = bundle["model"].coef_
        scaled_vals = X_s[0]
        contributions = coefs * scaled_vals
        shap_out = [
            {"feature": f, "shap": round(float(c), 4), "value": round(float(X_slim[f].iloc[0]), 3)}
            for f, c in zip(feature_list, contributions)
        ]
        shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

        return {
            "sport": "NBA",
            "ml_margin": round(margin, 2),
            "ml_win_prob_home": round(win_prob, 4),
            "ml_win_prob_away": round(1 - win_prob, 4),
            "shap": shap_out,
            "model_meta": {
                "n_train": bundle.get("n_games"),
                "mae_cv": bundle.get("cv_mae"),
                "trained_at": bundle.get("trained_at"),
                "model_type": bundle.get("architecture", "Lasso_v26"),
                "n_features": bundle.get("n_features_selected"),
                "has_isotonic": calibrator is not None,
            },
        }

    # ── Legacy stacked ensemble prediction path ──
    X = nba_build_features(row)
    X_s = bundle["scaler"].transform(X[bundle["feature_cols"]])

    margin = float(bundle["reg"].predict(X_s)[0])

    # Apply bias correction if available
    bias = bundle.get("bias_correction", 0.0)
    if bias:
        margin -= bias

    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])

    # Apply isotonic calibration if available
    isotonic = bundle.get("isotonic")
    if isotonic is not None:
        try:
            win_prob = float(isotonic.predict([win_prob])[0])
        except Exception:
            pass

    shap_out = []
    if bundle.get("explainer") is not None:
        try:
            shap_vals = bundle["explainer"].shap_values(X_s)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            sv = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
            shap_out = [
                {"feature": f, "shap": round(float(v), 4), "value": round(float(X[f].iloc[0]), 3)}
                for f, v in zip(bundle["feature_cols"], sv)
            ]
            shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)
        except Exception:
            pass

    return {
        "sport": "NBA",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out,
        "model_meta": {"n_train": bundle.get("n_train"), "mae_cv": bundle.get("mae_cv"),
                       "trained_at": bundle.get("trained_at"),
                       "model_type": bundle.get("model_type", "unknown"),
                       "has_isotonic": isotonic is not None},
    }

# ═══════════════════════════════════════════════════════════════
# NCAAB MODEL (v17 — Re-audit fixes R1-R10)
#   R1: Home bias correction via neutral_em_diff + bias subtraction
#   R2: Heuristic signal re-introduced as capped feature
#   R3: Conference game flag + season phase features
#   R4: Isotonic calibration on classifier probabilities
#   R5: Rest days wiring (column detection)
#   R6: Post-training bias correction stored in bundle
#   R7: ElasticNet replaces Ridge for diversity; meta weights logged
#   R8: SOS-weighted interaction features
# ═══════════════════════════════════════════════════════════════

# Conference HCA lookup (same as heuristic, used to decompose adj_em_diff)
_NCAA_CONF_HCA = {
    "Big 12": 3.8, "Southeastern Conference": 3.7, "SEC": 3.7,
    "Big Ten": 3.6, "Big Ten Conference": 3.6,
    "Atlantic Coast Conference": 3.4, "ACC": 3.4,
    "Big East": 3.3, "Big East Conference": 3.3,
    "Pac-12": 3.0, "Pac-12 Conference": 3.0,
    "Mountain West Conference": 3.2, "Mountain West": 3.2,
    "American Athletic Conference": 3.0, "AAC": 3.0,
    "West Coast Conference": 2.8, "WCC": 2.8,
    "Atlantic 10 Conference": 2.7, "A-10": 2.7,
    "Missouri Valley Conference": 2.9, "MVC": 2.9,
}
