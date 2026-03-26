"""
Dynamic league averages derived from historical data.
Replaces hardcoded constants where possible.
"""
import numpy as np
import pandas as pd
from db import sb_get


def _winsorized_mean(series, lower=0.05, upper=0.95):
    """
    Winsorized mean: cap outliers at percentile bounds instead of removing.
    More robust than raw mean for league averages (blowouts, rain-shortened, etc.)
    """
    s = series.dropna()
    if len(s) < 20:
        return float(s.mean()) if len(s) > 0 else None
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return float(s.clip(lo, hi).mean())


_mlb_cache = None
def compute_mlb_season_constants():
    global _mlb_cache
    if _mlb_cache is not None:
        return _mlb_cache
    """
    Derive MLB league averages per season from mlb_historical.
    Returns dict like SEASON_CONSTANTS: {2024: {"lg_rpg": 4.38, ...}, ...}
    
    What we CAN derive from game data:
      - lg_rpg: mean runs per team per game
      - lg_fip: mean of all SP FIP values (proxy)
      - lg_woba: mean of all team wOBA values (proxy)
    
    What we CANNOT derive (need FanGraphs Guts):
      - woba_scale: requires FanGraphs linear weights methodology
      - pa_pg: requires plate appearance data not in our tables
    
    So we derive what we can and keep woba_scale/pa_pg from static table.
    """
    # Static values that can't be derived from game data
    STATIC_WOBA_SCALE = {
        2015: 1.24, 2016: 1.21, 2017: 1.21, 2018: 1.23, 2019: 1.17,
        2021: 1.22, 2022: 1.24, 2023: 1.21, 2024: 1.25, 2025: 1.24, 2026: 1.24,
    }
    STATIC_PA_PG = {
        2015: 38.0, 2016: 38.0, 2017: 38.1, 2018: 37.9, 2019: 38.2,
        2021: 37.9, 2022: 37.6, 2023: 37.8, 2024: 37.8, 2025: 37.8, 2026: 37.8,
    }

    rows = sb_get(
        "mlb_historical",
        "is_outlier_season=eq.0&actual_home_runs=not.is.null"
        "&select=season,actual_home_runs,actual_away_runs,home_woba,away_woba,home_sp_fip,away_sp_fip"
        "&limit=100000"
    )
    if not rows:
        print("  WARNING: mlb_historical empty — using static constants")
        return None

    df = pd.DataFrame(rows)
    for col in ["actual_home_runs", "actual_away_runs", "home_woba", "away_woba",
                "home_sp_fip", "away_sp_fip", "season"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["season", "actual_home_runs", "actual_away_runs"])

    constants = {}
    for season, grp in df.groupby("season"):
        season = int(season)
        # lg_rpg: average runs per team per game (winsorized to exclude blowouts)
        all_runs = pd.concat([grp["actual_home_runs"], grp["actual_away_runs"]])
        lg_rpg = round(_winsorized_mean(all_runs), 2)

        # lg_woba: average of all team wOBA values (proxy for league wOBA)
        all_woba = pd.concat([
            grp["home_woba"].dropna(),
            grp["away_woba"].dropna()
        ])
        lg_woba = round(_winsorized_mean(all_woba), 3) if len(all_woba) > 50 else None

        # lg_fip: average of all SP FIP values
        all_fip = pd.concat([
            grp["home_sp_fip"].dropna(),
            grp["away_sp_fip"].dropna()
        ])
        # Filter out sentinel values (4.25 = missing)
        all_fip = all_fip[all_fip != 4.25]
        lg_fip = round(_winsorized_mean(all_fip), 2) if len(all_fip) > 50 else None

        constants[season] = {
            "lg_rpg": lg_rpg,
            "lg_woba": lg_woba if lg_woba else 0.315,  # fallback
            "lg_fip": lg_fip if lg_fip else 4.10,       # fallback
            "woba_scale": STATIC_WOBA_SCALE.get(season, 1.24),
            "pa_pg": STATIC_PA_PG.get(season, 37.8),
        }

    n_seasons = len(constants)
    _mlb_cache = constants  # Cache for subsequent calls
    print(f"  MLB dynamic constants: {n_seasons} seasons derived from {len(df)} games")
    for s in sorted(constants.keys()):
        c = constants[s]
        print(f"    {s}: lg_rpg={c['lg_rpg']}, lg_woba={c['lg_woba']}, lg_fip={c['lg_fip']}")

    return constants


_nba_cache = None
def compute_nba_league_averages():
    global _nba_cache
    if _nba_cache is not None:
        return _nba_cache
    """
    Derive NBA league averages from nba_historical for use as feature builder defaults.
    Returns dict of stat_name -> average value.
    """
    rows = sb_get(
        "nba_historical",
        "is_outlier_season=eq.false&actual_home_score=not.is.null"
        "&select=home_ppg,away_ppg,home_opp_ppg,away_opp_ppg,"
        "home_fgpct,away_fgpct,home_threepct,away_threepct,"
        "home_ftpct,away_ftpct,home_orb_pct,away_orb_pct,"
        "home_fta_rate,away_fta_rate,home_ato_ratio,away_ato_ratio,"
        "home_steals,away_steals,home_blocks,away_blocks,"
        "home_turnovers,away_turnovers,home_assists,away_assists,"
        "home_tempo,away_tempo,home_fga,away_fga,home_fta,away_fta"
        "&limit=50000"
    )
    # Fallback: if FGA/FTA columns don't exist in nba_historical, query without them
    if not rows:
        rows = sb_get(
            "nba_historical",
            "is_outlier_season=eq.false&actual_home_score=not.is.null"
            "&select=home_ppg,away_ppg,home_opp_ppg,away_opp_ppg,"
            "home_fgpct,away_fgpct,home_threepct,away_threepct,"
            "home_ftpct,away_ftpct,home_orb_pct,away_orb_pct,"
            "home_fta_rate,away_fta_rate,home_ato_ratio,away_ato_ratio,"
            "home_steals,away_steals,home_blocks,away_blocks,"
            "home_turnovers,away_turnovers,home_assists,away_assists,"
            "home_tempo,away_tempo"
            "&limit=50000"
        )
    if not rows:
        print("  WARNING: nba_historical empty — using static defaults")
        return None

    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    averages = {}
    stat_pairs = [
        ("ppg", "home_ppg", "away_ppg"),
        ("opp_ppg", "home_opp_ppg", "away_opp_ppg"),
        ("fgpct", "home_fgpct", "away_fgpct"),
        ("threepct", "home_threepct", "away_threepct"),
        ("ftpct", "home_ftpct", "away_ftpct"),
        ("orb_pct", "home_orb_pct", "away_orb_pct"),
        ("fta_rate", "home_fta_rate", "away_fta_rate"),
        ("ato_ratio", "home_ato_ratio", "away_ato_ratio"),
        ("steals", "home_steals", "away_steals"),
        ("blocks", "home_blocks", "away_blocks"),
        ("turnovers", "home_turnovers", "away_turnovers"),
        ("assists", "home_assists", "away_assists"),
        ("tempo", "home_tempo", "away_tempo"),
    ]

    for name, h_col, a_col in stat_pairs:
        if h_col in df.columns and a_col in df.columns:
            combined = pd.concat([df[h_col].dropna(), df[a_col].dropna()])
            if len(combined) > 100:
                averages[name] = round(_winsorized_mean(combined), 3)

    # Derive league TS% from PPG / (2 * (FGA + 0.44 * FTA))
    _h_fga = df["home_fga"].dropna() if "home_fga" in df.columns else pd.Series(dtype=float)
    _a_fga = df["away_fga"].dropna() if "away_fga" in df.columns else pd.Series(dtype=float)
    _h_fta = df["home_fta"].dropna() if "home_fta" in df.columns else pd.Series(dtype=float)
    _a_fta = df["away_fta"].dropna() if "away_fta" in df.columns else pd.Series(dtype=float)
    _h_ppg = df["home_ppg"].dropna() if "home_ppg" in df.columns else pd.Series(dtype=float)
    _a_ppg = df["away_ppg"].dropna() if "away_ppg" in df.columns else pd.Series(dtype=float)
    if len(_h_fga) > 100 and len(_h_fta) > 100:
        _all_ppg = pd.concat([_h_ppg, _a_ppg])
        _all_fga = pd.concat([_h_fga, _a_fga])
        _all_fta = pd.concat([_h_fta, _a_fta])
        _all_tsa = _all_fga + 0.44 * _all_fta
        _valid = _all_tsa > 0
        _all_ts = _all_ppg[_valid] / (2 * _all_tsa[_valid])
        if len(_all_ts) > 100:
            averages["ts_pct"] = round(float(_all_ts.median()), 4)

    _nba_cache = averages  # Cache for subsequent calls
    print(f"  NBA dynamic averages from {len(df)} historical games:")
    for k, v in sorted(averages.items()):
        print(f"    {k}: {v}")

    return averages


# Default fallbacks if derivation fails
MLB_DEFAULT_CONSTANTS = {
    "lg_woba": 0.315, "woba_scale": 1.24, "lg_rpg": 4.30, "lg_fip": 4.10, "pa_pg": 37.8
}
NBA_DEFAULT_AVERAGES = {
    "ppg": 110, "opp_ppg": 110, "fgpct": 0.46, "threepct": 0.36,
    "ftpct": 0.77, "assists": 25, "turnovers": 14, "tempo": 100,
    "orb_pct": 0.25, "fta_rate": 0.28, "ato_ratio": 1.7,
    "steals": 7.5, "blocks": 5.0, "ts_pct": 0.575,
}
