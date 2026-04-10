"""
mlb_ou_v3_serve.py — Serve-side MLB O/U v3 prediction
======================================================
Triple agreement (residual + classifier + ATS scores) with lineup features.

v3 additions:
  - lineup_delta_sum: lineup change signal (market misses rested starters)
  - lineup_total_top3: top-of-order quality combined (r=-0.111)

Integration:
  In sports/mlb.py predict_mlb_ou(), check bundle.get("_v3_lineup"):
    if bundle.get("_v3_lineup"):
        from mlb_ou_v3_serve import predict_mlb_ou_v3
        return predict_mlb_ou_v3(game, bundle)
"""
import numpy as np
import requests

MLB_API = "https://statsapi.mlb.com/api/v1"


def fetch_pitcher_recent_form(pitcher_id, n_starts=3, timeout=8):
    """Fetch pitcher's last N starts from MLB Stats API game log."""
    if not pitcher_id:
        return None
    try:
        from datetime import datetime
        season = datetime.now().year
        url = (f"{MLB_API}/people/{pitcher_id}/stats"
               f"?stats=gameLog&season={season}&group=pitching")
        r = requests.get(url, timeout=timeout)
        if not r.ok:
            return None
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        starts = []
        for s in reversed(splits):
            stat = s.get("stat", {})
            if int(stat.get("gamesStarted", 0)) > 0:
                runs = float(stat.get("runs", 0))
                ip_str = str(stat.get("inningsPitched", "0"))
                if "." in ip_str:
                    whole, frac = ip_str.split(".")
                    ip = int(whole) + int(frac) / 3.0
                else:
                    ip = float(ip_str)
                starts.append({"runs": runs, "ip": ip})
                if len(starts) >= n_starts:
                    break
        if not starts:
            return None
        return {
            "recent_avg_runs": round(sum(s["runs"] for s in starts) / len(starts), 2),
            "n_starts": len(starts),
        }
    except Exception as e:
        print(f"  [ou_v3] Game log fetch failed for {pitcher_id}: {e}")
        return None


def _parse_ip(ip_str):
    """Parse innings pitched string (e.g., '6.1' = 6.333)."""
    ip_str = str(ip_str or "0")
    if "." in ip_str:
        whole, frac = ip_str.split(".")
        return int(whole) + int(frac) / 3.0
    return float(ip_str)


def fetch_rolling_fip(pitcher_id, target_ip=60, timeout=8):
    """Compute rolling FIP over last ~60 IP, crossing into previous season if needed.
    
    FIP = (13*HR + 3*BB - 2*K) / IP + 3.10
    
    Returns: float FIP, or None if insufficient data.
    """
    if not pitcher_id:
        return None
    try:
        from datetime import datetime
        season = datetime.now().year
        all_starts = []

        # Fetch current + previous season game logs
        for yr in [season, season - 1]:
            url = (f"{MLB_API}/people/{pitcher_id}/stats"
                   f"?stats=gameLog&season={yr}&group=pitching")
            r = requests.get(url, timeout=timeout)
            if not r.ok:
                continue
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            # Reverse so most recent first
            for s in reversed(splits):
                stat = s.get("stat", {})
                if int(stat.get("gamesStarted", 0)) > 0:
                    ip = _parse_ip(stat.get("inningsPitched", "0"))
                    if ip <= 0:
                        continue
                    all_starts.append({
                        "ip": ip,
                        "hr": float(stat.get("homeRuns", 0)),
                        "bb": float(stat.get("baseOnBalls", 0)),
                        "k": float(stat.get("strikeOuts", 0)),
                        "season": yr,
                    })

        if not all_starts:
            return None

        # Accumulate starts until we hit target IP
        total_ip = 0
        total_hr = 0
        total_bb = 0
        total_k = 0
        n_starts = 0

        for s in all_starts:
            total_ip += s["ip"]
            total_hr += s["hr"]
            total_bb += s["bb"]
            total_k += s["k"]
            n_starts += 1
            if total_ip >= target_ip:
                break

        if total_ip < 10:  # Need at least 10 IP for any signal
            return None

        fip = (13 * total_hr + 3 * total_bb - 2 * total_k) / total_ip + 3.10
        fip = max(1.5, min(7.0, fip))  # Sanity clamp

        print(f"  [ou_v3] Rolling FIP for {pitcher_id}: {fip:.2f} "
              f"({n_starts} starts, {total_ip:.0f} IP, "
              f"spans {'2 seasons' if any(s['season'] != all_starts[0]['season'] for s in all_starts[:n_starts]) else '1 season'})")
        return round(fip, 3)

    except Exception as e:
        print(f"  [ou_v3] Rolling FIP fetch failed for {pitcher_id}: {e}")
        return None


def compute_sp_form_combined(home_pitcher_id, away_pitcher_id, home_fip, away_fip):
    """sp_form_combined = sum of (recent ERA - season FIP) for both starters."""
    sp_form = 0.0
    for pitcher_id, season_fip, label in [
        (home_pitcher_id, home_fip or 4.25, "home"),
        (away_pitcher_id, away_fip or 4.25, "away"),
    ]:
        form = fetch_pitcher_recent_form(pitcher_id)
        if form and form["n_starts"] >= 1:
            delta = form["recent_avg_runs"] - season_fip
            sp_form += delta
            print(f"  [ou_v3] {label} SP: last {form['n_starts']} starts avg {form['recent_avg_runs']:.1f} runs, "
                  f"FIP {season_fip:.2f}, delta={delta:+.2f}")
    return round(sp_form, 3)


def _build_feature_dict(game, sp_form):
    """Build full feature dict for all 3 components.
    
    Clamps early-season stats to training data range (Apr 15+ full-season values).
    Training never sees FIP < 2.5 or bullpen ERA < 2.5, so we clamp to prevent
    out-of-distribution extrapolation while preserving real signal.
    """
    _f = lambda k, d=0: float(game.get(k, d) or d)

    market_total = _f("market_ou_total", 0) or _f("market_total", 0)

    # Clamp to training data range — preserves "elite" vs "bad" signal
    home_fip = max(2.5, min(6.5, _f("home_sp_fip", 4.0) or 4.0))
    away_fip = max(2.5, min(6.5, _f("away_sp_fip", 4.0) or 4.0))
    home_woba = max(0.260, min(0.370, _f("home_woba", 0.315)))
    away_woba = max(0.260, min(0.370, _f("away_woba", 0.315)))
    home_bp = max(2.5, min(6.5, _f("home_bullpen_era", 4.10)))
    away_bp = max(2.5, min(6.5, _f("away_bullpen_era", 4.10)))
    home_k9 = max(5.0, min(13.0, _f("home_k9", 8.5)))
    away_k9 = max(5.0, min(13.0, _f("away_k9", 8.5)))
    home_bb9 = max(1.5, min(5.5, _f("home_bb9", 3.2)))
    away_bb9 = max(1.5, min(5.5, _f("away_bb9", 3.2)))
    home_sp_ip = _f("home_sp_ip", 5.5)
    away_sp_ip = _f("away_sp_ip", 5.5)
    park_factor = _f("park_factor", 1.0)
    temp_f = _f("temp_f", 72)
    wind_mph = _f("wind_mph", 5)
    wind_out = int(_f("wind_out_flag", 0))

    return {
        "market_total": market_total if market_total > 0 else 9.0,
        "park_factor": park_factor,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_out": wind_out,
        "is_warm": 1 if temp_f > 75 else 0,
        "is_cold": 1 if temp_f < 45 else 0,
        "temp_x_park": ((temp_f - 70) / 30.0) * park_factor,
        "fip_combined": home_fip + away_fip,
        "fip_diff": home_fip - away_fip,
        "bullpen_combined": home_bp + away_bp,
        "k_bb_combined": (home_k9 - home_bb9) + (away_k9 - away_bb9),
        "sp_ip_combined": home_sp_ip + away_sp_ip,
        "bp_exposure_combined": max(0, 5.5 - home_sp_ip) + max(0, 5.5 - away_sp_ip),
        "woba_combined": home_woba + away_woba,
        "woba_diff": home_woba - away_woba,
        "ump_run_env": _f("ump_run_env", 8.5),
        "ump_career_rpg": _f("ump_career_rpg", 8.5),
        "ump_career_bb": _f("ump_career_bb", 6.5),
        "scoring_entropy_combined": _f("scoring_entropy_combined", 5.0),
        "first_inn_rate_combined": _f("first_inn_rate_combined", 0.8),
        "rest_combined": _f("home_rest", 1) + _f("away_rest", 1),
        "series_game_num": _f("series_game_num", 1),
        "lg_rpg": _f("lg_rpg", 9.0),
        "sp_form_combined": sp_form,
        "lineup_delta_sum": _f("lineup_delta_sum", 0),
        "lineup_total_top3": _f("lineup_total_top3", 0.68),
        "lineup_total_woba": _f("lineup_total_woba", 0.630),
    }


def predict_mlb_ou_v3(game, bundle):
    """
    V3 O/U prediction: triple agreement + lineup features.
    """
    market_total = float(game.get("market_ou_total", 0) or game.get("market_total", 0) or 0)
    game_date = game.get("game_date", "")

    # ── Early-season regression applied in _build_feature_dict (no hard cutoff needed) ──

    # ── Clamps in _build_feature_dict handle out-of-distribution inputs ──
    # Rolling FIP tested: r≈0 against residual — market already prices pitcher quality.

    # Compute sp_form_combined using season FIP as baseline
    sp_form = game.get("sp_form_combined")
    if sp_form is None:
        sp_form = compute_sp_form_combined(
            game.get("home_starter_id"),
            game.get("away_starter_id"),
            game.get("home_sp_fip", 4.25),
            game.get("away_sp_fip", 4.25),
        )

    feats = _build_feature_dict(game, sp_form)

    # Residual models
    res_cols = bundle["res_feature_cols"]
    Xr = np.array([[feats.get(f, 0) for f in res_cols]])
    Xr_s = bundle["res_scaler"].transform(Xr)
    res_preds = [float(m.predict(Xr_s)[0]) for m in bundle["res_models"]]
    residual = np.mean(res_preds)

    # Classifier models
    cls_cols = bundle["cls_feature_cols"]
    Xc = np.array([[feats.get(f, 0) for f in cls_cols]])
    Xc_s = bundle["cls_scaler"].transform(Xc)
    cls_preds = []
    for m in bundle["cls_models"]:
        try:
            cls_preds.append(float(m.predict_proba(Xc_s)[0][1]))
        except Exception:
            cls_preds.append(float(m.predict(Xc_s)[0]))
    p_under = np.mean(cls_preds)

    # ATS score models
    ats_cols = bundle["ats_feature_cols"]
    Xa = np.array([[feats.get(f, 0) for f in ats_cols]])
    Xa_s = bundle["ats_scaler"].transform(Xa)
    home_preds = [float(m.predict(Xa_s)[0]) for m in bundle["ats_home_models"]]
    away_preds = [float(m.predict(Xa_s)[0]) for m in bundle["ats_away_models"]]
    ats_home = np.mean(home_preds)
    ats_away = np.mean(away_preds)
    ats_total = ats_home + ats_away
    ats_edge = ats_total - market_total if market_total > 0 else 0

    # Predicted total
    # Cap residual — no model should claim 3+ runs of edge vs market
    capped_residual = max(-2.5, min(2.5, residual))
    if abs(residual) > 2.5:
        print(f"  [ou_v3] Residual capped: {residual:+.2f} → {capped_residual:+.2f} (early-season noise)")
    
    res_pred = (market_total + capped_residual) if market_total > 0 else 9.0
    ats_pred = ats_total if ats_total > 0 else res_pred
    
    # If residual and ATS models disagree by 3+ runs, blend (early-season protection)
    if abs(res_pred - ats_pred) > 3.0 and market_total > 0:
        pred_total = (res_pred + ats_pred) / 2
        print(f"  [ou_v3] Blending: residual says {res_pred:.1f}, ATS says {ats_pred:.1f} → {pred_total:.1f}")
    else:
        pred_total = res_pred
    
    pred_total = max(5.0, min(16.0, pred_total))

    # Triple agreement tiering (from bundle thresholds)
    under_tiers = bundle.get("under_tiers", {})
    over_tiers = bundle.get("over_tiers", {})

    ou_pick = None
    ou_tier = 0
    ou_units = 0

    if market_total > 0:
        for tier in sorted(under_tiers.keys(), key=lambda x: int(x), reverse=True):
            t = under_tiers[tier]
            if (residual <= t.get("res_thresh", -0.3) and
                p_under >= t.get("cls_thresh", 0.52) and
                ats_edge <= t.get("ats_thresh", -0.5)):
                ou_pick = "UNDER"
                ou_tier = int(tier)
                ou_units = int(tier)
                break

        if ou_pick is None:
            for tier in sorted(over_tiers.keys(), key=lambda x: int(x), reverse=True):
                t = over_tiers[tier]
                if (residual >= t.get("res_thresh", 1.0) and
                    p_under <= t.get("cls_thresh", 0.48) and
                    ats_edge >= t.get("ats_thresh", 0.5)):
                    ou_pick = "OVER"
                    ou_tier = int(tier)
                    ou_units = int(tier)
                    break

    n_nz = sum(1 for f in res_cols if abs(feats.get(f, 0)) > 1e-6)
    print(f"  [ou_v3] res={residual:+.2f}, p_under={p_under:.3f}, ats_edge={ats_edge:+.2f}, "
          f"pick={ou_pick or '—'} {ou_units}u, lineup_delta={feats.get('lineup_delta_sum', 0):.3f}, "
          f"coverage={n_nz}/{len(res_cols)}")

    return {
        "sport": "MLB",
        "type": "ou_v3",
        "pred_total": round(pred_total, 2),
        "market_total": round(market_total, 1) if market_total > 0 else None,
        "ou_edge": round(residual, 3),
        "ou_pick": ou_pick,
        "ou_units": ou_units,
        "ou_tier": ou_tier,
        "ou_res_avg": round(residual, 3),
        "residual": round(residual, 3),
        "p_under": round(p_under, 3),
        "ats_total": round(ats_total, 2),
        "ats_edge": round(ats_edge, 2),
        "sp_form_combined": round(sp_form, 3),
        "lineup_delta_sum": round(feats.get("lineup_delta_sum", 0), 4),
        "model_meta": {
            "model_type": bundle.get("model_type", "mlb_ou_v3"),
            "n_train": bundle.get("n_train", 0),
            "mae_cv": bundle.get("mae_cv", 0),
            "ats_mae": bundle.get("ats_mae", 0),
        },
    }
