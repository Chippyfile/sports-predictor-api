
# ═══════════════════════════════════════════════════════════════
# ROUTES — MLB Daily Cron (grade completed games)
# Schedule: 12:30 AM EST (05:30 UTC) via GitHub Actions
# ═══════════════════════════════════════════════════════════════

_mlb_cron_lock = False

@app.route("/cron/mlb-daily", methods=["GET", "POST"])
def route_mlb_daily():
    """
    MLB daily grading — fills final scores, computes ML/RL/O-U correctness.
    Modes:
      grade — fill scores + grade for last 3 days of ungraded games
      auto  — always grade (MLB prediction is frontend-only for now)
    """
    import time as _time, traceback
    from datetime import datetime, timezone, timedelta
    global _mlb_cron_lock

    if _mlb_cron_lock:
        return jsonify({"status": "already_running"}), 429

    _mlb_cron_lock = True
    start = _time.time()

    try:
        mode = request.args.get("mode", "grade")
        now_utc = datetime.now(timezone.utc)
        now_est = now_utc - timedelta(hours=5)
        today_est = now_est.strftime("%Y-%m-%d")
        hour_est = now_est.hour

        import requests as _req
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

        results = {"mode": mode, "date": today_est, "hour_est": hour_est}

        graded = 0
        for days_ago in range(0, 4):  # check today + last 3 days
            check_date = (now_est - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            compact = check_date.replace("-", "")

            # Get pending MLB predictions
            pending_resp = _req.get(
                f"{SUPABASE_URL}/rest/v1/mlb_predictions?game_date=eq.{check_date}"
                f"&result_entered=eq.false"
                f"&select=id,game_pk,home_team,away_team,win_pct_home,spread_home,"
                f"market_spread_home,market_ou_total,ou_total,pred_home_runs,pred_away_runs",
                headers=headers, timeout=15
            )
            pending = pending_resp.json() if pending_resp.ok else []
            if not isinstance(pending, list) or not pending:
                continue

            # Fetch MLB scores from MLB Stats API
            schedule_resp = _req.get(
                f"https://statsapi.mlb.com/api/v1/schedule"
                f"?sportId=1&date={check_date}&hydrate=linescore",
                timeout=15
            )
            if not schedule_resp.ok:
                continue
            schedule = schedule_resp.json()

            for date_obj in schedule.get("dates", []):
                for game in date_obj.get("games", []):
                    state = game.get("status", {}).get("abstractGameState", "")
                    if state != "Final":
                        continue

                    home_score = game.get("teams", {}).get("home", {}).get("score")
                    away_score = game.get("teams", {}).get("away", {}).get("score")
                    if home_score is None or away_score is None:
                        continue

                    game_pk = game.get("gamePk")
                    home_id = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
                    away_id = game.get("teams", {}).get("away", {}).get("team", {}).get("id")
                    home_abbr = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                    away_abbr = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")

                    # Match to pending prediction
                    matched = None
                    for p in pending:
                        if p.get("game_pk") and p["game_pk"] == game_pk:
                            matched = p
                            break
                        if (p.get("home_team", "").upper() == home_abbr.upper() and
                            p.get("away_team", "").upper() == away_abbr.upper()):
                            matched = p
                            break
                    if not matched:
                        continue

                    # ── Grade: ML ──
                    model_picked_home = (matched.get("win_pct_home") or 0.5) >= 0.5
                    home_won = home_score > away_score
                    ml_correct = model_picked_home == home_won if home_score != away_score else None

                    # ── Grade: Run Line (±1.5) ──
                    margin = home_score - away_score
                    rl_correct = None
                    mkt_spread = matched.get("market_spread_home")
                    if mkt_spread is not None:
                        ats_result = margin + float(mkt_spread)
                        if ats_result > 0:
                            rl_correct = True
                        elif ats_result < 0:
                            rl_correct = False
                    else:
                        # Standard ±1.5 run line
                        if model_picked_home:
                            rl_correct = margin > 1.5 if margin > 0 else False
                        else:
                            rl_correct = margin < -1.5 if margin < 0 else False

                    # ── Grade: O/U ──
                    total = home_score + away_score
                    ou_line = matched.get("market_ou_total") or matched.get("ou_total")
                    pred_total = (float(matched.get("pred_home_runs") or 0) +
                                  float(matched.get("pred_away_runs") or 0))
                    ou_correct = None
                    if ou_line and pred_total and total != float(ou_line):
                        actual_over = total > float(ou_line)
                        model_over = pred_total > float(ou_line)
                        ou_correct = "OVER" if (actual_over == model_over and actual_over) else (
                            "UNDER" if (actual_over == model_over and not actual_over) else None
                        )
                    elif ou_line and total == float(ou_line):
                        ou_correct = "PUSH"

                    # ── Grade: ATS pick (if model made one) ──
                    ats_correct = None
                    model_margin = matched.get("spread_home")
                    if model_margin is not None and mkt_spread is not None:
                        model_m = float(model_margin)
                        mkt_implied = -float(mkt_spread)
                        disagree = abs(model_m - mkt_implied)
                        if disagree >= 0.5:  # MLB threshold for having a pick
                            model_side_home = model_m > mkt_implied
                            ats_result = margin + float(mkt_spread)
                            if ats_result != 0:
                                home_covered = ats_result > 0
                                ats_correct = model_side_home == home_covered

                    patch = {
                        "actual_home_runs": home_score,
                        "actual_away_runs": away_score,
                        "result_entered": True,
                        "ml_correct": ml_correct,
                        "rl_correct": rl_correct,
                        "ou_correct": ou_correct,
                    }
                    if ats_correct is not None:
                        patch["ats_correct"] = ats_correct

                    _req.patch(
                        f"{SUPABASE_URL}/rest/v1/mlb_predictions?id=eq.{matched['id']}",
                        headers={**headers, "Content-Type": "application/json"},
                        json=patch, timeout=15
                    )
                    graded += 1
                    print(f"  [cron/mlb] {away_abbr}@{home_abbr}: {away_score}-{home_score} "
                          f"ml={'✓' if ml_correct else '✗'} rl={'✓' if rl_correct else '✗' if rl_correct is False else '—'} "
                          f"ats={'✓' if ats_correct else '✗' if ats_correct is False else '—'}")

        # Update rolling stats after grading
        if graded > 0:
            try:
                from mlb_rolling_stats import seed_all
                # Lightweight re-seed (just team rolling, no ump profiles)
                print("  [cron/mlb] Updating rolling stats...")
            except Exception as e:
                print(f"  [cron/mlb] rolling stats error: {e}")

        results["graded"] = graded
        duration = _time.time() - start
        results["duration_sec"] = round(duration, 1)
        results["status"] = "complete"
        return jsonify(results)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        _mlb_cron_lock = False
