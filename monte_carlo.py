import numpy as np
from scipy.stats import nbinom

def _negbin_draw(rng, mu, k, n):
    """
    Draw n samples from NegBin(μ, k).
    Parameterization: mean=μ, variance=μ + μ²/k
    scipy.stats.nbinom uses (r=k, p=k/(k+μ))
    """
    mu  = max(float(mu), 0.5)
    p   = k / (k + mu)
    return rng.negative_binomial(int(round(k * 10)) / 10, p, n).astype(float)
    # Note: scipy parameterization for integer r only; use numpy directly:
    # numpy's negative_binomial(n, p) where n=k, p=k/(k+μ)

def _negbin_draw_numpy(rng, mu, k, size):
    """
    Numpy NegBin draw.  numpy uses n=number of successes (k), p=success prob.
    NegBin(μ, k): p = k / (k + μ)
    """
    mu = max(float(mu), 0.5)
    k  = max(float(k), 0.10)
    p  = k / (k + mu)
    # numpy negative_binomial: r (int or float via workaround)
    # Use gamma-Poisson mixture for non-integer k (more accurate):
    #   λ ~ Gamma(k, μ/k)   then   X ~ Poisson(λ)
    lam = rng.gamma(shape=k, scale=mu / k, size=size)
    return rng.poisson(lam).astype(float)

def monte_carlo(sport, home_mean, away_mean, n_sims=10_000, ou_line=None, game_id=None):
    """
    Run score simulations and return outcome distribution.

    MLB (v2):
      - Negative Binomial draws (overdispersed vs Poisson)
      - Correlated run environment via shared log-normal multiplier
      - Returns run-line cover %, over/under % at posted total

    Others:
      - Normal distribution (unchanged, appropriate for high-scoring sports)
      - Returns spread cover %, over/under % at posted total

    FIX (Finding #17): Seed is now derived from game_id so each game gets
    unique random draws. Fixed seed=42 caused identical simulations for
    any two games with the same run means, wasting 10k draws of signal.
    """
    if game_id is not None:
        seed = hash(str(game_id)) % (2**32)
    else:
        # No game_id provided — use time-based seed for uniqueness
        seed = int(datetime.utcnow().timestamp() * 1000) % (2**32)
    rng = np.random.default_rng(seed)

    if sport == "MLB":
        k_home, k_away = _get_mlb_k()

        # ── Shared run environment (correlation) ─────────────────────────
        # σ_env=0.12 means ~±12% game-level run environment shift.
        # This models factors that affect both teams: umpire strike zone,
        # wind, temperature, park conditions on the day.
        # Validated: MLB game total correlation ≈ 0.10–0.15 between teams.
        sigma_env = 0.12
        env_factor = rng.lognormal(mean=0.0, sigma=sigma_env, size=n_sims)

        home_mean_adj = np.maximum(home_mean * env_factor, 0.5)
        away_mean_adj = np.maximum(away_mean * env_factor, 0.5)

        # ── Negative Binomial draws via Gamma-Poisson mixture ────────────
        # FIX (Finding #15): Removed duplicate scalar-mean draw that was
        # immediately overwritten. That dead code wasted ~40k RNG calls,
        # shifting all subsequent draws to different sequence positions.
        # Only the per-sim correlated draw below is correct — it uses each
        # simulation's individually adjusted mean from the shared env_factor,
        # preserving the home/away run correlation within each simulated game.
        home_lam = rng.gamma(shape=k_home, scale=home_mean_adj / k_home, size=n_sims)
        away_lam = rng.gamma(shape=k_away, scale=away_mean_adj / k_away, size=n_sims)
        home_scores = rng.poisson(home_lam).astype(float)
        away_scores = rng.poisson(away_lam).astype(float)

        distribution_note = (
            f"Negative Binomial (k_home={k_home:.3f}, k_away={k_away:.3f}) "
            f"with correlated run environment (σ={sigma_env})"
        )

    else:
        # F5 FIX: Calibrated base_std per empirical D1 data
        base_std = {"NBA": 11.0, "NCAAB": 10.8, "NFL": 10.5, "NCAAF": 14.0}.get(sport, 10.0)

        # Finding 20: Dynamic σ based on expected game tempo
        avg_total = home_mean + away_mean
        tempo_norm = {"NBA": 220, "NCAAB": 140, "NFL": 44, "NCAAF": 52}.get(sport, 140)
        tempo_factor = avg_total / tempo_norm if tempo_norm > 0 else 1.0
        # F5 FIX: Widened tempo bounds from [0.75,1.25] to [0.80,1.30]
        std = base_std * max(0.80, min(1.30, tempo_factor))

        # Finding 21: Shared pace/environment correlation for basketball
        if sport in ("NBA", "NCAAB"):
            sigma_pace = 0.08  # ~±8% shared pace variance
            pace_factor = rng.lognormal(mean=0.0, sigma=sigma_pace, size=n_sims)
            home_scores = rng.normal(home_mean * pace_factor, std, n_sims)
            away_scores = rng.normal(away_mean * pace_factor, std, n_sims)
            distribution_note = f"Normal(σ={std:.1f}) with pace correlation (σ_pace={sigma_pace})"
        else:
            home_scores = rng.normal(home_mean, std, n_sims)
            away_scores = rng.normal(away_mean, std, n_sims)
            distribution_note = f"Normal(σ={std:.1f})"

    margins = home_scores - away_scores
    totals  = home_scores + away_scores

    # ── Run line / Spread cover probabilities ─────────────────────────────
    # Standard run line for MLB is -1.5 / +1.5
    rl_threshold = 1.5 if sport == "MLB" else 0.5
    home_rl_cover = float((margins > rl_threshold).mean())   # home -1.5 cover
    away_rl_cover = float((margins < -rl_threshold).mean())  # away +1.5 cover

    # ── Over/Under probabilities ──────────────────────────────────────────
    if ou_line is not None:
        over_pct  = float((totals > ou_line).mean())
        under_pct = float((totals < ou_line).mean())
        push_ou   = float((totals == ou_line).mean())
    else:
        # Use the simulation mean as a rough posted line if not provided
        sim_total = float(totals.mean())
        over_pct  = float((totals > sim_total).mean())
        under_pct = float((totals < sim_total).mean())
        push_ou   = float((totals == sim_total).mean())
        ou_line   = round(sim_total, 1)

    return {
        "n_sims":           n_sims,
        "distribution":     distribution_note,

        # Moneyline
        "home_win_pct":     round(float((margins > 0).mean()), 4),
        "away_win_pct":     round(float((margins < 0).mean()), 4),
        "push_pct":         round(float((margins == 0).mean()), 4),

        # Run line / ATS
        "home_rl_cover_pct": round(home_rl_cover, 4),
        "away_rl_cover_pct": round(away_rl_cover, 4),
        "rl_threshold":      rl_threshold,

        # Over/Under
        "ou_line":           ou_line,
        "over_pct":          round(over_pct, 4),
        "under_pct":         round(under_pct, 4),
        "push_ou_pct":       round(push_ou, 4),

        # Score distribution
        "avg_margin":        round(float(margins.mean()), 2),
        "avg_total":         round(float(totals.mean()), 2),
        "std_margin":        round(float(margins.std()), 2),
        "std_total":         round(float(totals.std()), 2),

        "margin_percentiles": {
            "p5":  round(float(np.percentile(margins, 5)),  1),
            "p10": round(float(np.percentile(margins, 10)), 1),
            "p25": round(float(np.percentile(margins, 25)), 1),
            "p50": round(float(np.percentile(margins, 50)), 1),
            "p75": round(float(np.percentile(margins, 75)), 1),
            "p90": round(float(np.percentile(margins, 90)), 1),
            "p95": round(float(np.percentile(margins, 95)), 1),
        },
        "total_percentiles": {
            "p10": round(float(np.percentile(totals, 10)), 1),
            "p25": round(float(np.percentile(totals, 25)), 1),
            "p50": round(float(np.percentile(totals, 50)), 1),
            "p75": round(float(np.percentile(totals, 75)), 1),
            "p90": round(float(np.percentile(totals, 90)), 1),
        },
        "histogram": _histogram(margins, bins=20),
    }

def _histogram(arr, bins=20):
    counts, edges = np.histogram(arr, bins=bins)
    return [
        {"bin": round(float((edges[i] + edges[i+1]) / 2), 1), "count": int(counts[i])}
        for i in range(len(counts))
    ]

# ═══════════════════════════════════════════════════════════════
# MODEL ACCURACY REPORT
# ═══════════════════════════════════════════════════════════════
