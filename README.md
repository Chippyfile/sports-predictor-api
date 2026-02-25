# Multi-Sport Predictor API

Python backend with **scikit-learn** models and **SHAP** explanations for the Multi-Sport Predictor v14.

---

## 🚀 Deploy to Railway (Step-by-Step)

### 1. Push this folder to GitHub

```bash
# In this folder:
git init
git add .
git commit -m "Initial sports predictor API"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/sports-predictor-api.git
git push -u origin main
```

### 2. Create Railway project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Choose **"Deploy from GitHub repo"**
3. Select `sports-predictor-api`
4. Railway auto-detects Python and installs requirements

### 3. Set environment variables

In Railway → your project → **Variables** tab, add:

| Variable | Value |
|----------|-------|
| `SUPABASE_URL` | `https://lxaaqtqvlwjvyuedyauo.supabase.co` |
| `SUPABASE_ANON_KEY` | your anon key from Supabase dashboard |

Railway sets `PORT` automatically — don't add it manually.

### 4. Get your API URL

Railway gives you a URL like:  
`https://sports-predictor-api-production.up.railway.app`

---

## 📡 API Endpoints

### Train all models
```
POST /train/all
```
Pulls your historical data from Supabase and trains models for all 5 sports.

### Train one sport
```
POST /train/mlb
POST /train/nba
POST /train/ncaa
POST /train/nfl
POST /train/ncaaf
```

### Get ML prediction + SHAP explanation
```
POST /predict/mlb
Content-Type: application/json

{
  "pred_home_runs": 4.91,
  "pred_away_runs": 4.80,
  "win_pct_home": 0.5086,
  "ou_total": 9.7,
  "model_ml_home": -104
}
```

**Response:**
```json
{
  "sport": "MLB",
  "ml_margin": 0.42,
  "ml_win_prob_home": 0.5231,
  "ml_win_prob_away": 0.4769,
  "shap": [
    { "feature": "run_diff_pred", "shap": 0.38, "value": 0.11 },
    { "feature": "win_pct_home",  "shap": 0.12, "value": 0.5086 },
    ...
  ],
  "model_meta": {
    "n_train": 45,
    "mae_cv": 2.14,
    "trained_at": "2026-02-24T18:00:00"
  }
}
```

### Run Monte Carlo simulation
```
POST /monte-carlo
Content-Type: application/json

{
  "sport": "MLB",
  "home_mean": 4.91,
  "away_mean": 4.80,
  "n_sims": 10000
}
```

### Check model accuracy from your historical data
```
GET /accuracy/all
GET /accuracy/mlb
GET /accuracy/nba
```

### Health check
```
GET /health
```

---

## 🔗 Connecting to your React App (App.jsx)

Add this helper to your App.jsx:

```javascript
const ML_API = "https://YOUR-RAILWAY-URL.up.railway.app";

async function mlPredict(sport, gameData) {
  try {
    const res = await fetch(`${ML_API}/predict/${sport.toLowerCase()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(gameData),
    });
    return await res.json();
  } catch (e) {
    console.warn("ML API unavailable, using heuristic fallback:", e);
    return null;
  }
}

async function runMonteCarlo(sport, homeMean, awayMean) {
  try {
    const res = await fetch(`${ML_API}/monte-carlo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport, home_mean: homeMean, away_mean: awayMean }),
    });
    return await res.json();
  } catch (e) {
    return null;
  }
}
```

Then in your MLB prediction function, after calculating your heuristic result:

```javascript
// After your existing prediction logic:
const mlResult = await mlPredict("mlb", {
  pred_home_runs: pred.homeRuns,
  pred_away_runs: pred.awayRuns,
  win_pct_home:   pred.homeWinPct,
  ou_total:       game.ouLine,
  model_ml_home:  pred.homeML,
});

if (mlResult && !mlResult.error) {
  pred.mlWinProbHome = mlResult.ml_win_prob_home;
  pred.mlMargin      = mlResult.ml_margin;
  pred.shapFactors   = mlResult.shap;  // Use this to show "Why this pick"
}
```

---

## 📊 SHAP Feature Explanations

The `shap` array in every prediction response tells you WHY the model made its pick:

- **Positive shap** → pushed prediction toward home team winning
- **Negative shap** → pushed prediction toward away team winning
- **Larger magnitude** → more influential factor

Display the top 3-4 SHAP factors in your UI as "Key Drivers" for each prediction.

---

## 🔁 Retraining

Models are retrained by calling `POST /train/all`. Best practice:

- Retrain **weekly** during active seasons (models improve as more results come in)
- Minimum data needed: **10 completed games** per sport
- With 50+ games, accuracy improves significantly
- With 200+ games, you have a genuinely robust model

---

## 📈 What the Models Do

| Sport | Algorithm | Key Features |
|-------|-----------|-------------|
| MLB | Ridge Regression + Logistic | run predictions, FIP proxy, O/U gap |
| NBA | Gradient Boosting + Logistic | net rating diff, score predictions, O/U gap |
| NCAAB | Gradient Boosting + Logistic | adj efficiency margin diff, neutral site |
| NFL | Gradient Boosting + Logistic | EPA diff, spread vs market, O/U gap |
| NCAAF | Gradient Boosting + Logistic | adj EM diff, rankings, neutral site |

Each sport trains two models:
1. **Regressor** — predicts the margin (home score - away score)
2. **Classifier** — predicts win probability (calibrated, not raw logistic)
