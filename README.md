# Mental Health Score Predictor

Predicts a student's **Mental Health Score (0–10)** from social media habits, sleep, study time, physical activity, and self-reported stress. Trained on 5,000 student records, served through a FastAPI endpoint, and wired to a browser front end.

**Live API:** https://mental-health-score-predictor-4.onrender.com · **Docs:** `/docs`

> **Not a diagnostic tool.** The target is a self-reported survey number, not a clinical measurement. The model finds correlations in one dataset — it cannot assess, screen, or advise any real person.

---

## What it does

A student fills in thirteen fields about their daily rhythm. The API runs them through a saved scikit-learn `Pipeline` and returns a single score between roughly 3.6 and 9.4, which the front end renders on a gauge with a band label (`strained` / `balanced` / `strong`).

The point of the project is the seam between the notebook and the server: preprocessing lives *inside* the pickled pipeline, so `main.py` never touches a scaler or an encoder. It loads one file and calls `.predict()` on raw input.

---

## Stack

| Layer | Tools |
|---|---|
| EDA & training | pandas, numpy, matplotlib, seaborn, scikit-learn |
| Model | `RandomForestRegressor` inside a `Pipeline` + `ColumnTransformer` |
| Serialization | joblib → `Mental_Health_Model.pkl` |
| API | FastAPI + Pydantic validation, CORS open |
| Front end | Vanilla HTML / CSS / JS, animated SVG gauge |
| Hosting | Render (API) |

---

## Repo layout

```
.
├── ML_Project.ipynb        # EDA → cleaning → feature engineering → training → export
├── ML Project.html         # rendered notebook
├── Mental_Health_Model.pkl # fitted Pipeline (preprocessing + Random Forest)
├── Student Social Media And Mental Health Impact.csv
├── main.py                 # FastAPI app: StudentData schema + /predict
├── index.html              # form UI
├── style.css               # design tokens, layout, gauge styling
├── script.js               # validation, fetch, gauge animation
└── requirements.txt
```

---

## Dataset

5,000 rows × 13 columns. Target `Mental_Health_Score` ranges 3.6 – 9.4.

| Group | Columns |
|---|---|
| Demographics | `Age`, `Gender`, `Country` (111 unique) , `Academic_Level` |
| Platform use | `Most_Used_Platform`, `Purpose_Of_Use`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks` |
| Lifestyle | `Study_Hours`, `Sleep_Hours_Per_Night`, `Physical_Activity_Hours` |
| Self-report | `Stress_Level` (Low → Very High) |

**Cleaning:** duplicates dropped; `Physical_Activity_Hours` had a negative minimum (impossible), clipped at 0 rather than dropping the row — the rest of that student's record is still valid.

**Feature engineering:** `Country` has 111 categories. One-hot encoding it adds 110 near-empty columns; dropping it discards real signal (internet access, sleep norms, culture). Compromise: keep the top 10 countries, bucket the rest into `Other` as `Grouped_Country` — 111 categories down to 11.

---

## Preprocessing

One `ColumnTransformer`, four branches, fitted only on the training split:

| Branch | Columns | Treatment |
|---|---|---|
| Skewed | `Study_Hours` | `log1p` → `StandardScaler` |
| Numeric | `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` | `StandardScaler` |
| Ordinal | `Stress_Level` | `OrdinalEncoder(['Low','Medium','High','Very High'])` — the order is real, EDA shows the score stepping down |
| Nominal | `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_Country` | `OneHotEncoder(handle_unknown='ignore')` |

`handle_unknown='ignore'` is what keeps an unseen platform or country from crashing the live endpoint.

---

## Results

70/30 split, `random_state=42`, evaluated on the held-out test set.

| Model | Train R² | Test R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression (baseline) | 0.724 | 0.740 | 0.536 | 0.676 |
| **Random Forest (default) — shipped** | 0.981 | **0.878** | **0.347** | **0.463** |
| Random Forest (tuned) | 0.955 | 0.865 | 0.369 | 0.487 |

Tuning used `RandomizedSearchCV` (15 draws, 5-fold, scored on R²); best params were `n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2`.

The default forest won on test R², so it is the one serialized. Note the train/test gap: 0.98 → 0.88 means an unconstrained forest is memorizing the training set. The tuned model gives up 0.013 of test R² for a much narrower gap (0.95 → 0.87) and would likely be the safer choice on new data.

The linear baseline explaining 74% is the number that makes the forest's 88% meaningful. Without it, 0.88 is a figure with nothing to be better than.

---

## API

### `GET /`
Health check.

### `POST /predict`

All fields required. Ranges are enforced by Pydantic; violations return `422`.

| Field | Type | Constraint |
|---|---|---|
| `age` | int | 10–100 |
| `gender` | str | `Male`, `Female` |
| `country` | str | free text; anything outside the top 10 is bucketed as `Other` |
| `academic_level` | str | `High School`, `Undergraduate`, `Graduate` |
| `most_used_platform` | str | Facebook, Instagram, Snapchat, Twitter, YouTube, TikTok, LinkedIn, LINE, KakaoTalk, VKontakte, WhatsApp, WeChat |
| `purpose_of_use` | str | `Networking`, `Education`, `Entertainment`, `News` |
| `avg_daily_usage_hours` | float | 0–24 |
| `daily_unlocks` | int | ≥ 0 |
| `study_hours` | float | 0–24 |
| `physical_activity_hours` | float | 0–24 |
| `sleep_hours_per_night` | float | 0–24 |
| `stress_level` | str | `Low`, `Medium`, `High`, `Very High` |

```bash
curl -X POST https://mental-health-score-predictor-r5t7.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 21, "gender": "Male", "country": "India",
    "academic_level": "Undergraduate", "most_used_platform": "Instagram",
    "purpose_of_use": "Entertainment", "avg_daily_usage_hours": 4.0,
    "daily_unlocks": 100, "study_hours": 4.0,
    "physical_activity_hours": 1.0, "sleep_hours_per_night": 6.5,
    "stress_level": "High"
  }'
```

```json
{ "predicted_mental_health_score": 6.43 }
```

---

## Run locally

```bash
git clone https://github.com/cyraemin/Mental-Health-Score-Predictor.git
cd Mental-Health-Score-Predictor

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload
```

API at `http://127.0.0.1:8000`, interactive docs at `/docs`.

For the front end, set `API_BASE` in `script.js` to `http://127.0.0.1:8000` and open `index.html` with any static server (`python -m http.server`). Opening the file directly over `file://` will fail CORS.

To retrain, run `ML_Project.ipynb` top to bottom — the last cell writes `Mental_Health_Model.pkl`.

---

## Limitations

- **Correlational, not causal.** More sleep is associated with a higher score. The model cannot tell you that sleeping more would *raise* your score — the arrow may run the other way, or through stress, which is measured here too.
- **`Stress_Level` does a lot of work.** It is a self-report collected alongside the target, so it likely shares whatever mood or framing effects the target has. A version of this model without it would be a fairer test of whether the *behavioural* features predict anything.
- **Distribution-bound.** Trained on one student survey. Accuracy outside that population is unmeasured.
- **Point estimate, no interval.** The UI shows `6.43` to two decimals, which reads far more precise than an MAE of ±0.35 justifies.

---

## Roadmap

- [ ] Prediction intervals instead of a bare point estimate
- [ ] Ablation without `Stress_Level` to isolate behavioural signal
- [ ] Permutation importance / SHAP for feature attribution
- [ ] Pin dependency versions and add a smoke test that loads the pickle and predicts one row
- [ ] Rate limiting and a narrower CORS policy before any real traffic

---

## Credits

Built following [Building & Deploying a Mental Health Score Predictor](https://youtu.be/WHTbyKPrcPg) by Sheryians AI School. Dataset: student social media and mental health impact survey.

## License

MIT
