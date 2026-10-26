# MultipleLR 3Datasets — Five Regression Models

Three independent Kaggle regression datasets, five regression estimators, F-test feature selection and RMSE/R² evaluation. Flask interactive dashboard with a dataset dropdown, a five-model dropdown, feature count slider, holdout diagnostics, and 95% split-conformal prediction intervals.

## Five models

- Multiple Linear Regression (OLS) — `LinearRegression()`
- Ridge Regression — `Ridge(alpha=1.0)`
- Lasso Regression — `Lasso(alpha=0.1)`
- Elastic Net Regression — `ElasticNet(alpha=0.1,l1_ratio=0.5)`
- Bayesian Ridge Regression — `BayesianRidge()`

Unlike the previous project, these five estimators **all fit and predict**; Lasso and ElasticNet are not merely ranking tools. Feature selection is performed by **F-test on the fit partition**. Hyperparameters are fixed for instructional comparisons, not tuned to test results.

## Data

| Dataset | Original candidate features | Target | Source |
|---|---:|---|---|
| Appliances Energy | 28 | Appliances | https://www.kaggle.com/competitions/machine-learning-sb1 |
| Parkinsons Telemonitoring | 19 | total_UPDRS | https://www.kaggle.com/datasets/leilahasan/parkinson-dataset |
| Student Performance Math | 32 | G3 | https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set |

Appliances contains two random control features (`rv1`, `rv2`). Parkinsons excludes the subject ID and `motor_UPDRS`. Student's categoricals are one-hot encoded: *k* counts encoded features, not original fields.

## Setup (Windows CMD)

```bat
cd /d D:\AIOTHW2\MultipleLR_3Datasets_Independent\MultipleLR_3Datasets
python -m pip install -r requirements.txt
python download_data.py
python train_all.py
python app.py
```

Open http://127.0.0.1:5000

The 3 JSON results are saved as `results/appliances.json`, `results/parkinsons.json`, `results/student.json`.

## Export 30 line charts

```bat
python export_30_model_charts.py
```

Output: `line_charts_30/`, exactly **3 datasets × 5 regression models × 2 metrics = 30 PNG images**. X-axis values are:

- Appliances: 6, 12, 17, 22, 27
- Parkinsons: 4, 8, 12, 16, 19
- Student: 6, 12, 18, 25, 32

## CRISP-DM

1. **Business Understanding:** energy, health score, and student performance prediction.
2. **Data Understanding:** review feature types, labels, missing values and sample counts.
3. **Data Preparation:** independent data splits, fit-only preprocessing, F-test ranking.
4. **Modeling:** fit five regression estimators for each k.
5. **Evaluation:** RMSE, R², MAE, residuals, prediction intervals and empirical coverage.
6. **Deployment:** Flask dashboard + Gunicorn/Render; cache precomputed JSON results.

## Prediction intervals and limitations

Intervals use split-conformal calibration (`alpha=0.05`) with held-out calibration residuals. Their marginal nominal coverage requires exchangeability of the calibration and evaluation observations, which time drift can violate. They are *prediction* intervals, not confidence intervals of a conditional mean. For time-based or grouped datasets, results should be interpreted with relevant dependence caveats.

## Render deployment

Run **training locally** before uploading to GitHub. The `data/` directory is ignored, but `results/*.json` must be committed to GitHub; without the CSV or cached results, Render cannot train the project on request.

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

To update:

```bat
git add .
git commit -m "Replace selection methods with five regression models"
git push
```

If the results JSON files are accidentally gitignored, adjust `.gitignore` and re-run git add.
