# Smart Coupon Engine

This workspace contains a self-contained implementation of a category-aware coupon recommendation engine built for the available `data/raw.csv` snapshot.

## What It Does

- Cleans and profiles the raw user-level coupon history data
- Builds category calibration tables for `exam_fin`
- Trains a lightweight hybrid policy model using only `pandas` and `numpy`
- Recommends the minimum discount bucket per user
- Estimates unnecessary discount spend with proof-confidence bands
- Generates CSV outputs and a static HTML report for demo/review

## Important Data Limitation

The current dataset is a user-level lifetime snapshot of purchasers, not an exposure-level table with non-converters. Because of that:

- the engine estimates likely unnecessary discount and minimum effective discount policy
- it does **not** claim perfect causal proof
- the report explicitly marks outputs as confidence-backed estimates

## Run

```powershell
python run_pipeline.py
```

Optional arguments:

```powershell
python run_pipeline.py --input data/raw.csv --output-dir outputs --artifacts-dir artifacts
```

## Main Outputs

- `artifacts/model.pkl`
- `artifacts/category_calibration.csv`
- `outputs/recommendations_2026.csv`
- `outputs/category_dashboard.csv`
- `outputs/waste_analysis_2025.csv`
- `outputs/report.html`

## Demo UI/API

Run the local demo server:

```powershell
python demo_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Useful API endpoints:

- `/api/health`
- `/api/overview`
- `/api/options`
- `/api/categories`
- `/api/users`
- `/api/user/<userid>`

## New Sample Training Drop

If you are sharing sampled cohort-style order history instead of full yearly raw files, place them here:

- `data/sample_training/bought_in_all_three_years_excludes_2026.csv`
- `data/sample_training/bought_in_2025_and_2026_not_2024_excludes_2026.csv`

Run the sample-training pipeline with:

```powershell
python run_sample_training_pipeline.py
```

Main outputs:

- `outputs/sample_user_profiles.csv`
- `outputs/sample_coupon_recommendations.csv`
- `outputs/sample_category_cluster_policy.csv`
- `outputs/sample_cluster_summary.csv`
- `outputs/sample_data_quality.csv`
- `outputs/sample_prediction_metrics.csv`

## Sample Demo UI/API

After running the sample-training pipeline, start the local sample demo with:

```powershell
python sample_demo_app.py
```

Then open:

```text
http://127.0.0.1:8010
```

Useful sample API endpoints:

- `/api/sample/health`
- `/api/sample/overview`
- `/api/sample/options`
- `/api/sample/policies`
- `/api/sample/users`
- `/api/sample/user/<userid>`

The sample demo now shows the predicted 2026 discount bucket, the historical reference bucket, the expected saving if 2026 follows historical behavior, and an exact `userid` lookup flow. If you later add a real 2026 order file to `data/sample_training`, the same screen will also compare the predicted discount against the actual 2026 discount.
