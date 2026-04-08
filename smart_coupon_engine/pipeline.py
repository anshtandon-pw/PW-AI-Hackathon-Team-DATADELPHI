from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR
from .data import load_raw_data, split_by_time
from .features import add_category_features, build_category_calibration, derive_proxy_required_discount_pct
from .model import SmartCouponModel
from .reporting import (
    build_category_dashboard,
    build_html_report,
    build_top_user_samples,
    build_waste_analysis,
    split_metrics,
    write_metadata,
)


def _ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def run_pipeline(input_path: Path, output_dir: Path, artifacts_dir: Path) -> dict[str, object]:
    _ensure_directories(output_dir, artifacts_dir)

    df = load_raw_data(input_path)
    df["proxy_required_discount_pct"] = derive_proxy_required_discount_pct(df)

    train_df, validation_df, test_df = split_by_time(df)

    calibration = build_category_calibration(train_df)
    train_df = add_category_features(train_df, calibration)
    validation_df = add_category_features(validation_df, calibration)
    test_df = add_category_features(test_df, calibration)

    model = SmartCouponModel.fit(train_df)

    train_scored = model.score(train_df)
    validation_scored = model.score(validation_df)
    test_scored = model.score(test_df)
    all_scored = pd.concat([train_scored, validation_scored, test_scored], ignore_index=True)

    model_path = artifacts_dir / "model.pkl"
    model.save(model_path)
    calibration.to_csv(artifacts_dir / "category_calibration.csv", index=False)

    split_metrics_df = pd.DataFrame(
        [
            split_metrics(train_scored, "train"),
            split_metrics(validation_scored, "validation"),
            split_metrics(test_scored, "test_2026"),
        ]
    )

    active_2025 = all_scored[all_scored["last_order_year"] == 2025].copy()
    waste_analysis_df = build_waste_analysis(active_2025)
    category_dashboard_df = build_category_dashboard(test_scored)
    sample_users_df = build_top_user_samples(test_scored)

    recommendation_columns = [
        "userid",
        "exam_fin",
        "persona",
        "reason_codes",
        "recommended_discount_pct",
        "recommended_discount_bucket",
        "reference_discount_pct",
        "default_policy_pct",
        "confidence_score",
        "proof_confidence_score",
        "proof_label",
        "expected_unnecessary_discount_amount",
        "conservative_avoidable_discount_amount",
        "upper_avoidable_discount_amount",
        "fallback_used",
    ]
    test_scored.loc[:, recommendation_columns].to_csv(output_dir / "recommendations_2026.csv", index=False)
    category_dashboard_df.to_csv(output_dir / "category_dashboard.csv", index=False)
    waste_analysis_df.to_csv(output_dir / "waste_analysis_2025.csv", index=False)
    split_metrics_df.to_csv(output_dir / "split_metrics.csv", index=False)
    sample_users_df.to_csv(output_dir / "top_user_samples.csv", index=False)

    metadata = {
        "input_path": str(input_path),
        "rows_total": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "categories": int(df["exam_fin"].nunique()),
        "constant_columns": df.attrs.get("constant_columns", []),
        "test_expected_avoidable": float(test_scored["expected_avoidable_discount_amount"].sum()),
        "test_conservative_avoidable": float(test_scored["conservative_avoidable_discount_amount"].sum()),
        "avg_recommended_bucket": float(test_scored["recommended_discount_bucket"].mean()),
    }
    write_metadata(artifacts_dir / "run_metadata.json", metadata)
    build_html_report(
        output_dir / "report.html",
        metadata,
        split_metrics_df,
        waste_analysis_df,
        category_dashboard_df,
        sample_users_df,
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Smart Coupon Engine pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args(argv)

    metadata = run_pipeline(args.input, args.output_dir, args.artifacts_dir)
    print("Pipeline completed.")
    print(f"Rows processed: {metadata['rows_total']:,}")
    print(f"2026 rows scored: {metadata['test_rows']:,}")
    print(f"Expected avoidable amount (2026 cohort): INR {metadata['test_expected_avoidable']:,.0f}")
    return 0
