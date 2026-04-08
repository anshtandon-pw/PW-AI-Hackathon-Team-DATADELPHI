from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_metrics(scored: pd.DataFrame, split_name: str) -> dict[str, float | str | int]:
    mae_pct = float(np.mean(np.abs(scored["recommended_discount_pct"] - scored["proxy_required_discount_pct"])))
    bucket_accuracy = float(np.mean(scored["recommended_discount_bucket"] == scored["proxy_target_bucket"]))
    conservative_rate = float(np.mean(scored["recommended_discount_bucket"] <= scored["proxy_target_bucket"]))
    avg_recommended = float(scored["recommended_discount_bucket"].mean())
    avg_reference = float(scored["reference_discount_pct"].mean())
    avg_gap = float(scored["reference_discount_gap_pct"].mean())
    return {
        "split": split_name,
        "rows": int(len(scored)),
        "mae_pct": round(mae_pct, 3),
        "bucket_accuracy": round(bucket_accuracy, 4),
        "conservative_rate": round(conservative_rate, 4),
        "avg_recommended_bucket": round(avg_recommended, 3),
        "avg_reference_discount_pct": round(avg_reference, 3),
        "avg_gap_pct": round(avg_gap, 3),
    }


def build_category_dashboard(scored_2026: pd.DataFrame) -> pd.DataFrame:
    dashboard = (
        scored_2026.groupby("exam_fin", dropna=False)
        .agg(
            users=("userid", "size"),
            avg_reference_discount_pct=("reference_discount_pct", "mean"),
            avg_recommended_discount_pct=("recommended_discount_bucket", "mean"),
            avg_confidence=("confidence_score", "mean"),
            avg_proof_confidence=("proof_confidence_score", "mean"),
            total_expected_avoidable_amount=("expected_avoidable_discount_amount", "sum"),
            total_conservative_avoidable_amount=("conservative_avoidable_discount_amount", "sum"),
            total_upper_avoidable_amount=("upper_avoidable_discount_amount", "sum"),
            organic_share=("has_organic_history", "mean"),
            high_discount_share=("is_high_discount_history", "mean"),
            low_history_share=("is_low_history", "mean"),
        )
        .reset_index()
        .sort_values("total_expected_avoidable_amount", ascending=False)
    )
    return dashboard


def build_waste_analysis(scored_2025: pd.DataFrame) -> pd.DataFrame:
    analysis = (
        scored_2025.groupby("proof_label", dropna=False)
        .agg(
            users=("userid", "size"),
            avg_reference_discount_pct=("reference_discount_pct", "mean"),
            avg_recommended_discount_pct=("recommended_discount_bucket", "mean"),
            conservative_avoidable_amount=("conservative_avoidable_discount_amount", "sum"),
            expected_avoidable_amount=("expected_avoidable_discount_amount", "sum"),
            upper_avoidable_amount=("upper_avoidable_discount_amount", "sum"),
        )
        .reset_index()
        .sort_values("expected_avoidable_amount", ascending=False)
    )
    return analysis


def build_top_user_samples(scored_2026: pd.DataFrame, sample_size: int = 50) -> pd.DataFrame:
    columns = [
        "userid",
        "exam_fin",
        "persona",
        "reason_codes",
        "proof_label",
        "reference_discount_pct",
        "recommended_discount_bucket",
        "confidence_score",
        "proof_confidence_score",
        "expected_avoidable_discount_amount",
        "organic_order_count",
        "coupon_order_count",
        "avg_order_value",
    ]
    return (
        scored_2026.sort_values(["expected_avoidable_discount_amount", "proof_confidence_score"], ascending=[False, False])
        .loc[:, columns]
        .head(sample_size)
        .reset_index(drop=True)
    )


def _currency(value: float) -> str:
    return f"₹{value:,.0f}"


def build_html_report(
    output_path: str | Path,
    metadata: dict[str, object],
    split_metrics_df: pd.DataFrame,
    waste_analysis_df: pd.DataFrame,
    category_dashboard_df: pd.DataFrame,
    sample_users_df: pd.DataFrame,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary_cards = [
        ("Rows Scored (2026)", f"{metadata['test_rows']:,}"),
        ("Categories", f"{metadata['categories']:,}"),
        ("Expected Avoidable (2026)", _currency(float(metadata["test_expected_avoidable"]))),
        ("Conservative Avoidable (2026)", _currency(float(metadata["test_conservative_avoidable"]))),
        ("Avg Recommended Bucket", f"{float(metadata['avg_recommended_bucket']):.1f}%"),
    ]

    cards_html = "".join(
        f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div></div>"
        for label, value in summary_cards
    )

    notes = [
        "This engine is built on a purchaser-only lifetime snapshot and should be interpreted as a revenue optimization policy model, not a full causal uplift model.",
        "Proof labels are confidence-backed estimates based on category backtesting, organic purchase evidence, historical minimum discount logic, and score confidence.",
        "A future 5-10% exploration group is still recommended to create stronger causal evidence in live traffic.",
    ]
    notes_html = "".join(f"<li>{note}</li>" for note in notes)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Smart Coupon Engine Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1c2630; background: #f5f7fa; }}
    h1, h2 {{ color: #102a43; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0 28px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(16,42,67,0.12); }}
    .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #627d98; margin-bottom: 8px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .section {{ background: white; border-radius: 12px; padding: 18px; margin: 18px 0; box-shadow: 0 1px 4px rgba(16,42,67,0.12); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e6ecf1; }}
    th {{ background: #f0f4f8; }}
    .muted {{ color: #627d98; }}
  </style>
</head>
<body>
  <h1>Smart Coupon Engine</h1>
  <p class="muted">Category-aware discount recommendation and proof-confidence report generated from the available raw user snapshot.</p>
  <div class="cards">{cards_html}</div>

  <div class="section">
    <h2>Validation Notes</h2>
    <ul>{notes_html}</ul>
  </div>

  <div class="section">
    <h2>Split Metrics</h2>
    {split_metrics_df.to_html(index=False, border=0)}
  </div>

  <div class="section">
    <h2>2025 Waste Analysis</h2>
    {waste_analysis_df.to_html(index=False, border=0)}
  </div>

  <div class="section">
    <h2>2026 Category Dashboard</h2>
    {category_dashboard_df.head(25).to_html(index=False, border=0)}
  </div>

  <div class="section">
    <h2>Top User Recommendations</h2>
    {sample_users_df.to_html(index=False, border=0)}
  </div>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def write_metadata(path: str | Path, metadata: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
