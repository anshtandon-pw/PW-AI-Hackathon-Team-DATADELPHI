from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT


SAMPLE_TRAINING_DIR = PROJECT_ROOT / "data" / "sample_training"
SAMPLE_OUTPUT_PREFIX = "sample"
RECOMMENDATION_BUCKETS = [0, 5, 10, 15, 20, 25, 30]
SAMPLE_REFERENCE_DATE = pd.Timestamp("2025-12-31")
SAMPLE_MODEL_SAMPLE_SIZE = 120_000
SAMPLE_RIDGE_ALPHA = 2.0
CLUSTER_NAMES = [
    "discount_independent",
    "mildly_sensitive",
    "highly_discount_dependent",
]


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _bucketize_pct(values: pd.Series | np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    buckets = np.empty(values_array.shape[0], dtype=float)
    for index, value in enumerate(values_array):
        chosen = RECOMMENDATION_BUCKETS[-1]
        for bucket in RECOMMENDATION_BUCKETS:
            if value <= bucket:
                chosen = bucket
                break
        buckets[index] = chosen
    return buckets


def _bucket_rank(value: float | int) -> int:
    numeric = int(round(float(value)))
    if numeric in RECOMMENDATION_BUCKETS:
        return RECOMMENDATION_BUCKETS.index(numeric)
    for index, bucket in enumerate(RECOMMENDATION_BUCKETS):
        if numeric <= bucket:
            return index
    return len(RECOMMENDATION_BUCKETS) - 1


def _shift_bucket(value: float | int, steps: int) -> int:
    rank = _bucket_rank(value)
    shifted = min(max(rank + steps, 0), len(RECOMMENDATION_BUCKETS) - 1)
    return RECOMMENDATION_BUCKETS[shifted]


def _support_threshold(total_orders: int) -> tuple[float, int]:
    if total_orders <= 1:
        return 0.35, 1
    if total_orders == 2:
        return 0.50, 1
    if total_orders == 3:
        return 0.55, 2
    return 0.60, 2


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.2816) -> float:
    if trials <= 0:
        return 0.0
    p_hat = successes / trials
    denominator = 1 + (z * z / trials)
    center = p_hat + (z * z / (2 * trials))
    margin = z * math.sqrt((p_hat * (1 - p_hat) / trials) + (z * z / (4 * trials * trials)))
    return max(0.0, (center - margin) / denominator)


def _binomial_one_sided_pvalue(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    probability = 0.0
    for value in range(k, n + 1):
        probability += math.comb(n, value) * (p**value) * ((1 - p) ** (n - value))
    return min(1.0, float(probability))


def _regression_slope(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(x) != len(y):
        return 0.0
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    centered_x = x_array - x_array.mean()
    denominator = float(np.sum(centered_x**2))
    if denominator == 0:
        return 0.0
    return float(np.sum(centered_x * (y_array - y_array.mean())) / denominator)


def _compute_history_reference(group: pd.DataFrame) -> dict[str, object]:
    ordered = group.sort_values(["order_date", "rn"], kind="stable").copy()
    buckets = _bucketize_pct(ordered["coupon_discount_pct"]).astype(int)
    total_orders = len(ordered)
    organic_orders = int((buckets == 0).sum())
    coupon_orders = int((buckets > 0).sum())
    recency_weights = np.linspace(1.0, 1.25, total_orders) if total_orders > 1 else np.array([1.0])
    weight_bonus = np.where(buckets == 0, 1.25, np.where(buckets <= 5, 0.15, 0.0))
    weights = recency_weights + weight_bonus
    total_weight = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
    recent_mask = np.zeros(total_orders, dtype=bool)
    recent_mask[-min(2, total_orders) :] = True
    required_share, required_orders = _support_threshold(total_orders)

    exact_bucket_counts = {bucket: int((buckets == bucket).sum()) for bucket in RECOMMENDATION_BUCKETS}
    cumulative_support_by_bucket: list[float] = []
    support_rows: list[dict[str, float]] = []
    for bucket in RECOMMENDATION_BUCKETS:
        at_or_below = buckets <= bucket
        exact = buckets == bucket
        support_weight = float(weights[at_or_below].sum())
        support_share = support_weight / total_weight
        order_count = int(at_or_below.sum())
        recent_count = int((at_or_below & recent_mask).sum())
        exact_count = int(exact.sum())
        support_lower_bound = _wilson_lower_bound(order_count, total_orders)
        support_score = support_share
        if order_count >= 2:
            support_score += 0.15
        if recent_count >= 1:
            support_score += 0.05
        if bucket == 0 and organic_orders >= 1:
            support_score += 0.10
        cumulative_support_by_bucket.append(support_share)
        support_rows.append(
            {
                "bucket": bucket,
                "support_share": support_share,
                "support_lower_bound": support_lower_bound,
                "support_score": support_score,
                "order_count": order_count,
                "recent_count": recent_count,
                "exact_count": exact_count,
            }
        )

    reference_bucket = None
    reference_support_share = 0.0
    reference_support_lower_bound = 0.0
    reference_support_score = 0.0
    orders_at_or_below_reference = 0
    for row in support_rows:
        if row["support_lower_bound"] >= required_share and row["order_count"] >= required_orders:
            reference_bucket = int(row["bucket"])
            reference_support_share = float(row["support_share"])
            reference_support_lower_bound = float(row["support_lower_bound"])
            reference_support_score = float(row["support_score"])
            orders_at_or_below_reference = int(row["order_count"])
            break

    if reference_bucket is None:
        fallback = next((row for row in support_rows if row["support_share"] >= required_share), None)
        if fallback is None:
            reference_bucket = int(buckets.min()) if total_orders else 0
            fallback = next((row for row in support_rows if int(row["bucket"]) == reference_bucket), support_rows[0])
        else:
            reference_bucket = int(fallback["bucket"])
        reference_support_share = float(fallback["support_share"])
        reference_support_lower_bound = float(fallback["support_lower_bound"])
        reference_support_score = float(fallback["support_score"])
        orders_at_or_below_reference = int(fallback["order_count"])

    positive_buckets = buckets[buckets > 0]
    unique_positive_buckets = np.unique(positive_buckets) if len(positive_buckets) else np.array([], dtype=int)
    bias_adjustment_steps = 0
    if reference_bucket >= 10 and organic_orders == 0 and len(unique_positive_buckets) <= 1:
        if coupon_orders == 1:
            reference_bucket = _shift_bucket(reference_bucket, -1)
            bias_adjustment_steps = -1
        elif coupon_orders == 2 and reference_bucket >= 15:
            reference_bucket = _shift_bucket(reference_bucket, -1)
            bias_adjustment_steps = -1

    recent_buckets = buckets[-min(2, total_orders) :]
    recent_reference_bucket = int(min(recent_buckets)) if len(recent_buckets) else reference_bucket
    low_discount_share = float(np.mean(buckets <= 5))
    high_discount_share = float(np.mean(buckets >= 10))
    organic_share = float(np.mean(buckets == 0))
    coupon_share = float(np.mean(buckets > 0))
    cumulative_support_5 = next((row["support_share"] for row in support_rows if int(row["bucket"]) == 5), low_discount_share)
    cumulative_support_10 = next((row["support_share"] for row in support_rows if int(row["bucket"]) == 10), low_discount_share)
    cumulative_support_20 = next((row["support_share"] for row in support_rows if int(row["bucket"]) == 20), 1.0)
    incremental_support_10_over_5 = float(max(0.0, cumulative_support_10 - cumulative_support_5))
    incremental_support_20_over_10 = float(max(0.0, cumulative_support_20 - cumulative_support_10))
    low_count_5 = int(sum(count for bucket, count in exact_bucket_counts.items() if bucket <= 5))
    increment_count_10 = int(sum(count for bucket, count in exact_bucket_counts.items() if 5 < bucket <= 10))
    low_count_10 = int(sum(count for bucket, count in exact_bucket_counts.items() if bucket <= 10))
    increment_count_20 = int(sum(count for bucket, count in exact_bucket_counts.items() if 10 < bucket <= 20))
    pvalue_10_over_5 = (
        _binomial_one_sided_pvalue(increment_count_10, low_count_5 + increment_count_10)
        if increment_count_10 > low_count_5
        else 1.0
    )
    pvalue_20_over_10 = (
        _binomial_one_sided_pvalue(increment_count_20, low_count_10 + increment_count_20)
        if increment_count_20 > low_count_10
        else 1.0
    )
    significant_10_over_5 = int((incremental_support_10_over_5 >= 0.20) and (pvalue_10_over_5 <= 0.10))
    significant_20_over_10 = int((incremental_support_20_over_10 >= 0.20) and (pvalue_20_over_10 <= 0.10))
    regression_slope = _regression_slope(list(range(len(RECOMMENDATION_BUCKETS))), cumulative_support_by_bucket)
    dependency_score = (
        45 * coupon_share
        + 20 * high_discount_share
        + 10 * int(reference_bucket >= 10)
        + 10 * int(reference_bucket >= 15)
        + 5 * int(len(unique_positive_buckets) == 1 and len(unique_positive_buckets) > 0 and unique_positive_buckets[0] >= 10)
        + 12 * significant_10_over_5
        + 15 * significant_20_over_10
        + 8 * np.clip(regression_slope / 0.12, 0, 1)
        - 35 * organic_share
        - 10 * low_discount_share
        - (20 if total_orders == 1 else 10 if total_orders == 2 else 0)
    )
    dependency_score = float(np.clip(dependency_score, 0, 100))
    if (
        (reference_bucket <= 5 and organic_share >= 0.40 and not significant_10_over_5 and not significant_20_over_10)
        or (reference_bucket == 0 and (organic_orders >= 1 or coupon_orders == 0))
        or dependency_score < 25
    ):
        sensitivity = "discount_independent"
    elif (
        reference_bucket >= 10
        and organic_share <= 0.20
        and ((coupon_share >= 0.75 and coupon_orders >= 2) or significant_10_over_5 or significant_20_over_10 or dependency_score >= 60)
    ):
        sensitivity = "highly_discount_dependent"
    else:
        sensitivity = "mildly_sensitive"
    if total_orders == 1 and reference_bucket >= 10:
        sensitivity = "mildly_sensitive"
    if significant_10_over_5 or significant_20_over_10:
        hypothesis_label = "reject_h0_discount_matters_proxy"
    elif organic_share >= 0.50 or reference_bucket == 0:
        hypothesis_label = "fail_to_reject_h0_discount_not_material_proxy"
    else:
        hypothesis_label = "insufficient_evidence_purchase_only_data"

    return {
        "historical_reference_discount_pct": float(reference_bucket),
        "historical_reference_bucket_pct": float(reference_bucket),
        "historical_reference_support_share": float(reference_support_share),
        "historical_reference_support_lower_bound": float(reference_support_lower_bound),
        "historical_reference_support_score": float(reference_support_score),
        "orders_at_or_below_reference": int(orders_at_or_below_reference),
        "recent_observed_discount_bucket_pct": float(recent_reference_bucket),
        "observed_min_discount_bucket_pct": float(int(buckets.min()) if total_orders else 0),
        "organic_order_share": organic_share,
        "low_discount_order_share": low_discount_share,
        "high_discount_order_share": high_discount_share,
        "incremental_support_10_over_5": incremental_support_10_over_5,
        "incremental_support_20_over_10": incremental_support_20_over_10,
        "pvalue_10_over_5": float(pvalue_10_over_5),
        "pvalue_20_over_10": float(pvalue_20_over_10),
        "significant_10_over_5": significant_10_over_5,
        "significant_20_over_10": significant_20_over_10,
        "discount_response_slope": float(regression_slope),
        "discount_dependency_score": dependency_score,
        "discount_sensitivity_class": sensitivity,
        "discount_effect_hypothesis": hypothesis_label,
        "reference_bias_adjustment_steps": int(bias_adjustment_steps),
    }


def _classify_source_file(file_name: str) -> str:
    upper = file_name.upper()
    if "ALL THE THREE YEAR" in upper:
        return "repeat_buyer_with_2024_history"
    if "2025 AND 2026 BUT NOT IN 2024" in upper:
        return "future_2026_buyer_without_2024_history"
    return "unknown_source_group"


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%d-%b-%y", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, format="%d %b, %Y", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _load_order_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"userid": "string", "coupon_code": "string"})
    df.columns = [column.strip() for column in df.columns]
    df["userid"] = df["userid"].astype("string").str.strip()
    df["exam_fin"] = df["exam_fin"].astype("string").str.strip()
    df["lead_type"] = df["lead_type"].astype("string").str.strip().str.title()
    df["order_type"] = df["order_type"].astype("string").str.strip()
    df["coupon_code"] = df["coupon_code"].fillna("").astype("string").str.strip()
    df["order_date"] = _parse_dates(df["order_date"])
    df["signupdate"] = _parse_dates(df["signupdate"])
    df["order_year"] = _clean_numeric(df["order_year"]).astype("Int64")
    df["item_price"] = _clean_numeric(df["item_price"]).astype("float32")
    df["item_coupon_discount"] = _clean_numeric(df["item_coupon_discount"]).astype("float32")
    df["rn"] = _clean_numeric(df["rn"]).astype("Int64")
    df["is_coupon_order"] = (
        df["order_type"].str.lower().eq("coupon_order")
        | (df["item_coupon_discount"].fillna(0) > 0)
        | (df["coupon_code"] != "")
    ).astype("int8")
    df["net_paid"] = (df["item_price"].fillna(0) - df["item_coupon_discount"].fillna(0)).clip(lower=0).astype("float32")
    df["coupon_discount_pct"] = np.where(
        df["item_price"].fillna(0) > 0,
        (df["item_coupon_discount"].fillna(0) / df["item_price"].replace(0, np.nan)) * 100,
        0,
    )
    df["coupon_discount_pct"] = np.nan_to_num(df["coupon_discount_pct"], nan=0.0).astype("float32")
    df["signup_to_order_days"] = (df["order_date"] - df["signupdate"]).dt.days.astype("float32")
    df["negative_signup_lag"] = (df["signup_to_order_days"] < 0).astype("int8")
    return df


def load_sample_training_data(folder: Path) -> pd.DataFrame:
    files = sorted(path for path in folder.glob("*.csv") if "EXCLUDE" in path.name.upper())
    if not files:
        raise FileNotFoundError(f"No sample-training CSV files found in {folder}")
    frames: list[pd.DataFrame] = []
    for path in files:
        df = _load_order_file(path)
        df["source_group"] = _classify_source_file(path.name)
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_actual_2026_data(folder: Path) -> pd.DataFrame | None:
    candidates = [
        path for path in folder.glob("*.csv") if ("2026" in path.name.upper()) and ("EXCLUDE" not in path.name.upper())
    ]
    if not candidates:
        return None
    actual = pd.concat([_load_order_file(path) for path in sorted(candidates)], ignore_index=True)
    actual = actual[actual["order_year"] == 2026].copy()
    return None if actual.empty else actual


def _most_common_or_last(group: pd.DataFrame, column: str) -> str:
    mode = group[column].mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(group[column].iloc[-1])


def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["userid", "order_date", "rn"], kind="stable").copy()
    ordered["prev_order_date"] = ordered.groupby("userid")["order_date"].shift(1)
    ordered["days_since_prev_order"] = (ordered["order_date"] - ordered["prev_order_date"]).dt.days
    profiles: list[dict[str, object]] = []
    for user_id, group in ordered.groupby("userid", sort=False):
        group = group.sort_values(["order_date", "rn"], kind="stable")
        first = group.iloc[0]
        last = group.iloc[-1]
        coupon_rows = group[group["is_coupon_order"] == 1]
        organic_rows = group[group["is_coupon_order"] == 0]
        rows_2025 = group[group["order_year"] == 2025]
        coupon_rows_2025 = rows_2025[rows_2025["is_coupon_order"] == 1]
        positive_coupon_pct = coupon_rows["coupon_discount_pct"][coupon_rows["coupon_discount_pct"] > 0]
        first_coupon_date = coupon_rows["order_date"].min() if not coupon_rows.empty else pd.NaT
        organic_before_first_coupon = bool(
            not organic_rows.empty and pd.notna(first_coupon_date) and (organic_rows["order_date"] < first_coupon_date).any()
        )
        history_reference = _compute_history_reference(group)
        profiles.append(
            {
                "userid": str(user_id),
                "source_group": _most_common_or_last(group, "source_group"),
                "lead_type": _most_common_or_last(group, "lead_type"),
                "primary_exam_fin": _most_common_or_last(group, "exam_fin"),
                "last_exam_fin": str(last["exam_fin"]),
                "exam_fin_count": int(group["exam_fin"].nunique()),
                "signup_date": first["signupdate"],
                "first_order_date": first["order_date"],
                "last_order_date": last["order_date"],
                "signup_to_first_order_days": float((first["order_date"] - first["signupdate"]).days)
                if pd.notna(first["order_date"]) and pd.notna(first["signupdate"])
                else np.nan,
                "customer_age_days_asof_2025_end": float((SAMPLE_REFERENCE_DATE - first["signupdate"]).days)
                if pd.notna(first["signupdate"])
                else np.nan,
                "days_since_last_order_asof_2025_end": float((SAMPLE_REFERENCE_DATE - last["order_date"]).days)
                if pd.notna(last["order_date"])
                else np.nan,
                "total_orders": int(len(group)),
                "orders_2024": int((group["order_year"] == 2024).sum()),
                "orders_2025": int((group["order_year"] == 2025).sum()),
                "coupon_orders_2025": int(coupon_rows_2025.shape[0]),
                "organic_orders_2025": int((rows_2025["is_coupon_order"] == 0).sum()),
                "coupon_share_2025": float(coupon_rows_2025.shape[0] / len(rows_2025)) if len(rows_2025) else 0.0,
                "active_years": int(group["order_year"].dropna().nunique()),
                "total_gmv": float(group["item_price"].sum()),
                "total_discount": float(group["item_coupon_discount"].sum()),
                "total_net_revenue": float(group["net_paid"].sum()),
                "avg_order_value": float(group["item_price"].mean()),
                "recent_avg_order_value": float(group["item_price"].tail(2).mean()),
                "avg_net_paid": float(group["net_paid"].mean()),
                "coupon_orders": int(group["is_coupon_order"].sum()),
                "organic_orders": int((group["is_coupon_order"] == 0).sum()),
                "coupon_share": float(group["is_coupon_order"].mean()),
                "first_order_coupon_flag": int(first["is_coupon_order"]),
                "last_order_coupon_flag": int(last["is_coupon_order"]),
                "avg_coupon_discount_pct": float(positive_coupon_pct.mean()) if not positive_coupon_pct.empty else 0.0,
                "min_coupon_discount_pct": float(positive_coupon_pct.min()) if not positive_coupon_pct.empty else 0.0,
                "max_coupon_discount_pct": float(positive_coupon_pct.max()) if not positive_coupon_pct.empty else 0.0,
                "last_coupon_discount_pct": float(positive_coupon_pct.iloc[-1]) if not positive_coupon_pct.empty else 0.0,
                "recent_coupon_discount_pct": float(positive_coupon_pct.tail(2).mean()) if not positive_coupon_pct.empty else 0.0,
                "avg_coupon_discount_amount": float(coupon_rows["item_coupon_discount"].mean()) if not coupon_rows.empty else 0.0,
                "unique_coupon_codes": int(coupon_rows["coupon_code"].replace("", pd.NA).dropna().nunique()),
                "avg_days_between_orders": float(group["days_since_prev_order"].dropna().mean())
                if group["days_since_prev_order"].notna().any()
                else np.nan,
                "organic_before_first_coupon": int(organic_before_first_coupon),
                "has_2024_history": int((group["order_year"] == 2024).any()),
                "has_2025_history": int((group["order_year"] == 2025).any()),
                "negative_signup_lag_seen": int(group["negative_signup_lag"].max()),
                "paid_lead_flag": int(_most_common_or_last(group, "lead_type") == "Paid"),
                **history_reference,
            }
        )
    profiles_df = pd.DataFrame(profiles)
    profiles_df["avg_days_between_orders"] = profiles_df["avg_days_between_orders"].fillna(
        (profiles_df["last_order_date"] - profiles_df["first_order_date"]).dt.days
    )
    profiles_df["discount_pct_of_gmv"] = np.where(
        profiles_df["total_gmv"] > 0, (profiles_df["total_discount"] / profiles_df["total_gmv"]) * 100, 0
    )
    profiles_df["has_organic_history"] = (profiles_df["organic_orders"] > 0).astype("int8")
    profiles_df["repeat_buyer_flag"] = (profiles_df["total_orders"] > 1).astype("int8")
    profiles_df["high_coupon_reliance"] = (
        (profiles_df["coupon_share"] >= 0.80) & (profiles_df["coupon_orders"] >= 1)
    ).astype("int8")
    profiles_df["strong_organic_signal"] = (
        (profiles_df["organic_orders"] >= 2)
        | ((profiles_df["organic_orders"] >= 1) & (profiles_df["first_order_coupon_flag"] == 0))
    ).astype("int8")
    profiles_df["reactivation_risk_flag"] = (
        (profiles_df["days_since_last_order_asof_2025_end"] >= 120) & (profiles_df["total_orders"] >= 1)
    ).astype("int8")
    profiles_df["consistent_reference_flag"] = (
        (profiles_df["historical_reference_support_lower_bound"] >= 0.45)
        & (profiles_df["orders_at_or_below_reference"] >= np.where(profiles_df["total_orders"] <= 2, 1, 2))
    ).astype("int8")
    profiles_df["recent_lower_than_reference_flag"] = (
        profiles_df["recent_observed_discount_bucket_pct"] < profiles_df["historical_reference_bucket_pct"]
    ).astype("int8")
    return profiles_df


def assign_coupon_cluster(profiles: pd.DataFrame) -> pd.DataFrame:
    clusters: list[str] = []
    reasons: list[str] = []
    for _, row in profiles.iterrows():
        reason_codes: list[str] = []
        cluster = str(row["discount_sensitivity_class"])
        if cluster == "discount_independent":
            reason_codes.append("organic_or_low_discount_history")
        elif cluster == "mildly_sensitive":
            reason_codes.append("mixed_conversion_behavior")
        else:
            reason_codes.append("coupon_dependent_pattern")
        if row["historical_reference_bucket_pct"] == 0:
            reason_codes.append("converts_without_coupon")
        else:
            reason_codes.append(f"reference_{int(row['historical_reference_bucket_pct'])}pct")
        if row["consistent_reference_flag"] == 1:
            reason_codes.append("repeated_reference_pattern")
        if row["organic_before_first_coupon"] == 1:
            reason_codes.append("organic_before_coupon")
        if row["reference_bias_adjustment_steps"] < 0:
            reason_codes.append("exposure_bias_adjusted_down")
        if row["recent_lower_than_reference_flag"] == 1:
            reason_codes.append("recent_lower_discount_seen")
        if row["reactivation_risk_flag"] == 1:
            reason_codes.append("reactivation_gap")
        if row["negative_signup_lag_seen"] == 1:
            reason_codes.append("signup_lag_issue")
        if row["paid_lead_flag"] == 1:
            reason_codes.append("paid_lead")
        clusters.append(cluster)
        reasons.append("|".join(reason_codes))
    clustered = profiles.copy()
    clustered["coupon_cluster"] = pd.Series(clusters, index=profiles.index, dtype="string")
    clustered["cluster_reason_codes"] = pd.Series(reasons, index=profiles.index, dtype="string")
    split_hash = clustered["userid"].apply(lambda value: int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16) % 100)
    clustered["prediction_split"] = np.where(split_hash < 80, "train", "validation")
    return clustered


def build_category_cluster_policy(clustered: pd.DataFrame) -> pd.DataFrame:
    policy = (
        clustered.groupby(["primary_exam_fin", "coupon_cluster"], dropna=False)
        .agg(
            policy_users=("userid", "size"),
            policy_avg_order_value=("avg_order_value", "median"),
            policy_coupon_share=("coupon_share", "mean"),
            policy_organic_share=("has_organic_history", "mean"),
            policy_reference_pct=("historical_reference_discount_pct", "median"),
            policy_reference_support=("historical_reference_support_share", "median"),
            policy_min_coupon_discount_pct=("min_coupon_discount_pct", "median"),
            policy_avg_coupon_discount_pct=("avg_coupon_discount_pct", "median"),
            policy_recent_coupon_discount_pct=("recent_observed_discount_bucket_pct", "median"),
            policy_paid_share=("paid_lead_flag", "mean"),
        )
        .reset_index()
    )
    baseline = policy["policy_reference_pct"].fillna(0).to_numpy(dtype=float)
    baseline = np.where(
        policy["coupon_cluster"] == "discount_independent",
        np.where(policy["policy_organic_share"] >= 0.40, 0, np.minimum(baseline, 5)),
        baseline,
    )
    baseline = np.where(
        policy["coupon_cluster"] == "mildly_sensitive",
        np.clip(baseline, 5, 10),
        baseline,
    )
    baseline = np.where(
        policy["coupon_cluster"] == "highly_discount_dependent",
        np.maximum(baseline, np.maximum(policy["policy_min_coupon_discount_pct"].fillna(0), 10)),
        baseline,
    )
    floor = np.where(
        policy["coupon_cluster"] == "discount_independent",
        0,
        np.where(
            policy["coupon_cluster"] == "mildly_sensitive",
            np.minimum(policy["policy_reference_pct"].fillna(5), 5),
            np.maximum(np.minimum(policy["policy_reference_pct"].fillna(10), 10), 5),
        ),
    )
    cap = np.where(
        policy["coupon_cluster"] == "discount_independent",
        5,
        np.where(
            policy["coupon_cluster"] == "mildly_sensitive",
            np.maximum(policy["policy_reference_pct"].fillna(5) + 5, 10),
            np.maximum(policy["policy_reference_pct"].fillna(10) + 5, 15),
        ),
    )
    policy["recommended_base_pct"] = np.clip(baseline, 0, 30)
    policy["recommended_base_bucket"] = _bucketize_pct(policy["recommended_base_pct"])
    policy["policy_floor_pct"] = np.clip(floor, 0, 30)
    policy["policy_cap_pct"] = np.clip(np.maximum(cap, policy["policy_floor_pct"]), 0, 30)
    return policy


def add_policy_features(clustered: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    merged = clustered.merge(policy, on=["primary_exam_fin", "coupon_cluster"], how="left")
    defaults = {
        "policy_users": 0,
        "policy_avg_order_value": merged["avg_order_value"].median(),
        "policy_coupon_share": merged["coupon_share"].median(),
        "policy_organic_share": merged["has_organic_history"].median(),
        "policy_reference_pct": merged["historical_reference_discount_pct"].median(),
        "policy_reference_support": merged["historical_reference_support_share"].median(),
        "policy_min_coupon_discount_pct": merged["min_coupon_discount_pct"].median(),
        "policy_avg_coupon_discount_pct": merged["avg_coupon_discount_pct"].median(),
        "policy_recent_coupon_discount_pct": merged["recent_observed_discount_bucket_pct"].median(),
        "policy_paid_share": merged["paid_lead_flag"].median(),
        "recommended_base_pct": merged["historical_reference_discount_pct"].median(),
        "recommended_base_bucket": 5,
        "policy_floor_pct": 0,
        "policy_cap_pct": 30,
    }
    for column, default_value in defaults.items():
        merged[column] = merged[column].fillna(default_value)
    merged["recommended_base_pct"] = merged["recommended_base_pct"].clip(lower=0, upper=30)
    merged["policy_floor_pct"] = merged["policy_floor_pct"].clip(lower=0, upper=30)
    merged["policy_cap_pct"] = merged["policy_cap_pct"].clip(lower=0, upper=30)
    merged["policy_cap_pct"] = np.maximum(merged["policy_cap_pct"], merged["policy_floor_pct"])
    return merged


def derive_proxy_2026_discount_pct(merged: pd.DataFrame) -> pd.Series:
    reference = merged["historical_reference_discount_pct"].fillna(0).astype(float)
    recent_anchor = merged["recent_observed_discount_bucket_pct"].fillna(reference).astype(float)
    target = reference.copy()
    target = np.where(
        merged["discount_sensitivity_class"] == "discount_independent",
        np.where(merged["strong_organic_signal"] == 1, 0, np.minimum(target, 5)),
        target,
    )
    target = np.where(
        (merged["discount_sensitivity_class"] == "mildly_sensitive") & (merged["recent_lower_than_reference_flag"] == 1),
        np.minimum(target, recent_anchor),
        target,
    )
    target = np.where(
        (merged["discount_sensitivity_class"] == "mildly_sensitive") & (merged["historical_reference_support_share"] < 0.60),
        np.minimum(target, np.maximum(0, reference - 5)),
        target,
    )
    target = np.where(
        (merged["discount_sensitivity_class"] == "highly_discount_dependent")
        & (merged["consistent_reference_flag"] == 1)
        & (merged["coupon_share_2025"] >= 0.80)
        & (merged["recent_observed_discount_bucket_pct"] > merged["historical_reference_bucket_pct"])
        & (merged["organic_orders"] == 0),
        np.maximum(target, merged["recent_observed_discount_bucket_pct"]),
        target,
    )
    target = np.where(
        (merged["reactivation_risk_flag"] == 1)
        & (merged["discount_sensitivity_class"] == "highly_discount_dependent")
        & (merged["historical_reference_support_share"] >= 0.65),
        np.minimum(np.maximum(target, reference + 5), merged["policy_cap_pct"]),
        target,
    )
    target = np.where(
        (merged["paid_lead_flag"] == 1)
        & (merged["discount_sensitivity_class"] == "mildly_sensitive")
        & (merged["total_orders"] == 1)
        & (merged["historical_reference_bucket_pct"] == 0),
        5,
        target,
    )
    target = np.where(
        (merged["organic_before_first_coupon"] == 1) & (merged["historical_reference_bucket_pct"] >= 5),
        np.minimum(target, merged["historical_reference_bucket_pct"]),
        target,
    )
    target = np.clip(target, merged["policy_floor_pct"], merged["policy_cap_pct"])
    return pd.Series(np.clip(target, 0, 30), index=merged.index, dtype="float32")


def build_prediction_feature_frame(merged: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "signup_to_first_order_days",
        "customer_age_days_asof_2025_end",
        "days_since_last_order_asof_2025_end",
        "total_orders",
        "orders_2024",
        "orders_2025",
        "coupon_orders",
        "organic_orders",
        "coupon_orders_2025",
        "organic_orders_2025",
        "active_years",
        "total_gmv",
        "total_discount",
        "total_net_revenue",
        "avg_order_value",
        "recent_avg_order_value",
        "avg_net_paid",
        "coupon_share",
        "coupon_share_2025",
        "avg_coupon_discount_pct",
        "min_coupon_discount_pct",
        "max_coupon_discount_pct",
        "last_coupon_discount_pct",
        "recent_coupon_discount_pct",
        "historical_reference_discount_pct",
        "historical_reference_support_share",
        "historical_reference_support_lower_bound",
        "historical_reference_support_score",
        "orders_at_or_below_reference",
        "recent_observed_discount_bucket_pct",
        "observed_min_discount_bucket_pct",
        "organic_order_share",
        "low_discount_order_share",
        "high_discount_order_share",
        "incremental_support_10_over_5",
        "incremental_support_20_over_10",
        "pvalue_10_over_5",
        "pvalue_20_over_10",
        "significant_10_over_5",
        "significant_20_over_10",
        "discount_response_slope",
        "discount_dependency_score",
        "consistent_reference_flag",
        "recent_lower_than_reference_flag",
        "avg_coupon_discount_amount",
        "unique_coupon_codes",
        "avg_days_between_orders",
        "discount_pct_of_gmv",
        "paid_lead_flag",
        "has_organic_history",
        "repeat_buyer_flag",
        "high_coupon_reliance",
        "strong_organic_signal",
        "organic_before_first_coupon",
        "reactivation_risk_flag",
        "policy_users",
        "policy_avg_order_value",
        "policy_coupon_share",
        "policy_organic_share",
        "policy_reference_pct",
        "policy_reference_support",
        "policy_min_coupon_discount_pct",
        "policy_avg_coupon_discount_pct",
        "policy_recent_coupon_discount_pct",
        "policy_paid_share",
        "recommended_base_pct",
        "policy_floor_pct",
        "policy_cap_pct",
    ]
    frame = merged.loc[:, numeric_columns].fillna(0).astype(float).copy()
    frame["log_total_gmv"] = np.log1p(frame["total_gmv"])
    frame["log_total_discount"] = np.log1p(frame["total_discount"])
    frame["log_avg_order_value"] = np.log1p(frame["avg_order_value"])
    frame["log_days_since_last_order"] = np.log1p(frame["days_since_last_order_asof_2025_end"].clip(lower=0))
    frame["log_customer_age_days"] = np.log1p(frame["customer_age_days_asof_2025_end"].clip(lower=0))
    for cluster_name in CLUSTER_NAMES:
        frame[f"cluster__{cluster_name}"] = (merged["coupon_cluster"] == cluster_name).astype(float)
    return frame


@dataclass
class SampleDiscountModel:
    feature_names: list[str]
    means: dict[str, float]
    scales: dict[str, float]
    coefficients: dict[str, float]
    intercept: float

    @classmethod
    def fit(cls, features: pd.DataFrame, target: pd.Series) -> "SampleDiscountModel":
        sampled = features.sample(n=min(len(features), SAMPLE_MODEL_SAMPLE_SIZE), random_state=42)
        sampled_target = target.loc[sampled.index].fillna(0).astype("float64")
        x = sampled.fillna(0).astype("float64")
        means = x.mean().to_dict()
        scales = x.std().replace(0, 1).to_dict()
        standardized = (x - pd.Series(means)) / pd.Series(scales)
        design_matrix = np.column_stack([np.ones(len(standardized)), standardized.to_numpy()])
        ridge = SAMPLE_RIDGE_ALPHA * np.eye(design_matrix.shape[1])
        ridge[0, 0] = 0
        beta = np.linalg.solve(design_matrix.T @ design_matrix + ridge, design_matrix.T @ sampled_target.to_numpy())
        return cls(
            feature_names=list(standardized.columns),
            means=means,
            scales=scales,
            coefficients={name: float(weight) for name, weight in zip(standardized.columns, beta[1:])},
            intercept=float(beta[0]),
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        design = features.loc[:, self.feature_names].fillna(0).astype("float64")
        standardized = (design - pd.Series(self.means)) / pd.Series(self.scales)
        weights = np.array([self.coefficients[name] for name in self.feature_names], dtype=float)
        return self.intercept + standardized.to_numpy() @ weights


def score_2026_discount_predictions(merged: pd.DataFrame, model: SampleDiscountModel) -> pd.DataFrame:
    scored = merged.copy()
    feature_frame = build_prediction_feature_frame(scored)
    scored["proxy_target_discount_pct"] = derive_proxy_2026_discount_pct(scored)
    raw_prediction = model.predict(feature_frame)
    history_anchor = scored["historical_reference_discount_pct"].to_numpy(dtype=float)
    recent_anchor = scored["recent_observed_discount_bucket_pct"].to_numpy(dtype=float)
    baseline = scored["recommended_base_pct"].to_numpy(dtype=float)
    confidence = (
        np.clip(scored["total_orders"] / 4, 0, 1) * 0.25
        + np.clip(scored["historical_reference_support_lower_bound"], 0, 1) * 0.20
        + np.clip(scored["organic_orders"] / 2, 0, 1) * 0.10
        + np.clip(scored["active_years"] / 2, 0, 1) * 0.15
        + np.clip(scored["policy_users"] / 300, 0, 1) * 0.20
        + (1 - np.clip(scored["exam_fin_count"] / 3, 0, 1)) * 0.10
        + (1 - scored["negative_signup_lag_seen"] * 0.20)
    )
    confidence = np.where(scored["total_orders"] <= 1, np.minimum(confidence, 0.60), confidence)
    confidence = np.clip(confidence, 0.15, 0.95)
    blended = 0.25 * np.clip(raw_prediction, 0, 30) + 0.50 * history_anchor + 0.15 * recent_anchor + 0.10 * baseline
    final_pct = confidence * blended + (1 - confidence) * baseline
    final_pct = np.where(scored["coupon_cluster"] == "discount_independent", np.minimum(final_pct, 5), final_pct)
    final_pct = np.where(
        (scored["coupon_cluster"] == "discount_independent") & (scored["historical_reference_bucket_pct"] == 0),
        0,
        final_pct,
    )
    final_pct = np.where(
        scored["coupon_cluster"] == "highly_discount_dependent",
        np.maximum(final_pct, scored["historical_reference_bucket_pct"]),
        final_pct,
    )
    final_pct = np.where(
        (scored["coupon_cluster"] == "mildly_sensitive") & (scored["recent_lower_than_reference_flag"] == 1),
        np.minimum(final_pct, scored["historical_reference_bucket_pct"]),
        final_pct,
    )
    final_pct = np.where(
        (scored["coupon_cluster"] == "mildly_sensitive") & (scored["historical_reference_support_share"] < 0.55),
        np.minimum(final_pct, np.maximum(0, scored["historical_reference_bucket_pct"] - 5)),
        final_pct,
    )
    final_pct = np.where(
        scored["discount_effect_hypothesis"] == "fail_to_reject_h0_discount_not_material_proxy",
        np.minimum(final_pct, scored["historical_reference_bucket_pct"]),
        final_pct,
    )
    final_pct = np.where(
        (scored["coupon_cluster"] == "highly_discount_dependent")
        & (scored["reactivation_risk_flag"] == 1)
        & (scored["historical_reference_support_share"] >= 0.65),
        np.maximum(final_pct, np.minimum(scored["historical_reference_bucket_pct"] + 5, scored["policy_cap_pct"])),
        final_pct,
    )
    final_pct = np.clip(final_pct, scored["policy_floor_pct"], scored["policy_cap_pct"])
    predicted_bucket = _bucketize_pct(final_pct)
    reference_pct = scored["historical_reference_discount_pct"].to_numpy(dtype=float)
    reference_bucket = _bucketize_pct(reference_pct)
    predicted_amount = scored["avg_order_value"].fillna(0).to_numpy() * predicted_bucket / 100
    reference_amount = scored["avg_order_value"].fillna(0).to_numpy() * reference_bucket / 100
    next_order_saving = np.maximum(0, reference_amount - predicted_amount)
    reason_codes: list[str] = []
    for _, row in scored.iterrows():
        tags = [str(row["discount_sensitivity_class"])]
        tags.append(f"hist_ref_{int(row['historical_reference_bucket_pct'])}pct")
        if row["historical_reference_support_share"] >= 0.60:
            tags.append("repeat_pattern")
        if row["historical_reference_support_lower_bound"] >= 0.45:
            tags.append("reference_ci_supported")
        if row["strong_organic_signal"] == 1 or row["historical_reference_bucket_pct"] == 0:
            tags.append("organic_evidence")
        if row["high_discount_order_share"] >= 0.50:
            tags.append("higher_discount_history")
        if row["significant_10_over_5"] == 1:
            tags.append("sig_10_over_5")
        if row["significant_20_over_10"] == 1:
            tags.append("sig_20_over_10")
        tags.append(str(row["discount_effect_hypothesis"]))
        if row["reactivation_risk_flag"] == 1:
            tags.append("reactivation")
        if row["paid_lead_flag"] == 1:
            tags.append("paid_lead")
        if row["reference_bias_adjustment_steps"] < 0:
            tags.append("bias_adjusted")
        reason_codes.append("|".join(tags))
    scored["predicted_2026_discount_pct_raw"] = np.clip(raw_prediction, 0, 30).astype("float32")
    scored["predicted_2026_discount_pct"] = final_pct.astype("float32")
    scored["predicted_2026_discount_bucket_pct"] = predicted_bucket.astype("float32")
    scored["historical_reference_discount_pct"] = reference_pct.astype("float32")
    scored["historical_reference_discount_bucket_pct"] = reference_bucket.astype("float32")
    scored["predicted_2026_discount_amount"] = predicted_amount.astype("float32")
    scored["historical_reference_discount_amount"] = reference_amount.astype("float32")
    scored["estimated_saving_next_order_2026"] = next_order_saving.astype("float32")
    scored["estimated_saving_if_2026_matches_history"] = (
        next_order_saving * np.maximum(scored["coupon_orders"].fillna(0).to_numpy(), 1)
    ).astype("float32")
    scored["prediction_confidence"] = confidence.astype("float32")
    scored["prediction_reason_codes"] = pd.Series(reason_codes, index=scored.index, dtype="string")
    return scored


def build_prediction_metrics(scored: pd.DataFrame, split_name: str) -> dict[str, object]:
    target_bucket = _bucketize_pct(scored["proxy_target_discount_pct"])
    predicted_bucket = scored["predicted_2026_discount_bucket_pct"].to_numpy(dtype=float)
    return {
        "split": split_name,
        "rows": int(len(scored)),
        "bucket_accuracy": float(np.mean(target_bucket == predicted_bucket)),
        "bucket_mae": float(np.mean(np.abs(target_bucket - predicted_bucket))),
        "pct_mae": float(np.mean(np.abs(scored["proxy_target_discount_pct"] - scored["predicted_2026_discount_pct"]))),
        "avg_predicted_bucket": float(np.mean(predicted_bucket)),
        "avg_target_bucket": float(np.mean(target_bucket)),
    }


def build_cluster_summary(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["coupon_cluster"], dropna=False)
        .agg(
            users=("userid", "size"),
            paid_share=("paid_lead_flag", "mean"),
            coupon_share=("coupon_share", "mean"),
            organic_share=("has_organic_history", "mean"),
            avg_prediction_confidence=("prediction_confidence", "mean"),
            avg_predicted_bucket=("predicted_2026_discount_bucket_pct", "mean"),
            estimated_saving_if_2026_matches_history=("estimated_saving_if_2026_matches_history", "sum"),
        )
        .reset_index()
        .sort_values("estimated_saving_if_2026_matches_history", ascending=False)
    )


def build_data_quality_summary(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "rows_total", "value": int(len(df))},
            {"metric": "unique_users", "value": int(df["userid"].nunique())},
            {"metric": "users_with_multiple_exam_fin", "value": int((profiles["exam_fin_count"] > 1).sum())},
            {"metric": "negative_signup_lag_rows", "value": int(df["negative_signup_lag"].sum())},
            {"metric": "negative_signup_lag_users", "value": int(profiles["negative_signup_lag_seen"].sum())},
            {"metric": "coupon_orders", "value": int(df["is_coupon_order"].sum())},
            {"metric": "non_coupon_orders", "value": int((df["is_coupon_order"] == 0).sum())},
        ]
    )


def attach_actual_2026_comparison(scored: pd.DataFrame, actual_orders: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, object]]:
    if actual_orders is None:
        enriched = scored.copy()
        for column in [
            "actual_2026_avg_order_value",
            "actual_2026_discount_pct",
            "actual_2026_discount_bucket_pct",
            "actual_2026_discount_amount",
            "saved_amount_vs_actual_2026",
        ]:
            enriched[column] = np.nan
        for column in ["actual_2026_orders", "actual_2026_coupon_orders"]:
            enriched[column] = pd.Series([pd.NA] * len(enriched), dtype="Int64")
        return enriched, {"actual_2026_available": False, "actual_2026_rows": 0, "actual_2026_users": 0}
    grouped = (
        actual_orders.groupby("userid", dropna=False)
        .agg(
            actual_2026_orders=("userid", "size"),
            actual_2026_coupon_orders=("is_coupon_order", "sum"),
            actual_2026_avg_order_value=("item_price", "mean"),
            actual_2026_total_discount=("item_coupon_discount", "sum"),
            actual_2026_total_gmv=("item_price", "sum"),
        )
        .reset_index()
    )
    grouped["actual_2026_discount_pct"] = np.where(
        grouped["actual_2026_total_gmv"] > 0,
        (grouped["actual_2026_total_discount"] / grouped["actual_2026_total_gmv"]) * 100,
        0,
    )
    grouped["actual_2026_discount_bucket_pct"] = _bucketize_pct(grouped["actual_2026_discount_pct"])
    grouped["actual_2026_discount_amount"] = (
        grouped["actual_2026_avg_order_value"] * grouped["actual_2026_discount_bucket_pct"] / 100
    )
    enriched = scored.merge(grouped, on="userid", how="left")
    comparable_predicted_amount = np.where(
        enriched["actual_2026_avg_order_value"].notna(),
        enriched["actual_2026_avg_order_value"] * enriched["predicted_2026_discount_bucket_pct"] / 100,
        enriched["predicted_2026_discount_amount"],
    )
    enriched["saved_amount_vs_actual_2026"] = np.where(
        enriched["actual_2026_discount_amount"].notna(),
        np.maximum(0, enriched["actual_2026_discount_amount"] - comparable_predicted_amount)
        * np.maximum(enriched["actual_2026_coupon_orders"].fillna(0), 1),
        np.nan,
    )
    return enriched, {
        "actual_2026_available": True,
        "actual_2026_rows": int(len(actual_orders)),
        "actual_2026_users": int(actual_orders["userid"].nunique()),
    }


def run_sample_pipeline(
    sample_dir: Path = SAMPLE_TRAINING_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, object]:
    print(f"[1/7] Loading sample training files from {sample_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    orders = load_sample_training_data(sample_dir)
    actual_2026_orders = load_actual_2026_data(sample_dir)
    print(f"Loaded {len(orders):,} rows across {orders['userid'].nunique():,} users", flush=True)
    print("[2/7] Building user profiles", flush=True)
    profiles = build_user_profiles(orders)
    print("[3/7] Assigning coupon clusters", flush=True)
    clustered = assign_coupon_cluster(profiles)
    print("[4/7] Building category-cluster policy table", flush=True)
    train_clustered = clustered[clustered["prediction_split"] == "train"].copy()
    validation_clustered = clustered[clustered["prediction_split"] == "validation"].copy()
    policy = build_category_cluster_policy(train_clustered)
    train_merged = add_policy_features(train_clustered, policy)
    validation_merged = add_policy_features(validation_clustered, policy)
    all_merged = add_policy_features(clustered, policy)
    print("[5/7] Training and scoring the 2026 discount predictor", flush=True)
    model = SampleDiscountModel.fit(build_prediction_feature_frame(train_merged), derive_proxy_2026_discount_pct(train_merged))
    train_scored = score_2026_discount_predictions(train_merged, model)
    validation_scored = score_2026_discount_predictions(validation_merged, model)
    all_scored = score_2026_discount_predictions(all_merged, model)
    prediction_metrics = pd.DataFrame(
        [build_prediction_metrics(train_scored, "train"), build_prediction_metrics(validation_scored, "validation")]
    )
    print("[6/7] Joining optional actual 2026 comparison data", flush=True)
    all_scored, actual_2026_metadata = attach_actual_2026_comparison(all_scored, actual_2026_orders)
    cluster_summary = build_cluster_summary(all_scored)
    data_quality = build_data_quality_summary(orders, profiles)
    print("[7/7] Writing outputs", flush=True)
    profiles.to_csv(output_dir / f"{SAMPLE_OUTPUT_PREFIX}_user_profiles.csv", index=False)
    all_scored.to_csv(output_dir / f"{SAMPLE_OUTPUT_PREFIX}_coupon_recommendations.csv", index=False)
    policy.to_csv(output_dir / f"{SAMPLE_OUTPUT_PREFIX}_category_cluster_policy.csv", index=False)
    cluster_summary.to_csv(output_dir / f"{SAMPLE_OUTPUT_PREFIX}_cluster_summary.csv", index=False)
    data_quality.to_csv(output_dir / f"{SAMPLE_OUTPUT_PREFIX}_data_quality.csv", index=False)
    prediction_metrics.to_csv(output_dir / f"{SAMPLE_OUTPUT_PREFIX}_prediction_metrics.csv", index=False)
    metadata = {
        "sample_rows": int(len(orders)),
        "sample_users": int(orders["userid"].nunique()),
        "clustered_users": int(len(all_scored)),
        "estimated_avoidable_coupon_spend": float(all_scored["estimated_saving_if_2026_matches_history"].sum()),
        "avg_predicted_2026_bucket": float(all_scored["predicted_2026_discount_bucket_pct"].mean()),
        "clusters": sorted(all_scored["coupon_cluster"].dropna().unique().tolist()),
        "validation_bucket_accuracy": float(
            prediction_metrics.loc[prediction_metrics["split"] == "validation", "bucket_accuracy"].iloc[0]
        ),
        "validation_bucket_mae": float(
            prediction_metrics.loc[prediction_metrics["split"] == "validation", "bucket_mae"].iloc[0]
        ),
        **actual_2026_metadata,
    }
    (artifacts_dir / f"{SAMPLE_OUTPUT_PREFIX}_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the sample-training coupon policy pipeline.")
    parser.add_argument("--sample-dir", type=Path, default=SAMPLE_TRAINING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args(argv)
    metadata = run_sample_pipeline(args.sample_dir, args.output_dir, args.artifacts_dir)
    print("Sample training pipeline completed.")
    print(f"Rows processed: {metadata['sample_rows']:,}")
    print(f"Users profiled: {metadata['clustered_users']:,}")
    print(f"Predicted 2026 avg bucket: {metadata['avg_predicted_2026_bucket']:.2f}%")
    print(f"Validation bucket accuracy: {metadata['validation_bucket_accuracy']:.3f}")
    if metadata["actual_2026_available"]:
        print(f"Actual 2026 users joined: {metadata['actual_2026_users']:,}")
    else:
        print("Actual 2026 comparison: not available yet")
    print(f"Estimated avoidable coupon spend: INR {metadata['estimated_avoidable_coupon_spend']:,.0f}")
    return 0
