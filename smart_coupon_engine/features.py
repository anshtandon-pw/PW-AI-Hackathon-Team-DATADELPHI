from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DISCOUNT_BUCKETS, MAX_DISCOUNT_PCT


def bucketize_discount_pct(values: pd.Series | np.ndarray, buckets: list[int] | None = None) -> np.ndarray:
    active_buckets = buckets or DISCOUNT_BUCKETS
    values_array = np.asarray(values, dtype=float)
    result = np.empty(values_array.shape[0], dtype=float)
    for index, value in enumerate(values_array):
        bucket = active_buckets[-1]
        for candidate in active_buckets:
            if value <= candidate:
                bucket = candidate
                break
        result[index] = bucket
    return result


def derive_proxy_required_discount_pct(df: pd.DataFrame) -> pd.Series:
    baseline = df["min_discount_pct_used"].fillna(df["avg_discount_pct_used"]).fillna(df["discount_rate_pct"]).fillna(0)

    strong_organic = (
        (df["organic_order_count"] >= 1)
        & (
            (df["organic_purchase_score"] >= 50)
            | (df["organic_revenue_share"] >= 0.25)
            | (df["organic_order_share"] >= 0.40)
        )
    )
    moderate_organic = (
        (df["organic_order_count"] >= 1)
        & (
            (df["organic_purchase_score"] >= 25)
            | (df["organic_revenue_share"] >= 0.10)
            | (df["organic_order_share"] >= 0.20)
        )
    )
    high_coupon_dependence = (df["coupon_usage_rate"] >= 80) & (df["organic_order_count"] == 0)
    low_history = (df["total_orders"] <= 1) | (df["customer_tenure_days"] <= 30)
    reactivation = (df["days_since_last_order"] >= 180) & (df["total_orders"] > 1)

    target = baseline.copy()
    target = np.where(strong_organic, 0, target)
    target = np.where(~strong_organic & moderate_organic, np.minimum(target, 10), target)
    avg_discount_fallback = df["avg_discount_pct_used"].fillna(pd.Series(target, index=df.index))
    target = np.where(
        low_history & high_coupon_dependence,
        np.maximum(target, avg_discount_fallback * 0.70),
        target,
    )
    target = np.where(
        reactivation & ~strong_organic,
        np.maximum(target, np.minimum(avg_discount_fallback, target + 5)),
        target,
    )
    return pd.Series(np.clip(target, 0, MAX_DISCOUNT_PCT), index=df.index, dtype="float32")


def build_category_calibration(train_df: pd.DataFrame) -> pd.DataFrame:
    calibration = (
        train_df.groupby("exam_fin", dropna=False)
        .agg(
            category_count=("userid", "size"),
            category_default_policy_pct=("avg_discount_pct_used", "median"),
            category_baseline_pct=("proxy_required_discount_pct", "median"),
            category_floor_pct=("proxy_required_discount_pct", lambda s: float(np.nanquantile(s, 0.10))),
            category_cap_pct=("avg_discount_pct_used", lambda s: float(np.nanquantile(s, 0.90))),
            category_organic_share=("has_organic_history", "mean"),
            category_high_discount_share=("is_high_discount_history", "mean"),
            category_coupon_share=("coupon_order_share", "mean"),
            category_volatility=("proxy_required_discount_pct", "std"),
            category_avg_order_value=("avg_order_value", "median"),
        )
        .reset_index()
    )

    calibration["category_volatility"] = calibration["category_volatility"].fillna(0)
    calibration["category_floor_pct"] = calibration["category_floor_pct"].clip(lower=0, upper=MAX_DISCOUNT_PCT)
    calibration["category_cap_pct"] = calibration["category_cap_pct"].clip(lower=10, upper=MAX_DISCOUNT_PCT)
    calibration["category_baseline_pct"] = calibration["category_baseline_pct"].clip(lower=0, upper=MAX_DISCOUNT_PCT)
    calibration["category_default_policy_pct"] = calibration["category_default_policy_pct"].clip(
        lower=0, upper=MAX_DISCOUNT_PCT
    )

    calibration["category_support_weight"] = (
        np.log1p(calibration["category_count"]) / np.log1p(calibration["category_count"].max())
    ).clip(lower=0.10, upper=1.0)

    calibration["category_floor_pct"] = np.where(
        calibration["category_organic_share"] < 0.05,
        np.maximum(calibration["category_floor_pct"], 10),
        calibration["category_floor_pct"],
    )
    calibration["category_cap_pct"] = np.maximum(
        calibration["category_cap_pct"],
        calibration["category_floor_pct"],
    )
    calibration["category_baseline_bucket"] = bucketize_discount_pct(calibration["category_baseline_pct"])
    return calibration


def add_category_features(df: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(calibration, on="exam_fin", how="left")

    fallback_defaults = {
        "category_count": 0,
        "category_default_policy_pct": merged["avg_discount_pct_used"].median(),
        "category_baseline_pct": merged["proxy_required_discount_pct"].median(),
        "category_floor_pct": 0,
        "category_cap_pct": MAX_DISCOUNT_PCT,
        "category_organic_share": 0,
        "category_high_discount_share": 0,
        "category_coupon_share": 1,
        "category_volatility": 0,
        "category_avg_order_value": merged["avg_order_value"].median(),
        "category_support_weight": 0.10,
        "category_baseline_bucket": 10,
    }

    for column, default_value in fallback_defaults.items():
        merged[column] = merged[column].fillna(default_value)

    return merged


def feature_columns() -> list[str]:
    return [
        "total_orders",
        "customer_tenure_days",
        "days_since_last_order",
        "lifetime_revenue",
        "lifetime_discount",
        "lifetime_net_revenue",
        "discount_rate_pct",
        "coupon_order_count",
        "organic_order_count",
        "coupon_usage_rate",
        "unique_coupons_used",
        "organic_purchase_score",
        "avg_days_between_orders",
        "avg_order_value",
        "organic_order_share",
        "organic_revenue_share",
        "discount_amount_share",
        "avg_discount_pct_gap",
        "category_baseline_pct",
        "category_default_policy_pct",
        "category_organic_share",
        "category_high_discount_share",
        "category_coupon_share",
        "category_volatility",
        "category_avg_order_value",
        "category_support_weight",
        "is_low_history",
        "is_reactivation",
        "has_organic_history",
        "is_high_coupon_usage",
        "is_high_discount_history",
        "log_lifetime_revenue",
        "log_avg_order_value",
        "log_days_since_last_order",
        "log_customer_tenure_days",
    ]
