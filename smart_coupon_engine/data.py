from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATE_COLUMNS, NUMERIC_COLUMNS, TRAIN_END_DATE, VALIDATION_END_DATE


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_raw_data(path: str | Path) -> pd.DataFrame:
    raw_path = Path(path)
    df = pd.read_csv(raw_path, low_memory=False, dtype={"userid": "string", "exam_fin": "string"})

    for column in NUMERIC_COLUMNS:
        df[column] = _clean_numeric(df[column]).astype("float32")

    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], format="%d-%b-%y", errors="coerce")

    df["exam_fin"] = df["exam_fin"].fillna("UNKNOWN").str.strip()

    df["avg_days_between_orders"] = df["avg_days_between_orders"].fillna(df["customer_tenure_days"])
    df["never_used_coupon"] = df["never_used_coupon"].fillna(0)

    df["first_order_year"] = df["first_order_date"].dt.year.astype("Int64")
    df["last_order_year"] = df["last_order_date"].dt.year.astype("Int64")

    safe_total_orders = df["total_orders"].replace(0, np.nan)
    safe_revenue = df["lifetime_revenue"].replace(0, np.nan)

    df["avg_order_value"] = (df["lifetime_revenue"] / safe_total_orders).fillna(0).astype("float32")
    df["avg_net_revenue_per_order"] = (df["lifetime_net_revenue"] / safe_total_orders).fillna(0).astype("float32")
    df["avg_discount_pct_gap"] = (
        df["avg_discount_pct_used"].fillna(0) - df["min_discount_pct_used"].fillna(0)
    ).clip(lower=0).astype("float32")
    df["organic_order_share"] = (df["organic_order_count"] / safe_total_orders).fillna(0).astype("float32")
    df["coupon_order_share"] = (df["coupon_order_count"] / safe_total_orders).fillna(0).astype("float32")
    df["organic_revenue_share"] = (df["organic_revenue"] / safe_revenue).fillna(0).astype("float32")
    df["discount_amount_share"] = (df["lifetime_discount"] / safe_revenue).fillna(0).astype("float32")
    df["coupon_discount_amount_per_order"] = df["avg_discount_per_coupon_order"].fillna(0).astype("float32")
    df["log_lifetime_revenue"] = np.log1p(df["lifetime_revenue"]).astype("float32")
    df["log_avg_order_value"] = np.log1p(df["avg_order_value"]).astype("float32")
    df["log_days_since_last_order"] = np.log1p(df["days_since_last_order"]).astype("float32")
    df["log_customer_tenure_days"] = np.log1p(df["customer_tenure_days"]).astype("float32")

    df["is_low_history"] = (
        (df["total_orders"] <= 1) | (df["customer_tenure_days"] <= 30)
    ).astype("int8")
    df["is_reactivation"] = (
        (df["days_since_last_order"] >= 180) & (df["total_orders"] > 1)
    ).astype("int8")
    df["has_organic_history"] = (df["organic_order_count"] > 0).astype("int8")
    df["is_high_coupon_usage"] = (df["coupon_usage_rate"] >= 80).astype("int8")
    df["is_high_discount_history"] = (
        (df["min_discount_pct_used"] >= 30) | (df["avg_discount_pct_used"] >= 40)
    ).astype("int8")

    constant_columns = []
    for column in df.columns:
        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    df.attrs["constant_columns"] = constant_columns
    return df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(TRAIN_END_DATE)
    validation_end = pd.Timestamp(VALIDATION_END_DATE)

    train = df[df["last_order_date"] <= train_end].copy()
    validation = df[(df["last_order_date"] > train_end) & (df["last_order_date"] <= validation_end)].copy()
    test = df[df["last_order_date"] > validation_end].copy()
    return train, validation, test
