from __future__ import annotations

import numpy as np
import pandas as pd


def assign_personas(df: pd.DataFrame) -> pd.DataFrame:
    persona = np.full(len(df), "coupon-dependent buyer", dtype=object)

    organic_mask = (
        (df["organic_order_count"] >= 1)
        & (
            (df["organic_purchase_score"] >= 40)
            | (df["organic_revenue_share"] >= 0.20)
            | (df["organic_order_share"] >= 0.30)
        )
    )
    new_mask = (df["total_orders"] <= 1) | (df["customer_tenure_days"] <= 30)
    reactivation_mask = (df["days_since_last_order"] >= 180) & (df["total_orders"] > 1)
    high_discount_mask = (
        (df["min_discount_pct_used"] >= 30)
        | (df["avg_discount_pct_used"] >= 40)
        | (df["discount_rate_pct"] >= 40)
    )

    persona[new_mask] = "new/low-history user"
    persona[reactivation_mask] = "reactivation user"
    persona[high_discount_mask] = "high-discount dependent"
    persona[organic_mask] = "organic buyer"

    reason_codes = []
    for _, row in df.iterrows():
        codes: list[str] = []
        if row["organic_order_count"] >= 1:
            codes.append("organic_history")
        if row["coupon_usage_rate"] >= 80:
            codes.append("coupon_heavy")
        if row["min_discount_pct_used"] >= 30 or row["avg_discount_pct_used"] >= 40:
            codes.append("high_discount_history")
        if row["days_since_last_order"] >= 180 and row["total_orders"] > 1:
            codes.append("reactivation_risk")
        if row["total_orders"] <= 1 or row["customer_tenure_days"] <= 30:
            codes.append("low_history")
        if not codes:
            codes.append("category_baseline")
        reason_codes.append("|".join(codes))

    enriched = df.copy()
    enriched["persona"] = pd.Series(persona, index=df.index, dtype="string")
    enriched["reason_codes"] = pd.Series(reason_codes, index=df.index, dtype="string")
    return enriched
