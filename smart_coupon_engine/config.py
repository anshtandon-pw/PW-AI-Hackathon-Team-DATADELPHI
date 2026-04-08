from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DISCOUNT_BUCKETS = [0, 10, 15, 20, 25, 30, 40, 50]
MAX_DISCOUNT_PCT = 50.0
TRAIN_END_DATE = "2025-09-30"
VALIDATION_END_DATE = "2025-12-31"
RIDGE_ALPHA = 1.5
MODEL_SAMPLE_SIZE = 200_000
RANDOM_SEED = 42

NUMERIC_COLUMNS = [
    "total_orders",
    "customer_tenure_days",
    "days_since_last_order",
    "lifetime_revenue",
    "lifetime_discount",
    "lifetime_net_revenue",
    "discount_rate_pct",
    "coupon_order_count",
    "organic_order_count",
    "unique_coupons_used",
    "coupon_usage_rate",
    "avg_discount_per_coupon_order",
    "min_discount_pct_used",
    "avg_discount_pct_used",
    "organic_revenue",
    "organic_purchase_score",
    "avg_days_between_orders",
    "never_used_coupon",
]

DATE_COLUMNS = ["first_order_date", "last_order_date"]
