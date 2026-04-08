from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DISCOUNT_BUCKETS, MAX_DISCOUNT_PCT, MODEL_SAMPLE_SIZE, RANDOM_SEED, RIDGE_ALPHA
from .features import bucketize_discount_pct, feature_columns
from .personas import assign_personas


@dataclass
class SmartCouponModel:
    feature_names: list[str]
    means: dict[str, float]
    scales: dict[str, float]
    coefficients: dict[str, float]
    intercept: float
    category_calibration: pd.DataFrame

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SmartCouponModel":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "SmartCouponModel":
        features = feature_columns()
        sampled = train_df.sample(
            n=min(len(train_df), MODEL_SAMPLE_SIZE),
            random_state=RANDOM_SEED,
        )
        x = sampled[features].fillna(0).astype("float64")
        y = sampled["proxy_required_discount_pct"].fillna(0).astype("float64")

        means = x.mean().to_dict()
        scales = x.std().replace(0, 1).to_dict()

        standardized = (x - pd.Series(means)) / pd.Series(scales)
        design_matrix = np.column_stack([np.ones(len(standardized)), standardized.to_numpy()])

        ridge = RIDGE_ALPHA * np.eye(design_matrix.shape[1])
        ridge[0, 0] = 0
        beta = np.linalg.solve(design_matrix.T @ design_matrix + ridge, design_matrix.T @ y.to_numpy())

        intercept = float(beta[0])
        weights = {name: float(weight) for name, weight in zip(features, beta[1:])}
        calibration_columns = [
            "exam_fin",
            "category_count",
            "category_default_policy_pct",
            "category_baseline_pct",
            "category_floor_pct",
            "category_cap_pct",
            "category_organic_share",
            "category_high_discount_share",
            "category_coupon_share",
            "category_volatility",
            "category_avg_order_value",
            "category_support_weight",
            "category_baseline_bucket",
        ]
        calibration = train_df[calibration_columns].drop_duplicates("exam_fin").reset_index(drop=True)

        return cls(
            feature_names=features,
            means=means,
            scales=scales,
            coefficients=weights,
            intercept=intercept,
            category_calibration=calibration,
        )

    def _base_prediction(self, df: pd.DataFrame) -> np.ndarray:
        design = df[self.feature_names].fillna(0).astype("float64")
        standardized = (design - pd.Series(self.means)) / pd.Series(self.scales)
        weights = np.array([self.coefficients[name] for name in self.feature_names], dtype=float)
        return self.intercept + standardized.to_numpy() @ weights

    def _history_anchor(self, df: pd.DataFrame) -> np.ndarray:
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

        anchor = df["min_discount_pct_used"].fillna(df["avg_discount_pct_used"]).fillna(df["category_baseline_pct"]).astype(float)
        avg_discount_fallback = df["avg_discount_pct_used"].fillna(pd.Series(anchor, index=df.index))
        anchor = np.where(strong_organic, 0, anchor)
        anchor = np.where(~strong_organic & moderate_organic, np.minimum(anchor, 10), anchor)
        anchor = np.where(
            (df["coupon_usage_rate"] >= 80) & (df["total_orders"] <= 1),
            np.maximum(anchor, avg_discount_fallback * 0.70),
            anchor,
        )
        anchor = np.where(
            (df["days_since_last_order"] >= 180) & (df["total_orders"] > 1) & ~strong_organic,
            np.minimum(anchor + 5, np.maximum(anchor, df["category_baseline_pct"].to_numpy())),
            anchor,
        )
        return np.clip(anchor, 0, MAX_DISCOUNT_PCT)

    def _confidence(self, df: pd.DataFrame) -> np.ndarray:
        history_score = (
            np.clip(df["total_orders"] / 5, 0, 1) * 0.20
            + np.clip(df["coupon_order_count"] / 4, 0, 1) * 0.15
            + np.clip(df["organic_order_count"] / 2, 0, 1) * 0.10
        )
        category_score = np.clip(df["category_support_weight"], 0, 1) * 0.25
        stability_score = (1 - np.clip(df["avg_discount_pct_gap"] / 25, 0, 1)) * 0.15
        signal_score = (
            np.clip(df["organic_purchase_score"] / 50, 0, 1) * 0.10
            + (1 - np.clip(df["coupon_usage_rate"] / 100, 0, 1)) * 0.05
        )
        confidence = history_score + category_score + stability_score + signal_score
        confidence = np.where(df["is_low_history"] == 1, np.minimum(confidence, 0.45), confidence)
        return np.clip(confidence, 0.10, 0.95)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = assign_personas(df)

        base_pred = self._base_prediction(scored)
        history_anchor = self._history_anchor(scored)
        confidence = self._confidence(scored)

        blended = 0.35 * base_pred + 0.50 * history_anchor + 0.15 * scored["category_baseline_pct"].to_numpy()
        final_pct = confidence * blended + (1 - confidence) * scored["category_baseline_pct"].to_numpy()

        fallback_floor = np.where(
            confidence < 0.45,
            np.maximum(scored["category_floor_pct"], np.minimum(scored["category_default_policy_pct"], 10)),
            scored["category_floor_pct"],
        )
        final_pct = np.clip(final_pct, fallback_floor, scored["category_cap_pct"].to_numpy())

        strong_organic = (
            (scored["organic_order_count"] >= 1)
            & (
                (scored["organic_purchase_score"] >= 50)
                | (scored["organic_revenue_share"] >= 0.25)
                | (scored["organic_order_share"] >= 0.40)
            )
            & (confidence >= 0.65)
        )
        final_pct = np.where(strong_organic, 0, final_pct)

        recommended_bucket = bucketize_discount_pct(final_pct, DISCOUNT_BUCKETS)
        reference_discount_pct = scored["avg_discount_pct_used"].fillna(scored["category_default_policy_pct"]).to_numpy()
        default_policy_pct = scored["category_default_policy_pct"].to_numpy()

        reference_order_discount_amount = scored["avg_order_value"].to_numpy() * reference_discount_pct / 100
        default_policy_discount_amount = scored["avg_order_value"].to_numpy() * default_policy_pct / 100
        recommended_discount_amount = scored["avg_order_value"].to_numpy() * recommended_bucket / 100

        next_order_avoidable_amount = np.maximum(0, reference_order_discount_amount - recommended_discount_amount)
        lifetime_avoidable_amount = next_order_avoidable_amount * scored["coupon_order_count"].fillna(1).to_numpy()

        organic_evidence = (
            np.clip(scored["organic_order_count"] / 2, 0, 1) * 0.50
            + np.clip(scored["organic_purchase_score"] / 50, 0, 1) * 0.50
        )
        gap_pct = np.maximum(0, reference_discount_pct - recommended_bucket)
        gap_score = np.clip(gap_pct / 20, 0, 1)
        proof_confidence = np.clip(0.50 * confidence + 0.30 * organic_evidence + 0.20 * gap_score, 0.10, 0.95)

        conservative_amount = lifetime_avoidable_amount * (0.30 + 0.40 * proof_confidence)
        expected_amount = lifetime_avoidable_amount * proof_confidence
        upper_amount = lifetime_avoidable_amount * (1.10 + 0.40 * (1 - proof_confidence))

        proof_label = np.where(
            gap_pct <= 3,
            "likely_necessary",
            np.where((gap_pct >= 10) & (proof_confidence >= 0.70), "likely_unnecessary", "uncertain"),
        )

        scored["base_prediction_pct"] = np.clip(base_pred, 0, MAX_DISCOUNT_PCT).astype("float32")
        scored["history_anchor_pct"] = history_anchor.astype("float32")
        scored["confidence_score"] = confidence.astype("float32")
        scored["recommended_discount_pct"] = final_pct.astype("float32")
        scored["recommended_discount_bucket"] = recommended_bucket.astype("float32")
        scored["reference_discount_pct"] = reference_discount_pct.astype("float32")
        scored["default_policy_pct"] = default_policy_pct.astype("float32")
        scored["expected_unnecessary_discount_amount"] = lifetime_avoidable_amount.astype("float32")
        scored["expected_unnecessary_discount_amount_next_order"] = next_order_avoidable_amount.astype("float32")
        scored["conservative_avoidable_discount_amount"] = conservative_amount.astype("float32")
        scored["expected_avoidable_discount_amount"] = expected_amount.astype("float32")
        scored["upper_avoidable_discount_amount"] = upper_amount.astype("float32")
        scored["proof_confidence_score"] = proof_confidence.astype("float32")
        scored["proof_label"] = pd.Series(proof_label, index=scored.index, dtype="string")
        scored["recommended_discount_amount"] = recommended_discount_amount.astype("float32")
        scored["default_policy_discount_amount"] = default_policy_discount_amount.astype("float32")
        scored["reference_discount_gap_pct"] = gap_pct.astype("float32")
        scored["fallback_used"] = (confidence < 0.45).astype("int8")
        scored["proxy_target_bucket"] = bucketize_discount_pct(scored["proxy_required_discount_pct"]).astype("float32")
        return scored
