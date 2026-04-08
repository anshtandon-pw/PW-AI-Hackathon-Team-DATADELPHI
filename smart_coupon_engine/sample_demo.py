from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from .sample_pipeline import SAMPLE_TRAINING_DIR, run_sample_pipeline


class SampleDemoDataStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.outputs_dir = project_root / "outputs"
        self.artifacts_dir = project_root / "artifacts"
        self.static_dir = project_root / "demo_ui"
        self._ensure_outputs()
        self._load()

    def _ensure_outputs(self) -> None:
        required = [
            self.outputs_dir / "sample_coupon_recommendations.csv",
            self.outputs_dir / "sample_category_cluster_policy.csv",
            self.outputs_dir / "sample_cluster_summary.csv",
            self.outputs_dir / "sample_data_quality.csv",
            self.outputs_dir / "sample_prediction_metrics.csv",
            self.artifacts_dir / "sample_run_metadata.json",
        ]
        if any(not path.exists() for path in required):
            run_sample_pipeline(SAMPLE_TRAINING_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_ARTIFACTS_DIR)

    def _load(self) -> None:
        with (self.artifacts_dir / "sample_run_metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.recommendations = pd.read_csv(
            self.outputs_dir / "sample_coupon_recommendations.csv",
            dtype={
                "userid": "string",
                "source_group": "string",
                "lead_type": "string",
                "primary_exam_fin": "string",
                "last_exam_fin": "string",
                "coupon_cluster": "string",
                "cluster_reason_codes": "string",
                "prediction_reason_codes": "string",
                "prediction_split": "string",
            },
        )
        self.policy = pd.read_csv(self.outputs_dir / "sample_category_cluster_policy.csv")
        self.cluster_summary = pd.read_csv(self.outputs_dir / "sample_cluster_summary.csv")
        self.data_quality = pd.read_csv(self.outputs_dir / "sample_data_quality.csv")
        self.prediction_metrics = pd.read_csv(self.outputs_dir / "sample_prediction_metrics.csv")
        for column in ["userid", "source_group", "lead_type", "primary_exam_fin", "last_exam_fin", "coupon_cluster", "cluster_reason_codes", "prediction_reason_codes"]:
            if column in self.recommendations.columns:
                self.recommendations[column] = self.recommendations[column].fillna("").astype("string")

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        return json.loads(frame.to_json(orient="records"))

    def _filtered_users_frame(
        self,
        query: str = "",
        category: str = "",
        cluster: str = "",
        lead_type: str = "",
        source_group: str = "",
    ) -> pd.DataFrame:
        frame = self.recommendations
        if category:
            frame = frame[frame["primary_exam_fin"] == category]
        if cluster:
            frame = frame[frame["coupon_cluster"] == cluster]
        if lead_type:
            frame = frame[frame["lead_type"] == lead_type]
        if source_group:
            frame = frame[frame["source_group"] == source_group]
        query = query.strip().lower()
        if query:
            exact = frame[frame["userid"].str.lower() == query]
            contains = frame[
                frame["userid"].str.lower().str.contains(query, na=False)
                | frame["primary_exam_fin"].str.lower().str.contains(query, na=False)
                | frame["coupon_cluster"].str.lower().str.contains(query, na=False)
                | frame["lead_type"].str.lower().str.contains(query, na=False)
            ]
            frame = pd.concat([exact, contains]).drop_duplicates(subset=["userid"], keep="first")
        return frame.sort_values(
            ["estimated_saving_if_2026_matches_history", "prediction_confidence"],
            ascending=[False, False],
        )

    def options(self) -> dict[str, Any]:
        return {
            "categories": sorted(self.recommendations["primary_exam_fin"].dropna().unique().tolist()),
            "clusters": sorted(self.recommendations["coupon_cluster"].dropna().unique().tolist()),
            "lead_types": sorted(self.recommendations["lead_type"].dropna().unique().tolist()),
            "source_groups": sorted(self.recommendations["source_group"].dropna().unique().tolist()),
        }

    def overview(self) -> dict[str, Any]:
        top_users = (
            self.recommendations.sort_values(
                ["estimated_saving_if_2026_matches_history", "prediction_confidence"],
                ascending=[False, False],
            )
            .head(12)
            .copy()
        )
        top_policies = self.policy.sort_values(["policy_users", "recommended_base_bucket"], ascending=[False, False]).head(15)
        return {
            "metadata": self.metadata,
            "cluster_summary": self._records(self.cluster_summary),
            "prediction_metrics": self._records(self.prediction_metrics),
            "top_policies": self._records(top_policies),
            "top_users": self._records(
                top_users.loc[
                    :,
                    [
                        "userid",
                        "primary_exam_fin",
                        "lead_type",
                        "coupon_cluster",
                        "predicted_2026_discount_bucket_pct",
                        "historical_reference_discount_bucket_pct",
                        "estimated_saving_next_order_2026",
                        "prediction_confidence",
                    ],
                ]
            ),
            "data_quality": self._records(self.data_quality),
        }

    def search_users(
        self,
        query: str = "",
        category: str = "",
        cluster: str = "",
        lead_type: str = "",
        source_group: str = "",
        limit: int = 25,
    ) -> dict[str, Any]:
        frame = self._filtered_users_frame(query, category, cluster, lead_type, source_group)
        total_matches = int(len(frame))
        limit = max(1, min(int(limit), 1000))
        frame = frame.head(limit)
        columns = [
            "userid",
            "primary_exam_fin",
            "lead_type",
            "source_group",
            "coupon_cluster",
            "predicted_2026_discount_bucket_pct",
            "historical_reference_discount_bucket_pct",
            "estimated_saving_next_order_2026",
            "estimated_saving_if_2026_matches_history",
            "prediction_confidence",
        ]
        return {
            "total_matches": total_matches,
            "shown_count": int(len(frame)),
            "items": self._records(frame.loc[:, columns]),
        }

    def filtered_users_csv(
        self,
        query: str = "",
        category: str = "",
        cluster: str = "",
        lead_type: str = "",
        source_group: str = "",
    ) -> bytes:
        frame = self._filtered_users_frame(query, category, cluster, lead_type, source_group)
        columns = [
            "userid",
            "primary_exam_fin",
            "lead_type",
            "source_group",
            "coupon_cluster",
            "historical_reference_discount_bucket_pct",
            "predicted_2026_discount_bucket_pct",
            "estimated_saving_next_order_2026",
            "estimated_saving_if_2026_matches_history",
            "prediction_confidence",
            "discount_sensitivity_class",
            "historical_reference_support_share",
            "prediction_reason_codes",
        ]
        available_columns = [column for column in columns if column in frame.columns]
        return frame.loc[:, available_columns].to_csv(index=False).encode("utf-8")

    def user_detail(self, user_id: str) -> dict[str, Any] | None:
        selected = self.recommendations[self.recommendations["userid"] == user_id]
        if selected.empty:
            return None
        first = selected.iloc[0]
        policy = self.policy[
            (self.policy["primary_exam_fin"] == first["primary_exam_fin"])
            & (self.policy["coupon_cluster"] == first["coupon_cluster"])
        ]
        peers = (
            self.recommendations[
                (self.recommendations["primary_exam_fin"] == first["primary_exam_fin"])
                & (self.recommendations["coupon_cluster"] == first["coupon_cluster"])
            ]
            .sort_values(["estimated_saving_if_2026_matches_history", "prediction_confidence"], ascending=[False, False])
            .head(8)
        )
        return {
            "user": self._records(selected.head(1))[0],
            "policy": self._records(policy.head(1))[0] if not policy.empty else None,
            "peers": self._records(
                peers.loc[
                    :,
                    [
                        "userid",
                        "lead_type",
                        "predicted_2026_discount_bucket_pct",
                        "historical_reference_discount_bucket_pct",
                        "estimated_saving_next_order_2026",
                        "prediction_confidence",
                    ],
                ]
            ),
        }

    def policies(self, limit: int = 25) -> list[dict[str, Any]]:
        frame = self.policy.sort_values(["policy_users", "recommended_base_bucket"], ascending=[False, False]).head(limit)
        return self._records(frame)


def make_handler(state: SampleDemoDataStore) -> type[BaseHTTPRequestHandler]:
    class SampleDemoRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_csv(self, content: bytes, file_name: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{file_name}"')
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_file(self, file_path: Path, status: int = 200) -> None:
            if not file_path.exists():
                self._send_json({"error": "Not found"}, status=404)
                return
            content = file_path.read_bytes()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            self.send_response(status)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/api/sample/health":
                self._send_json({"status": "ok"})
                return
            if path == "/api/sample/overview":
                self._send_json(state.overview())
                return
            if path == "/api/sample/options":
                self._send_json(state.options())
                return
            if path == "/api/sample/policies":
                self._send_json({"items": state.policies(limit=int(query.get("limit", ["25"])[0]))})
                return
            if path == "/api/sample/users":
                self._send_json(
                    state.search_users(
                        query=query.get("query", [""])[0],
                        category=query.get("category", [""])[0],
                        cluster=query.get("cluster", [""])[0],
                        lead_type=query.get("lead_type", [""])[0],
                        source_group=query.get("source_group", [""])[0],
                        limit=int(query.get("limit", ["25"])[0]),
                    )
                )
                return
            if path == "/api/sample/users-export":
                content = state.filtered_users_csv(
                    query=query.get("query", [""])[0],
                    category=query.get("category", [""])[0],
                    cluster=query.get("cluster", [""])[0],
                    lead_type=query.get("lead_type", [""])[0],
                    source_group=query.get("source_group", [""])[0],
                )
                self._send_csv(content, "sample_filtered_users.csv")
                return
            if path.startswith("/api/sample/user/"):
                user_id = unquote(path.split("/api/sample/user/", 1)[1])
                payload = state.user_detail(user_id)
                if payload is None:
                    self._send_json({"error": "User not found"}, status=404)
                    return
                self._send_json(payload)
                return
            if path in ("/", "/sample", "/sample/index.html"):
                self._send_file(state.static_dir / "sample_index.html")
                return
            if path == "/sample_app.js":
                self._send_file(state.static_dir / "sample_app.js")
                return
            if path == "/styles.css":
                self._send_file(state.static_dir / "styles.css")
                return
            self._send_json({"error": "Not found"}, status=404)

    return SampleDemoRequestHandler


def create_server(host: str = "127.0.0.1", port: int = 8010) -> tuple[ThreadingHTTPServer, SampleDemoDataStore]:
    state = SampleDemoDataStore(PROJECT_ROOT)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the sample-training coupon demo UI/API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args(argv)
    server, _ = create_server(args.host, args.port)
    print(f"Sample coupon demo running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
