from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from .pipeline import run_pipeline


class DemoDataStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.outputs_dir = project_root / "outputs"
        self.artifacts_dir = project_root / "artifacts"
        self.static_dir = project_root / "demo_ui"
        self._ensure_outputs()
        self._load()

    def _ensure_outputs(self) -> None:
        required = [
            self.outputs_dir / "recommendations_2026.csv",
            self.outputs_dir / "category_dashboard.csv",
            self.outputs_dir / "waste_analysis_2025.csv",
            self.outputs_dir / "split_metrics.csv",
            self.artifacts_dir / "run_metadata.json",
        ]
        if any(not path.exists() for path in required):
            run_pipeline(DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_ARTIFACTS_DIR)

    def _load(self) -> None:
        with (self.artifacts_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.recommendations = pd.read_csv(
            self.outputs_dir / "recommendations_2026.csv",
            dtype={
                "userid": "string",
                "exam_fin": "string",
                "persona": "string",
                "reason_codes": "string",
                "proof_label": "string",
            },
        )
        self.category_dashboard = pd.read_csv(self.outputs_dir / "category_dashboard.csv")
        self.waste_analysis = pd.read_csv(self.outputs_dir / "waste_analysis_2025.csv")
        self.split_metrics = pd.read_csv(self.outputs_dir / "split_metrics.csv")

        top_users_path = self.outputs_dir / "top_user_samples.csv"
        if top_users_path.exists():
            self.top_users = pd.read_csv(
                top_users_path,
                dtype={"userid": "string", "exam_fin": "string", "persona": "string"},
            )
        else:
            self.top_users = self.recommendations.sort_values(
                ["expected_unnecessary_discount_amount", "proof_confidence_score"],
                ascending=[False, False],
            ).head(50)

        self.recommendations["userid"] = self.recommendations["userid"].fillna("").astype("string")
        self.recommendations["exam_fin"] = self.recommendations["exam_fin"].fillna("").astype("string")
        self.recommendations["persona"] = self.recommendations["persona"].fillna("").astype("string")
        self.recommendations["proof_label"] = self.recommendations["proof_label"].fillna("").astype("string")

        # Older generated sample files may not contain every display field used by the demo.
        sample_defaults: dict[str, Any] = {
            "userid": "",
            "exam_fin": "",
            "persona": "unknown",
            "proof_label": "uncertain",
            "recommended_discount_bucket": 0,
            "reference_discount_pct": 0,
            "confidence_score": 0,
            "proof_confidence_score": 0,
            "expected_unnecessary_discount_amount": 0,
            "fallback_used": 0,
        }
        for column, default_value in sample_defaults.items():
            if column not in self.top_users.columns:
                self.top_users[column] = default_value

        self.top_users["userid"] = self.top_users["userid"].fillna("").astype("string")
        self.top_users["exam_fin"] = self.top_users["exam_fin"].fillna("").astype("string")
        self.top_users["persona"] = self.top_users["persona"].fillna("unknown").astype("string")
        self.top_users["proof_label"] = self.top_users["proof_label"].fillna("uncertain").astype("string")

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        return json.loads(frame.to_json(orient="records"))

    def options(self) -> dict[str, Any]:
        return {
            "categories": sorted(self.recommendations["exam_fin"].dropna().unique().tolist()),
            "personas": sorted(self.recommendations["persona"].dropna().unique().tolist()),
            "proof_labels": sorted(self.recommendations["proof_label"].dropna().unique().tolist()),
        }

    def overview(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "split_metrics": self._records(self.split_metrics),
            "waste_analysis": self._records(self.waste_analysis),
            "top_categories": self._records(self.category_dashboard.head(12)),
            "top_users": self._records(self.top_users.head(12)),
        }

    def search_users(
        self,
        query: str = "",
        category: str = "",
        persona: str = "",
        proof_label: str = "",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        frame = self.recommendations

        if category:
            frame = frame[frame["exam_fin"] == category]
        if persona:
            frame = frame[frame["persona"] == persona]
        if proof_label:
            frame = frame[frame["proof_label"] == proof_label]

        query = query.strip().lower()
        if query:
            frame = frame[
                frame["userid"].str.lower().str.contains(query, na=False)
                | frame["exam_fin"].str.lower().str.contains(query, na=False)
                | frame["persona"].str.lower().str.contains(query, na=False)
            ]

        limit = max(1, min(int(limit), 100))
        frame = frame.sort_values(
            ["expected_unnecessary_discount_amount", "proof_confidence_score"],
            ascending=[False, False],
        ).head(limit)

        columns = [
            "userid",
            "exam_fin",
            "persona",
            "recommended_discount_bucket",
            "reference_discount_pct",
            "proof_label",
            "confidence_score",
            "proof_confidence_score",
            "expected_unnecessary_discount_amount",
            "fallback_used",
        ]
        return self._records(frame.loc[:, columns])

    def user_detail(self, user_id: str) -> dict[str, Any] | None:
        selected = self.recommendations[self.recommendations["userid"] == user_id]
        if selected.empty:
            return None

        user = self._records(selected.head(1))[0]
        category_name = selected.iloc[0]["exam_fin"]

        category = self.category_dashboard[self.category_dashboard["exam_fin"] == category_name]
        peer_rows = (
            self.recommendations[self.recommendations["exam_fin"] == category_name]
            .sort_values(["expected_unnecessary_discount_amount", "proof_confidence_score"], ascending=[False, False])
            .head(8)
        )

        return {
            "user": user,
            "category": self._records(category.head(1))[0] if not category.empty else None,
            "peers": self._records(
                peer_rows.loc[
                    :,
                    [
                        "userid",
                        "persona",
                        "recommended_discount_bucket",
                        "reference_discount_pct",
                        "proof_label",
                        "expected_unnecessary_discount_amount",
                    ],
                ]
            ),
        }

    def top_categories(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        return self._records(self.category_dashboard.head(limit))


def make_handler(state: DemoDataStore) -> type[BaseHTTPRequestHandler]:
    class DemoRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

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

            if path == "/api/health":
                self._send_json({"status": "ok"})
                return

            if path == "/api/overview":
                self._send_json(state.overview())
                return

            if path == "/api/options":
                self._send_json(state.options())
                return

            if path == "/api/categories":
                limit = int(query.get("limit", ["20"])[0])
                self._send_json({"items": state.top_categories(limit=limit)})
                return

            if path == "/api/users":
                payload = state.search_users(
                    query=query.get("query", [""])[0],
                    category=query.get("category", [""])[0],
                    persona=query.get("persona", [""])[0],
                    proof_label=query.get("proof_label", [""])[0],
                    limit=int(query.get("limit", ["25"])[0]),
                )
                self._send_json({"items": payload})
                return

            if path.startswith("/api/user/"):
                user_id = unquote(path.split("/api/user/", 1)[1])
                payload = state.user_detail(user_id)
                if payload is None:
                    self._send_json({"error": "User not found"}, status=404)
                    return
                self._send_json(payload)
                return

            if path in ("/", "/index.html"):
                self._send_file(state.static_dir / "index.html")
                return
            if path == "/app.js":
                self._send_file(state.static_dir / "app.js")
                return
            if path == "/styles.css":
                self._send_file(state.static_dir / "styles.css")
                return

            self._send_json({"error": "Not found"}, status=404)

    return DemoRequestHandler


def create_server(host: str = "127.0.0.1", port: int = 8000) -> tuple[ThreadingHTTPServer, DemoDataStore]:
    state = DemoDataStore(PROJECT_ROOT)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Smart Coupon Engine demo UI/API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    server, _ = create_server(args.host, args.port)
    print(f"Smart Coupon Engine demo running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
