"""Tests for Model.serve() — the FastAPI inference server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skip serve tests")

from tests.test_model import _make_mock_base_model, _make_mock_peft_model, _make_mock_tokenizer


@pytest.fixture()
def served_app(tmp_path: Path):
    """Build a Model and capture the FastAPI app serve() constructs, without binding a socket."""
    mock_tokenizer = _make_mock_tokenizer()
    mock_base = _make_mock_base_model()
    mock_peft = _make_mock_peft_model()
    mock_peft.to = MagicMock(return_value=mock_peft)

    with (
        patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
        patch("pyrecall.model.get_peft_model", return_value=mock_peft),
    ):
        from pyrecall.model import Model

        m = Model("test/model", snapshot_dir=tmp_path / "snapshots")
        m.model = mock_peft
        m.tokenizer = mock_tokenizer
        m.device = "cpu"

    # Assigned directly (not via patch.object) so the mock stays in place for the
    # life of the test — the app's /generate closure calls self.generate_stream
    # lazily, on each request, well after serve() itself has returned.
    m.generate_stream = MagicMock(return_value=iter(["Hello", " world"]))

    with patch("uvicorn.run") as mock_run:
        m.serve(port=8000, host="127.0.0.1")
        app = mock_run.call_args.args[0]
        captured_host = mock_run.call_args.kwargs.get("host")
        captured_port = mock_run.call_args.kwargs.get("port")

    return app, captured_host, captured_port, m


class TestServe:
    def test_serve_binds_configured_host_and_port(self, served_app) -> None:
        _app, host, port, _m = served_app
        assert host == "127.0.0.1"
        assert port == 8000

    def test_health_endpoint(self, served_app) -> None:
        from fastapi.testclient import TestClient

        app, _host, _port, m = served_app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == m.model_name

    def test_generate_streams_tokens(self, served_app) -> None:
        from fastapi.testclient import TestClient

        app, _host, _port, _m = served_app
        client = TestClient(app)
        resp = client.post("/generate", json={"prompt": "hi"})
        assert resp.status_code == 200
        body = resp.text
        assert '"token": "Hello"' in body
        assert '"done": true' in body
