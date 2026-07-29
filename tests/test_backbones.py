import json
import os
import pickle

import numpy as np
import pytest

from survfm.backbones import _TabH2OAdapter, get_backbone_info, list_backbones


def test_registry_exposes_named_backbones_including_default():
    names = [item.name for item in list_backbones()]
    assert names[:5] == ["tabpfn", "tabicl", "tabdpt", "tabh2o", "mitra"]
    assert get_backbone_info("tabpfn").display_name == "SurvFM-TabPFN"


def test_tabh2o_adapter_never_serializes_api_key(monkeypatch):
    secret = "private-test-key-that-must-not-persist"
    monkeypatch.setenv("TABH2O_API_KEY", secret)
    model = _TabH2OAdapter()
    model.fit(np.array([[0.0], [1.0]]), np.array([0.5, 1.0]))
    payload = pickle.dumps(model)
    assert secret.encode() not in payload
    assert "TABH2O_API_KEY" in pickle.loads(payload).api_key_env


def test_tabh2o_adapter_builds_anonymized_payload(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"predictions": [0.4]}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setenv("TABH2O_API_KEY", "runtime-only")
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(requests, "post", fake_post)
    model = _TabH2OAdapter().fit(np.array([[0.0], [1.0]]), np.array([0.5, 1.0]))
    assert model.predict(np.array([[0.25]])).tolist() == [0.4]
    assert captured["payload"]["train"]["columns"] == ["x0", "target"]
    assert captured["payload"]["test"]["columns"] == ["x0"]
    assert "runtime-only" not in json.dumps(captured["payload"])
    assert os.environ["TABH2O_API_KEY"] == "runtime-only"
