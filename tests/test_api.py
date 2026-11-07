"""API tests with a fake detector: they check the HTTP contract, not the model."""

import pytest
from fastapi.testclient import TestClient

from src.api import create_app


class FakeDetector:
    threshold = 0.87

    def __init__(self):
        self.calls = []

    def predict(self, question: str, answer: str) -> dict:
        self.calls.append((question, answer))
        p = 0.95 if "significantly" in answer else 0.10
        return {"p_hallucinated": p, "threshold": self.threshold,
                "label": "hallucinated" if p >= self.threshold else "ground_truth"}


@pytest.fixture
def fake():
    return FakeDetector()


@pytest.fixture
def client(fake):
    with TestClient(create_app(detector=fake)) as c:  # `with` runs startup/shutdown
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.parametrize("answer, label", [
    ("X significantly reduces Y.", "hallucinated"),
    ("X may reduce Y.", "ground_truth"),
])
def test_detect_returns_label_probability_and_threshold(client, answer, label):
    r = client.post("/detect", json={"question": "Does X reduce Y?", "answer": answer})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == label
    assert 0.0 <= body["p_hallucinated"] <= 1.0
    assert body["threshold"] == FakeDetector.threshold


def test_detect_strips_whitespace_before_predicting(client, fake):
    client.post("/detect", json={"question": "  Q?  ", "answer": " A. "})
    assert fake.calls == [("Q?", "A.")]


@pytest.mark.parametrize("payload", [
    {"question": "Q?"},                         # missing answer
    {"question": "   ", "answer": "A."},        # whitespace-only question
    {"question": "Q?", "answer": ""},           # empty answer
    {"question": "Q?", "answer": "x" * 5001},   # too long
])
def test_detect_rejects_invalid_input(client, fake, payload):
    r = client.post("/detect", json=payload)
    assert r.status_code == 422
    assert fake.calls == []  # invalid input never reaches the model
