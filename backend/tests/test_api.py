from fastapi.testclient import TestClient

from app.main import app
from app.schemas import PitchInput
from app.services.inference import InferenceService


def _sample_payload(title: str) -> dict:
    return {
        "title": title,
        "transcript": "",
        "language_hint": "en-ta",
        "presenter_profile": {"team": 3},
        "slide_text": ["Problem", "Solution", "Traction"],
        "video": {
            "file_name": f"{title.lower()}_pitch.mp4",
            "file_format": "mp4",
            "duration_sec": 90,
            "transcript_text": "We help local retailers with AI demand forecasting and bilingual advisory support.",
        },
        "slides": [
            {"title": "Problem", "content": "Inventory mismatch causes loss"},
            {"title": "Solution", "content": "Forecasting and automation"},
            {"title": "Traction", "content": "Pilot adoption in 12 stores"},
        ],
        "user_details": {
            "founder_name": "Founder",
            "startup_name": title,
            "sector": "RetailTech",
            "stage": "Seed",
        },
    }


def test_single_evaluation_endpoint() -> None:
    client = TestClient(app)
    response = client.post("/evaluate", json=_sample_payload("RetailPulse"))
    assert response.status_code == 200

    body = response.json()
    assert "request_id" in body
    assert 0 <= body["summary"]["overall_score"] <= 10
    assert body["summary"]["investment_band"] in {"high-potential", "watchlist", "early-risk"}
    assert body["summary"]["language_detected"] in {"en", "ta", "ta-en"}
    assert len(body["chunk_reports"]) > 0
    assert len(body["dashboard"]["quantitative_scores"]) == 10


def test_batch_evaluation_endpoint() -> None:
    client = TestClient(app)
    payload = {
        "pitches": [
            _sample_payload("RetailPulse"),
            _sample_payload("HealthMesh"),
        ]
    }

    response = client.post("/evaluate/batch", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert len(body["evaluations"]) == 2


def test_batch_evaluation_requires_non_empty_pitches() -> None:
    client = TestClient(app)
    response = client.post("/evaluate/batch", json={"pitches": []})
    assert response.status_code == 400


def test_evaluate_endpoint_matches_shared_inference_output() -> None:
    payload_dict = _sample_payload("ParityCheck")

    client = TestClient(app)
    api_response = client.post("/evaluate", json=payload_dict)
    assert api_response.status_code == 200
    api_body = api_response.json()

    service = InferenceService()
    expected = service.evaluate_payload(PitchInput.model_validate(payload_dict)).model_dump()

    assert api_body == expected