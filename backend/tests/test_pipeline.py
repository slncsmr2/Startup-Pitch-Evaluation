from app.pipeline import StartupPitchPipeline
from app.schemas import PitchInput, PitchVideoInput, SlideInput, UserDetails


def test_pipeline_returns_scores_and_feedback() -> None:
    payload = PitchInput(
        title="AgriSage",
        transcript="",
        language_hint="en-ta",
        presenter_profile={"experience": "5 years"},
        slide_text=["Problem", "Solution", "Traction", "Business Model"],
        video=PitchVideoInput(
            file_name="agrisage_pitch.mp4",
            file_format="mp4",
            duration_sec=120,
            transcript_text="We solve crop loss for farmers using bilingual AI support. "
            "Our pilot improved yield and market access in Tamil Nadu.",
        ),
        slides=[
            SlideInput(title="Problem", content="Post-harvest crop loss is high"),
            SlideInput(title="Market", content="Large underserved smallholder segment"),
            SlideInput(title="Traction", content="Pilot growth and repeat usage"),
        ],
        user_details=UserDetails(
            founder_name="Aravind",
            startup_name="AgriSage",
            sector="AgriTech",
            stage="Seed",
        ),
    )

    response = StartupPitchPipeline(window_seconds=5).evaluate(payload, request_id="test-id")

    assert response.request_id == "test-id"
    assert 0 <= response.summary.overall_score <= 10
    assert 0 <= response.summary.confidence_score <= 10
    assert response.summary.language_detected in {"en", "ta", "ta-en"}
    assert len(response.chunk_reports) > 0
    assert len(response.dashboard.quantitative_scores) == 10
