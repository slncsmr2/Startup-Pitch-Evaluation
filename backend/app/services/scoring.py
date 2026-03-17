from app.schemas import MetricScore


def _avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def score_chunk(text_features: dict, visual_features: dict, audio_features: dict, fused: dict) -> dict:
    text_metrics = [
        MetricScore(
            name="Problem Clarity",
            score=round(text_features["problem_clarity"], 2),
            rationale="How clearly the startup pain point is communicated.",
        ),
        MetricScore(
            name="Market Opportunity",
            score=round(text_features["market_opportunity"], 2),
            rationale="Presence and depth of market framing.",
        ),
        MetricScore(
            name="Solution Uniqueness",
            score=round(text_features["solution_uniqueness"], 2),
            rationale="Distinctiveness of the proposed solution.",
        ),
        MetricScore(
            name="Traction Evidence",
            score=round(text_features["traction_evidence"], 2),
            rationale="Evidence of adoption, growth, pilots, or revenue.",
        ),
        MetricScore(
            name="Business Model Strength",
            score=round(text_features["business_model_strength"], 2),
            rationale="Clarity on monetization and economics.",
        ),
        MetricScore(
            name="Team Readiness",
            score=round(text_features["team_readiness"], 2),
            rationale="Signals of execution capability and preparedness.",
        ),
    ]

    av_metrics = [
        MetricScore(
            name="Delivery Clarity",
            score=round(visual_features["delivery_clarity"], 2),
            rationale="Quality of visual communication and structure.",
        ),
        MetricScore(
            name="Presenter Confidence",
            score=round(visual_features["presenter_confidence"], 2),
            rationale="Confidence cues from visual stream proxies.",
        ),
        MetricScore(
            name="Voice Pace",
            score=round(audio_features["voice_pace"], 2),
            rationale="Speech pacing suitability for investor comprehension.",
        ),
        MetricScore(
            name="Voice Prosody",
            score=round(audio_features["prosody"], 2),
            rationale="Variation and emphasis in delivery.",
        ),
    ]

    text_score = _avg([m.score for m in text_metrics])
    av_score = _avg([m.score for m in av_metrics])

    fusion_signal = _avg(fused["vector"]) * 10
    aggregate = (text_score * 0.5) + (av_score * 0.35) + (fusion_signal * 0.15)

    return {
        "text_metrics": text_metrics,
        "av_metrics": av_metrics,
        "quantitative_scores": text_metrics + av_metrics,
        "aggregate": round(min(10.0, aggregate), 2),
        "confidence": round(min(10.0, (av_score * 0.5) + (text_score * 0.5)), 2),
    }
