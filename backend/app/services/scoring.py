from app.schemas import MetricScore
from models.scoring_head import ScoringHead


_scoring_head = ScoringHead()


_TEXT_RATIONALES = {
    "Problem Clarity": "How clearly the startup pain point is communicated.",
    "Market Opportunity": "Presence and depth of market framing.",
    "Solution Uniqueness": "Distinctiveness of the proposed solution.",
    "Traction Evidence": "Evidence of adoption, growth, pilots, or revenue.",
    "Business Model Strength": "Clarity on monetization and economics.",
    "Team Readiness": "Signals of execution capability and preparedness.",
}

_AV_RATIONALES = {
    "Delivery Clarity": "Quality of visual communication and structure.",
    "Presenter Confidence": "Confidence cues from visual stream proxies.",
    "Voice Pace": "Speech pacing suitability for investor comprehension.",
    "Voice Prosody": "Variation and emphasis in delivery.",
}


def score_chunk(text_features: dict, visual_features: dict, audio_features: dict, fused: dict) -> dict:
    outputs = _scoring_head.infer(text_features, visual_features, audio_features, fused)

    text_metrics = [
        MetricScore(name=name, score=round(score, 2), rationale=_TEXT_RATIONALES[name])
        for name, score in outputs["text_scores"].items()
    ]

    av_metrics = [
        MetricScore(name=name, score=round(score, 2), rationale=_AV_RATIONALES[name])
        for name, score in outputs["av_scores"].items()
    ]

    return {
        "text_metrics": text_metrics,
        "av_metrics": av_metrics,
        "quantitative_scores": text_metrics + av_metrics,
        "aggregate": outputs["aggregate"],
        "confidence": outputs["confidence"],
    }
