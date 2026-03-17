import logging

from app.core.config import settings
from app.schemas import (
    ChunkReport,
    DashboardSeriesPoint,
    EvaluationResponse,
    EvaluationSummary,
    InvestorDashboard,
    PitchInput,
)
from app.services.extractors import AudioFeatureExtractor, TextFeatureExtractor, VisualFeatureExtractor
from app.services.fusion import fuse_modalities
from app.services.preprocessing import temporal_synchronize_and_segment
from app.services.reporting import build_feedback, build_investor_dashboard
from app.services.risk import detect_risk_flags
from app.services.scoring import score_chunk

logger = logging.getLogger(__name__)


class StartupPitchPipeline:
    def __init__(self, window_seconds: int = 5):
        self.window_seconds = window_seconds
        self.text_extractor = TextFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        
        # Log active pipeline mode
        logger.info(
            "Pipeline initialized | "
            f"use_heuristic_pipeline={settings.use_heuristic_pipeline} | "
            f"use_local_transcriber={settings.use_local_transcriber}"
        )

    def evaluate(self, payload: PitchInput, request_id: str) -> EvaluationResponse:
        chunks = temporal_synchronize_and_segment(payload, self.window_seconds)
        chunk_reports: list[ChunkReport] = []

        aggregate_scores: list[float] = []
        confidence_scores: list[float] = []
        text_scores: list[float] = []
        av_scores: list[float] = []
        language_predictions: list[str] = []
        all_quantitative_scores: list[dict] = []
        modality_totals = {"text": 0.0, "visual": 0.0, "audio": 0.0}
        risk_counts: dict[str, int] = {}
        user_stage = payload.user_details.stage if payload.user_details else ""

        for chunk in chunks:
            text_features = self.text_extractor.extract(chunk.text, payload.language_hint, chunk.slide_context)
            visual_features = self.visual_extractor.extract(chunk.slide_context, chunk.chunk_id, user_stage)
            audio_features = self.audio_extractor.extract(chunk.text)
            language_predictions.append(text_features["language_detected"])

            fused = fuse_modalities(text_features, visual_features, audio_features)
            scored = score_chunk(text_features, visual_features, audio_features, fused)
            all_quantitative_scores.extend(
                {"name": item.name, "score": item.score} for item in scored["quantitative_scores"]
            )
            modality_totals["text"] += fused["attention"]["text"]
            modality_totals["visual"] += fused["attention"]["visual"]
            modality_totals["audio"] += fused["attention"]["audio"]

            text_avg = sum(m.score for m in scored["text_metrics"]) / len(scored["text_metrics"])
            av_avg = sum(m.score for m in scored["av_metrics"]) / len(scored["av_metrics"])

            text_scores.append(text_avg)
            av_scores.append(av_avg)
            aggregate_scores.append(scored["aggregate"])
            confidence_scores.append(scored["confidence"])

            risk_flags = detect_risk_flags(chunk.text, scored["aggregate"])
            for flag in risk_flags:
                risk_counts[flag] = risk_counts.get(flag, 0) + 1

            chunk_reports.append(
                ChunkReport(
                    chunk_id=chunk.chunk_id,
                    start_sec=chunk.start_sec,
                    end_sec=chunk.end_sec,
                    text_metrics=scored["text_metrics"],
                    av_metrics=scored["av_metrics"],
                    attention=fused["attention"],
                    risk_flags=risk_flags,
                    aggregate_score=scored["aggregate"],
                )
            )

        overall_score = round(sum(aggregate_scores) / max(1, len(aggregate_scores)), 2)
        confidence_score = round(sum(confidence_scores) / max(1, len(confidence_scores)), 2)

        sorted_risks = [
            risk for risk, _count in sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        feedback = build_feedback(
            overall_score=overall_score,
            text_avg=sum(text_scores) / max(1, len(text_scores)),
            av_avg=sum(av_scores) / max(1, len(av_scores)),
            top_risks=sorted_risks,
        )

        if overall_score >= 8.0:
            investment_band = "high-potential"
        elif overall_score >= 6.0:
            investment_band = "watchlist"
        else:
            investment_band = "early-risk"

        language_detected = max(set(language_predictions), key=language_predictions.count)

        averaged_modalities = {
            key: (value / max(1, len(chunk_reports))) for key, value in modality_totals.items()
        }

        averaged_metric_scores: dict[str, list[float]] = {}
        for metric in all_quantitative_scores:
            averaged_metric_scores.setdefault(metric["name"], []).append(metric["score"])
        metric_rows = [
            {"name": name, "score": sum(values) / len(values)}
            for name, values in averaged_metric_scores.items()
        ]

        dashboard_raw = build_investor_dashboard(
            quantitative_scores=metric_rows,
            modality_attention=averaged_modalities,
            risk_counts=risk_counts,
        )

        dashboard = InvestorDashboard(
            quantitative_scores=[DashboardSeriesPoint(**item) for item in dashboard_raw["quantitative_scores"]],
            modality_weights=[DashboardSeriesPoint(**item) for item in dashboard_raw["modality_weights"]],
            risk_distribution=[DashboardSeriesPoint(**item) for item in dashboard_raw["risk_distribution"]],
        )

        summary = EvaluationSummary(
            overall_score=overall_score,
            confidence_score=confidence_score,
            investment_band=investment_band,
            language_detected=language_detected,
            strengths=feedback["strengths"],
            weaknesses=feedback["weaknesses"],
            suggestions=feedback["suggestions"],
        )

        return EvaluationResponse(
            request_id=request_id,
            summary=summary,
            chunk_reports=chunk_reports,
            dashboard=dashboard,
        )
