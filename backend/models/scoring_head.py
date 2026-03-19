class ScoringHead:
    """Multi-output scoring head that maps modality outputs into final scores."""

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    @staticmethod
    def _clamp_10(value: float) -> float:
        return max(0.0, min(10.0, value))

    def infer(self, text_features: dict, visual_features: dict, audio_features: dict, fused: dict) -> dict:
        text_scores = {
            "Problem Clarity": float(text_features["problem_clarity"]),
            "Market Opportunity": float(text_features["market_opportunity"]),
            "Solution Uniqueness": float(text_features["solution_uniqueness"]),
            "Traction Evidence": float(text_features["traction_evidence"]),
            "Business Model Strength": float(text_features["business_model_strength"]),
            "Team Readiness": float(text_features["team_readiness"]),
        }
        av_scores = {
            "Delivery Clarity": float(visual_features["delivery_clarity"]),
            "Presenter Confidence": float(visual_features["presenter_confidence"]),
            "Voice Pace": float(audio_features["voice_pace"]),
            "Voice Prosody": float(audio_features["prosody"]),
        }

        text_avg = self._avg(list(text_scores.values()))
        av_avg = self._avg(list(av_scores.values()))
        fusion_signal = self._avg(fused["vector"]) * 10.0

        aggregate = self._clamp_10((text_avg * 0.5) + (av_avg * 0.35) + (fusion_signal * 0.15))
        confidence = self._clamp_10((av_avg * 0.5) + (text_avg * 0.5))

        return {
            "text_scores": text_scores,
            "av_scores": av_scores,
            "aggregate": round(aggregate, 2),
            "confidence": round(confidence, 2),
        }
