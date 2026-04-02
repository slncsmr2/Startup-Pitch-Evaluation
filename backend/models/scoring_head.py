import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class ScoringHead:
    """Multi-output scoring head that maps modality outputs into final scores."""

    def __init__(self, fused_dim=256):
        self.use_heuristic = settings.use_heuristic_pipeline
        self.fused_dim = fused_dim
        self._torch = None
        self._device = "cpu"
        
        self.metric_names_text = [
            "Problem Clarity",
            "Market Opportunity",
            "Solution Uniqueness",
            "Traction Evidence",
            "Business Model Strength",
            "Team Readiness",
        ]
        
        self.metric_names_av = [
            "Delivery Clarity",
            "Presenter Confidence",
            "Voice Pace",
            "Voice Prosody",
        ]

        if not self.use_heuristic:
            self._initialize_neural_backend()
            
    def _initialize_neural_backend(self) -> None:
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            requested = settings.nn_device.strip().lower()
            use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda", "gpu"}
            self._device = "cuda" if use_cuda else "cpu"
            
            class InternalScoringHead(nn.Module):
                def __init__(self, fused_dim=256):
                    super().__init__()
                    self.metrics_net = nn.Sequential(
                        nn.Linear(fused_dim, 128),
                        nn.LayerNorm(128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        nn.Sigmoid()
                    )
                    self.overall_net = nn.Sequential(
                        nn.Linear(10, 1),
                        nn.Sigmoid()
                    )

                def forward(self, fused_vector):
                    metrics_out = self.metrics_net(fused_vector) * 10.0
                    overall_out = self.overall_net(metrics_out) * 10.0
                    return metrics_out, overall_out

            self._internal_model = InternalScoringHead(self.fused_dim)
            self._internal_model.to(self._device)
            self._internal_model.eval()

            if settings.nn_checkpoint_path and os.path.exists(settings.nn_checkpoint_path):
                try:
                    state_dict = torch.load(settings.nn_checkpoint_path, map_location=self._device)
                    if "scoring_head" in state_dict:
                        self._internal_model.load_state_dict(state_dict["scoring_head"])
                    else:
                        self._internal_model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint from {settings.nn_checkpoint_path}: {e}")
        except ImportError:
            logger.warning("Neural scoring disabled. Enable by installing torch.")
            self.use_heuristic = True

    @staticmethod
    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / max(1, len(values))

    @staticmethod
    def _clamp_10(value: float) -> float:
        return max(0.0, min(10.0, value))

    def infer(self, text_features: dict, visual_features: dict, audio_features: dict, fused: dict) -> dict:
        if self.use_heuristic or len(fused["vector"]) == 24 or self._torch is None:
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
        else:
            with self._torch.no_grad():
                f_tensor = self._torch.tensor(fused["vector"], dtype=self._torch.float32, device=self._device).unsqueeze(0)
                metrics_out, overall_out = self._internal_model(f_tensor)
                
                m_vals = metrics_out.squeeze(0).tolist()
                overall_val = overall_out.squeeze(0).item()
                
                text_scores = {name: float(m_vals[i]) for i, name in enumerate(self.metric_names_text)}
                av_scores = {name: float(m_vals[i + len(self.metric_names_text)]) for i, name in enumerate(self.metric_names_av)}
                
                text_avg = self._avg(list(text_scores.values()))
                av_avg = self._avg(list(av_scores.values()))
                confidence = self._clamp_10((av_avg * 0.5) + (text_avg * 0.5))

                return {
                    "text_scores": text_scores,
                    "av_scores": av_scores,
                    "aggregate": round(float(overall_val), 2),
                    "confidence": round(confidence, 2),
                }
