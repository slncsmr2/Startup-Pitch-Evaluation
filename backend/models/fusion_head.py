import logging
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)

class FusionHead:
    """Cross-modal fusion head with adaptive attention from modality energy."""
    def __init__(self, text_dim=384, visual_dim=256, audio_dim=128, common_dim=256):
        self.use_heuristic = settings.use_heuristic_pipeline
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.common_dim = common_dim
        
        self._torch = None
        self._attention_layer = None
        self._device = "cpu"
        self._checkpoint_loaded = False
        
        if not self.use_heuristic:
            self._initialize_neural_backend()

    @staticmethod
    def _resolve_checkpoint_path(raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate

        backend_root = Path(__file__).resolve().parents[1]
        project_root = Path(__file__).resolve().parents[2]

        backend_candidate = backend_root / candidate
        if backend_candidate.exists():
            return backend_candidate

        return project_root / candidate

    def _initialize_neural_backend(self) -> None:
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            requested = settings.nn_device.strip().lower()
            use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda", "gpu"}
            self._device = "cuda" if use_cuda else "cpu"
            
            class CrossModalAttentionLayer(nn.Module):
                def __init__(self, text_dim=384, visual_dim=256, audio_dim=128, common_dim=256):
                    super().__init__()
                    self.proj_text = nn.Linear(text_dim, common_dim)
                    self.proj_visual = nn.Linear(visual_dim, common_dim)
                    self.proj_audio = nn.Linear(audio_dim, common_dim)

                    self.attn_text = nn.Linear(common_dim, 1)
                    self.attn_visual = nn.Linear(common_dim, 1)
                    self.attn_audio = nn.Linear(common_dim, 1)

                def forward(self, text_emb, visual_emb, audio_emb):
                    t_proj = self.proj_text(text_emb)
                    v_proj = self.proj_visual(visual_emb)
                    a_proj = self.proj_audio(audio_emb)

                    s_t = self.attn_text(t_proj)
                    s_v = self.attn_visual(v_proj)
                    s_a = self.attn_audio(a_proj)

                    # For torchaudio/torch compat
                    scores = torch.cat([s_t, s_v, s_a], dim=-1)
                    attn_weights = torch.softmax(scores, dim=-1)

                    attn_t, attn_v, attn_a = attn_weights.split(1, dim=-1)

                    fused = (attn_t * t_proj) + (attn_v * v_proj) + (attn_a * a_proj)
                    return fused, attn_weights

            self._attention_layer = CrossModalAttentionLayer(
                self.text_dim, self.visual_dim, self.audio_dim, self.common_dim
            )

            if settings.nn_checkpoint_path:
                checkpoint_path = self._resolve_checkpoint_path(settings.nn_checkpoint_path)
                if checkpoint_path.exists():
                    try:
                        state_dict = torch.load(str(checkpoint_path), map_location=self._device,weights_only=True)
                        if "fusion_head" in state_dict:
                            self._attention_layer.load_state_dict(state_dict["fusion_head"], strict=False)
                            self._checkpoint_loaded = True
                    except Exception as exc:
                        logger.warning("Failed to load fusion_head checkpoint from %s: %s", checkpoint_path, exc)

            self._attention_layer.to(self._device)
            self._attention_layer.eval()
        except ImportError:
            logger.warning("Neural fusion disabled. Enable by installing torch.")
            self.use_heuristic = True

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    @staticmethod
    def _to_common_dim(values: list[float], common_dim: int) -> list[float]:
        if not values:
            return [0.0] * common_dim
        if len(values) >= common_dim:
            return values[:common_dim]
        repeats = (common_dim // len(values)) + 1
        return (values * repeats)[:common_dim]

    @staticmethod
    def _compute_attention_from_strengths(text_s: float, visual_s: float, audio_s: float) -> tuple[float, float, float]:
        strengths = [max(1e-6, text_s), max(1e-6, visual_s), max(1e-6, audio_s)]
        total = sum(strengths)
        base = [s / total for s in strengths]

        min_share = 0.15
        max_share = 0.6
        floors = [min_share, min_share, min_share]
        remaining = 1.0 - sum(floors)

        extra = [max(0.0, b - min_share) for b in base]
        extra_total = sum(extra)
        if extra_total <= 1e-9:
            return min_share, min_share, min_share + remaining

        weights = [f + (remaining * (e / extra_total)) for f, e in zip(floors, extra)]

        # Prevent a single modality from dominating fallback attention.
        max_idx = max(range(3), key=lambda i: weights[i])
        if weights[max_idx] > max_share:
            excess = weights[max_idx] - max_share
            weights[max_idx] = max_share

            other_indices = [i for i in range(3) if i != max_idx]
            slack = [max(0.0, max_share - weights[i]) for i in other_indices]
            slack_total = sum(slack)
            if slack_total <= 1e-9:
                split = excess / 2.0
                weights[other_indices[0]] += split
                weights[other_indices[1]] += split
            else:
                weights[other_indices[0]] += excess * (slack[0] / slack_total)
                weights[other_indices[1]] += excess * (slack[1] / slack_total)

        return weights[0], weights[1], weights[2]

    def _deterministic_fallback_fusion(
        self,
        text_embedding: list[float],
        visual_embedding: list[float],
        audio_embedding: list[float],
    ) -> dict:
        t_vec = self._to_common_dim(text_embedding, self.common_dim)
        v_vec = self._to_common_dim(visual_embedding, self.common_dim)
        a_vec = self._to_common_dim(audio_embedding, self.common_dim)

        text_strength = self._mean([abs(v) for v in text_embedding])
        visual_strength = self._mean([abs(v) for v in visual_embedding])
        audio_strength = self._mean([abs(v) for v in audio_embedding])

        text_w, visual_w, audio_w = self._compute_attention_from_strengths(
            text_strength,
            visual_strength,
            audio_strength,
        )

        fused_vector = [
            (text_w * t) + (visual_w * v) + (audio_w * a)
            for t, v, a in zip(t_vec, v_vec, a_vec)
        ]

        return {
            "vector": fused_vector,
            "attention": {
                "text": round(text_w, 4),
                "visual": round(visual_w, 4),
                "audio": round(audio_w, 4),
            },
        }

    def infer(self, text_embedding: list[float], visual_embedding: list[float], audio_embedding: list[float]) -> dict:
        if self.use_heuristic or len(text_embedding) == 24 or self._torch is None:
            text_energy = self._mean([abs(v) for v in text_embedding])
            visual_energy = self._mean([abs(v) for v in visual_embedding])
            audio_energy = self._mean([abs(v) for v in audio_embedding])

            text_w, visual_w, audio_w = self._compute_attention_from_strengths(
                text_energy,
                visual_energy,
                audio_energy,
            )

            fused_vector = [
                (text_w * t) + (visual_w * v) + (audio_w * a)
                for t, v, a in zip(text_embedding, visual_embedding, audio_embedding)
            ]

            return {
                "vector": fused_vector,
                "attention": {
                    "text": round(text_w, 4),
                    "visual": round(visual_w, 4),
                    "audio": round(audio_w, 4),
                },
            }

        if not self._checkpoint_loaded:
            return self._deterministic_fallback_fusion(text_embedding, visual_embedding, audio_embedding)
        
        with self._torch.no_grad():
            t_tensor = self._torch.tensor(text_embedding, dtype=self._torch.float32, device=self._device).unsqueeze(0)
            v_tensor = self._torch.tensor(visual_embedding, dtype=self._torch.float32, device=self._device).unsqueeze(0)
            a_tensor = self._torch.tensor(audio_embedding, dtype=self._torch.float32, device=self._device).unsqueeze(0)

            fused, attn_weights = self._attention_layer(t_tensor, v_tensor, a_tensor)

            return {
                "vector": fused.squeeze(0).tolist(),
                "attention": {
                    "text": round(attn_weights[0, 0].item(), 4),
                    "visual": round(attn_weights[0, 1].item(), 4),
                    "audio": round(attn_weights[0, 2].item(), 4),
                },
            }
