import hashlib
import logging
from pathlib import Path
from app.core.config import settings


logger = logging.getLogger(__name__)


class VisualEncoder:
    """Visual encoder proxy for chunk-level delivery signals."""

    def __init__(
        self,
        embedding_dim: int = 24,
        use_heuristic: bool = True,
        backbone_name: str = "mobilenet_v3_small",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.use_heuristic = use_heuristic
        self.backbone_name = backbone_name.strip().lower()

        self._torch = None
        self._torchvision = None
        self._weights = None
        self._backbone = None
        self._preprocess = None
        self._projection = None
        self._scoring_mlp = None
        self._device = "cpu"

        if not self.use_heuristic:
            self._initialize_neural_backend()

    def _initialize_neural_backend(self) -> None:
        if self.backbone_name != "mobilenet_v3_small":
            logger.warning("Unsupported visual backbone '%s'; defaulting to mobilenet_v3_small", self.backbone_name)
            self.backbone_name = "mobilenet_v3_small"

        try:
            import torch  # type: ignore
            import torchvision  # type: ignore
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small  # type: ignore
        except Exception as exc:
            logger.warning("Neural visual encoder disabled (torch/torchvision unavailable): %s", exc)
            return

        self._torch = torch
        self._torchvision = torchvision
        requested = settings.nn_device.strip().lower()
        use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda", "gpu"}
        self._device = "cuda" if use_cuda else "cpu"

        weights = None
        try:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = mobilenet_v3_small(weights=weights)
        except Exception as exc:
            logger.warning("Pretrained MobileNetV3 unavailable; using random init backbone: %s", exc)
            model = mobilenet_v3_small(weights=None)

        self._weights = weights
        self._backbone = model.features
        self._preprocess = weights.transforms() if weights is not None else None
        self._projection = torch.nn.Linear(576, 256)
        self._scoring_mlp = torch.nn.Sequential(
            torch.nn.Linear(256 + 5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Sigmoid(),
        )

        torch.manual_seed(17)
        self.embedding_dim = 256
        self._backbone.to(self._device)
        self._projection.to(self._device)
        self._scoring_mlp.to(self._device)
        self._backbone.eval()
        self._projection.eval()
        self._scoring_mlp.eval()

    def _neural_backend_ready(self) -> bool:
        return all(
            component is not None
            for component in [self._torch, self._backbone, self._projection, self._scoring_mlp]
        )

    def _extract_aux_features(self, video_metadata: object | None) -> list[float]:
        return [
            float(getattr(video_metadata, "face_ratio", 0.0)),
            float(getattr(video_metadata, "eye_contact_score", 0.0)),
            float(getattr(video_metadata, "pose_ratio", 0.0)),
            float(getattr(video_metadata, "gesture_energy", 0.0)),
            float(getattr(video_metadata, "motion_score", 0.0)),
        ]

    def _load_frame_tensors(self, frame_dir: str) -> list:
        path = Path(frame_dir)
        if not path.is_dir():
            return []

        frame_files = sorted(path.glob("*.jpg"))
        if not frame_files:
            return []

        transforms = self._torchvision.transforms
        to_float = transforms.ConvertImageDtype(self._torch.float32)
        resize = transforms.Resize((224, 224), antialias=True)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        tensors = []
        for frame_path in frame_files:
            try:
                image = self._torchvision.io.read_image(str(frame_path))
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                elif image.shape[0] > 3:
                    image = image[:3, :, :]
                image = normalize(resize(to_float(image)))
                tensors.append(image)
            except Exception as exc:
                logger.debug("Skipping unreadable frame %s: %s", frame_path, exc)
        return tensors

    def _neural_infer(self, frame_tensors: list, aux_features: list[float]) -> dict:
        with self._torch.no_grad():
            batch = self._torch.stack(frame_tensors, dim=0).to(self._device)
            features = self._backbone(batch)
            pooled = self._torch.nn.functional.adaptive_avg_pool2d(features, output_size=1).flatten(1)
            chunk_feature = pooled.mean(dim=0)

            embedding = self._projection(chunk_feature)
            aux = self._torch.tensor(aux_features, dtype=self._torch.float32, device=self._device)
            scoring_input = self._torch.cat([embedding, aux], dim=0)
            scores = self._scoring_mlp(scoring_input.unsqueeze(0)).squeeze(0)

            delivery_clarity = float(scores[0].item() * 10.0)
            presenter_confidence = float(scores[1].item() * 10.0)
            vector = embedding.detach().cpu().numpy().astype("float32").tolist()

        return {
            "embedding": vector,
            "delivery_clarity": self._clamp_10(delivery_clarity),
            "presenter_confidence": self._clamp_10(presenter_confidence),
        }

    def _heuristic_infer(
        self,
        slide_context: str,
        chunk_id: int,
        user_stage: str,
        frame_count: float,
        start_sec: float,
        end_sec: float,
        extraction_status: str,
        face_ratio: float,
        motion_score: float,
        eye_contact_score: float,
        pose_ratio: float,
        gesture_energy: float,
    ) -> dict:
        text = slide_context if slide_context else "no-slides"
        density = min(len(text.split()), 120) / 120.0
        stage_bonus = 0.8 if user_stage.lower() in {"seed", "series-a", "series a"} else 0.3

        duration = max(1.0, end_sec - start_sec)
        frame_coverage = min(2.0, frame_count / duration)
        if extraction_status == "success":
            extraction_bonus = 0.8
        elif extraction_status == "pending":
            extraction_bonus = 0.35
        else:
            extraction_bonus = 0.0

        missing_visual_signal = extraction_status in {"missing-video", "backend-unavailable", "skipped"}
        if missing_visual_signal:
            delivery_clarity = self._clamp_10(5.1 + density * 1.5 + stage_bonus * 0.2)
            presenter_confidence = self._clamp_10(5.0 + density * 1.0 + stage_bonus * 0.5)
        else:
            delivery_clarity = self._clamp_10(
                3.8
                + density * 2.8
                + frame_coverage * 0.9
                + motion_score * 0.6
                + pose_ratio * 1.2
                + gesture_energy * 0.7
                + extraction_bonus
            )
            presenter_confidence = self._clamp_10(
                4.0
                + density * 1.6
                + stage_bonus
                + extraction_bonus
                + face_ratio * 1.0
                + eye_contact_score * 1.5
                + pose_ratio * 1.0
                + gesture_energy * 0.8
            )

        return {
            "embedding": self._hash_to_vector(
                f"visual::{chunk_id}::{text}::{user_stage}::frames={frame_count}::face={face_ratio:.3f}::eye={eye_contact_score:.3f}::pose={pose_ratio:.3f}::gesture={gesture_energy:.3f}::motion={motion_score:.3f}::status={extraction_status}"
            ),
            "delivery_clarity": delivery_clarity,
            "presenter_confidence": presenter_confidence,
        }

    def _hash_to_vector(self, value: str) -> list[float]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        values = [b / 255.0 for b in digest]
        if self.embedding_dim <= len(values):
            return values[: self.embedding_dim]
        repeats = (self.embedding_dim // len(values)) + 1
        return (values * repeats)[: self.embedding_dim]

    @staticmethod
    def _clamp_10(value: float) -> float:
        return max(0.0, min(10.0, value))

    def infer(self, slide_context: str, chunk_id: int, user_stage: str, video_metadata: object | None = None) -> dict:
        frame_count = float(getattr(video_metadata, "frame_count", 5))
        start_sec = float(getattr(video_metadata, "start_sec", chunk_id * 5))
        end_sec = float(getattr(video_metadata, "end_sec", (chunk_id + 1) * 5))
        extraction_status = str(getattr(video_metadata, "extraction_status", "skipped"))
        face_ratio = float(getattr(video_metadata, "face_ratio", 0.0))
        motion_score = float(getattr(video_metadata, "motion_score", 0.0))
        eye_contact_score = float(getattr(video_metadata, "eye_contact_score", 0.0))
        pose_ratio = float(getattr(video_metadata, "pose_ratio", 0.0))
        gesture_energy = float(getattr(video_metadata, "gesture_energy", 0.0))

        if self.use_heuristic:
            return self._heuristic_infer(
                slide_context=slide_context,
                chunk_id=chunk_id,
                user_stage=user_stage,
                frame_count=frame_count,
                start_sec=start_sec,
                end_sec=end_sec,
                extraction_status=extraction_status,
                face_ratio=face_ratio,
                motion_score=motion_score,
                eye_contact_score=eye_contact_score,
                pose_ratio=pose_ratio,
                gesture_energy=gesture_energy,
            )

        if extraction_status != "success" or not self._neural_backend_ready():
            return self._heuristic_infer(
                slide_context=slide_context,
                chunk_id=chunk_id,
                user_stage=user_stage,
                frame_count=frame_count,
                start_sec=start_sec,
                end_sec=end_sec,
                extraction_status=extraction_status,
                face_ratio=face_ratio,
                motion_score=motion_score,
                eye_contact_score=eye_contact_score,
                pose_ratio=pose_ratio,
                gesture_energy=gesture_energy,
            )

        frame_dir = str(getattr(video_metadata, "frame_dir", ""))
        frame_tensors = self._load_frame_tensors(frame_dir)
        if not frame_tensors:
            return self._heuristic_infer(
                slide_context=slide_context,
                chunk_id=chunk_id,
                user_stage=user_stage,
                frame_count=frame_count,
                start_sec=start_sec,
                end_sec=end_sec,
                extraction_status=extraction_status,
                face_ratio=face_ratio,
                motion_score=motion_score,
                eye_contact_score=eye_contact_score,
                pose_ratio=pose_ratio,
                gesture_energy=gesture_energy,
            )

        try:
            aux_features = self._extract_aux_features(video_metadata)
            return self._neural_infer(frame_tensors, aux_features)
        except Exception as exc:
            logger.warning("Neural visual inference failed; falling back to heuristics: %s", exc)
            return self._heuristic_infer(
                slide_context=slide_context,
                chunk_id=chunk_id,
                user_stage=user_stage,
                frame_count=frame_count,
                start_sec=start_sec,
                end_sec=end_sec,
                extraction_status=extraction_status,
                face_ratio=face_ratio,
                motion_score=motion_score,
                eye_contact_score=eye_contact_score,
                pose_ratio=pose_ratio,
                gesture_energy=gesture_energy,
            )
