import hashlib


class VisualEncoder:
    """Visual encoder proxy for chunk-level delivery signals."""

    def __init__(self, embedding_dim: int = 24) -> None:
        self.embedding_dim = embedding_dim

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
        text = slide_context if slide_context else "no-slides"
        density = min(len(text.split()), 120) / 120.0
        stage_bonus = 0.8 if user_stage.lower() in {"seed", "series-a", "series a"} else 0.3

        frame_count = float(getattr(video_metadata, "frame_count", 5))
        start_sec = float(getattr(video_metadata, "start_sec", chunk_id * 5))
        end_sec = float(getattr(video_metadata, "end_sec", (chunk_id + 1) * 5))
        extraction_status = str(getattr(video_metadata, "extraction_status", "skipped"))
        face_ratio = float(getattr(video_metadata, "face_ratio", 0.0))
        motion_score = float(getattr(video_metadata, "motion_score", 0.0))
        eye_contact_score = float(getattr(video_metadata, "eye_contact_score", 0.0))
        pose_ratio = float(getattr(video_metadata, "pose_ratio", 0.0))
        gesture_energy = float(getattr(video_metadata, "gesture_energy", 0.0))
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
