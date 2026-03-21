"""
Video processing module for frame extraction and visual feature persistence.

Handles per-5-second chunk frame extraction from video inputs.
Provides local frame extraction and lightweight visual signals with safe fallbacks.
"""

import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractMetadata:
    """Metadata for extracted frames from a chunk."""
    chunk_id: int
    start_sec: int
    end_sec: int
    frame_count: int
    frame_dir: str  # Directory where frames would be persisted
    frame_hash: str  # Deterministic hash for consistency
    extraction_status: str  # "success", "missing-video", "backend-unavailable", "skipped"
    face_ratio: float = 0.0  # Approximate face presence ratio across sampled frames
    motion_score: float = 0.0  # Approximate frame-to-frame motion score (0-1)
    eye_contact_score: float = 0.0  # Approximate eye-contact score (0-1)
    pose_ratio: float = 0.0  # Pose detection ratio across sampled frames
    gesture_energy: float = 0.0  # Relative hand/arm movement score (0-1)


class VideoProcessor:
    """
    Extracts frames from video per 5-second chunk.
    
    Design:
    - Extracts evenly sampled frames per chunk
    - Computes lightweight face/motion cues for scoring
    - Falls back to deterministic metadata when dependencies/media are unavailable
    """

    def __init__(self, frame_extraction_enabled: bool = False, output_root: str = "outputs/frames"):
        """
        Initialize video processor.
        
        Args:
            frame_extraction_enabled: If False, generates metadata only (for non-ML phases).
                If True, extracts frames using OpenCV.
        """
        self.frame_extraction_enabled = frame_extraction_enabled
        self.output_root = Path(output_root)
        logger.info(f"VideoProcessor initialized | frame_extraction_enabled={frame_extraction_enabled}")

    def extract_frames_for_chunk(
        self,
        video_file_path: str,
        chunk_id: int,
        start_sec: int,
        end_sec: int,
        frames_per_chunk: int = 5,
    ) -> FrameExtractMetadata:
        """
        Extract frames from chunk window.
        
        Args:
            video_file_path: Path to video file
            chunk_id: Chunk identifier
            start_sec: Chunk start time (seconds)
            end_sec: Chunk end time (seconds)
            frames_per_chunk: Number of evenly-spaced frames to extract
        
        Returns:
            FrameExtractMetadata with extraction status and locations
        """
        frame_hash = self._compute_frame_hash(video_file_path, chunk_id, start_sec, end_sec)
        frame_dir = self._build_frame_dir(video_file_path, chunk_id)

        if not self.frame_extraction_enabled:
            logger.debug("Frame extraction skipped for chunk %s (not enabled)", chunk_id)
            return self._build_metadata(
                chunk_id=chunk_id,
                start_sec=start_sec,
                end_sec=end_sec,
                frame_count=frames_per_chunk,
                frame_dir=str(frame_dir),
                frame_hash=frame_hash,
                extraction_status="skipped",
            )

        if not Path(video_file_path).is_file():
            logger.warning("Frame extraction skipped; video missing: %s", video_file_path)
            return self._build_metadata(
                chunk_id=chunk_id,
                start_sec=start_sec,
                end_sec=end_sec,
                frame_count=0,
                frame_dir=str(frame_dir),
                frame_hash=frame_hash,
                extraction_status="missing-video",
            )

        cv2, np, cv_error = self._import_cv_backends()
        if cv2 is None or np is None:
            logger.warning("OpenCV/Numpy unavailable for frame extraction: %s", cv_error)
            return self._build_metadata(
                chunk_id=chunk_id,
                start_sec=start_sec,
                end_sec=end_sec,
                frame_count=0,
                frame_dir=str(frame_dir),
                frame_hash=frame_hash,
                extraction_status="backend-unavailable",
            )

        mediapipe_ready, mp_face_mesh, mp_pose = self._init_mediapipe_models()

        frame_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(video_file_path)
        if not capture.isOpened():
            logger.warning("Unable to open video for frame extraction: %s", video_file_path)
            return self._build_metadata(
                chunk_id=chunk_id,
                start_sec=start_sec,
                end_sec=end_sec,
                frame_count=0,
                frame_dir=str(frame_dir),
                frame_hash=frame_hash,
                extraction_status="missing-video",
            )

        metrics = self._extract_frame_metrics(
            capture=capture,
            frame_dir=frame_dir,
            start_sec=start_sec,
            end_sec=end_sec,
            frames_per_chunk=frames_per_chunk,
            cv2=cv2,
            np=np,
            mediapipe_ready=mediapipe_ready,
            mp_face_mesh=mp_face_mesh,
            mp_pose=mp_pose,
        )
        status = "success" if metrics["extracted"] > 0 else "backend-unavailable"
        return self._build_metadata(
            chunk_id=chunk_id,
            start_sec=start_sec,
            end_sec=end_sec,
            frame_count=metrics["extracted"],
            frame_dir=str(frame_dir),
            frame_hash=frame_hash,
            extraction_status=status,
            face_ratio=metrics["face_ratio"],
            motion_score=metrics["motion_score"],
            eye_contact_score=metrics["eye_contact_score"],
            pose_ratio=metrics["pose_ratio"],
            gesture_energy=metrics["gesture_energy"],
        )

    @staticmethod
    def _import_cv_backends():
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            return cv2, np, None
        except Exception as exc:
            return None, None, exc

    @staticmethod
    def _build_metadata(
        chunk_id: int,
        start_sec: int,
        end_sec: int,
        frame_count: int,
        frame_dir: str,
        frame_hash: str,
        extraction_status: str,
        face_ratio: float = 0.0,
        motion_score: float = 0.0,
        eye_contact_score: float = 0.0,
        pose_ratio: float = 0.0,
        gesture_energy: float = 0.0,
    ) -> FrameExtractMetadata:
        return FrameExtractMetadata(
            chunk_id=chunk_id,
            start_sec=start_sec,
            end_sec=end_sec,
            frame_count=frame_count,
            frame_dir=frame_dir,
            frame_hash=frame_hash,
            extraction_status=extraction_status,
            face_ratio=round(max(0.0, min(1.0, face_ratio)), 3),
            motion_score=round(max(0.0, min(1.0, motion_score)), 3),
            eye_contact_score=round(max(0.0, min(1.0, eye_contact_score)), 3),
            pose_ratio=round(max(0.0, min(1.0, pose_ratio)), 3),
            gesture_energy=round(max(0.0, min(1.0, gesture_energy)), 3),
        )

    @staticmethod
    def _init_mediapipe_models():
        try:
            import mediapipe as mp  # type: ignore

            mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
            mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
            return True, mp_face_mesh, mp_pose
        except Exception as exc:
            logger.info("MediaPipe unavailable, using OpenCV-only visual metrics: %s", exc)
            return False, None, None

    def _extract_frame_metrics(
        self,
        capture,
        frame_dir: Path,
        start_sec: int,
        end_sec: int,
        frames_per_chunk: int,
        cv2,
        np,
        mediapipe_ready: bool,
        mp_face_mesh,
        mp_pose,
    ) -> dict[str, float | int]:
        timestamps = self._sample_timestamps(start_sec=start_sec, end_sec=end_sec, count=frames_per_chunk)
        cascade = self._load_face_detector(cv2)
        extracted = 0
        face_hits = 0
        motion_sum = 0.0
        eye_contact_sum = 0.0
        pose_hits = 0
        gesture_sum = 0.0
        previous_gray = None

        try:
            for idx, ts in enumerate(timestamps):
                frame = self._read_frame(capture, cv2, ts)
                if frame is None:
                    continue

                extracted += 1
                gray = self._save_and_grayscale_frame(cv2, frame, frame_dir, idx)

                if cascade is not None:
                    face_hits += self._detect_face_hit(cascade, gray)

                if mediapipe_ready:
                    mp_values = self._analyze_mediapipe_frame(cv2, frame, mp_face_mesh, mp_pose)
                    face_hits += mp_values["face_hits"]
                    eye_contact_sum += mp_values["eye_contact"]
                    pose_hits += mp_values["pose_hits"]
                    gesture_sum += mp_values["gesture"]

                motion_sum += self._compute_motion_delta(cv2, np, previous_gray, gray)
                previous_gray = gray
        finally:
            capture.release()
            if mp_face_mesh is not None:
                mp_face_mesh.close()
            if mp_pose is not None:
                mp_pose.close()

        return {
            "extracted": extracted,
            "face_ratio": (face_hits / extracted) if extracted else 0.0,
            "motion_score": (motion_sum / max(1, extracted - 1)) if extracted > 1 else 0.0,
            "eye_contact_score": (eye_contact_sum / extracted) if extracted else 0.0,
            "pose_ratio": (pose_hits / extracted) if extracted else 0.0,
            "gesture_energy": (gesture_sum / max(1, pose_hits)) if pose_hits > 0 else 0.0,
        }

    @staticmethod
    def _read_frame(capture, cv2, timestamp: float):
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp * 1000.0))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return frame

    @staticmethod
    def _save_and_grayscale_frame(cv2, frame, frame_dir: Path, frame_idx: int):
        cv2.imwrite(str(frame_dir / f"frame_{frame_idx:02d}.jpg"), frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _detect_face_hit(cascade, gray) -> int:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        return 1 if len(faces) > 0 else 0

    @staticmethod
    def _analyze_mediapipe_frame(cv2, frame, mp_face_mesh, mp_pose) -> dict[str, float | int]:
        if mp_face_mesh is None or mp_pose is None:
            return {"face_hits": 0, "eye_contact": 0.0, "pose_hits": 0, "gesture": 0.0}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_hits = 0
        eye_contact = 0.0
        pose_hits = 0
        gesture = 0.0

        face_result = mp_face_mesh.process(frame_rgb)
        if face_result.multi_face_landmarks:
            face_hits = 1
            face_landmarks = face_result.multi_face_landmarks[0].landmark
            nose_x = float(face_landmarks[1].x)
            eye_contact = max(0.0, 1.0 - abs(0.5 - nose_x) * 2.0)

        pose_result = mp_pose.process(frame_rgb)
        if pose_result.pose_landmarks:
            pose_hits = 1
            lm = pose_result.pose_landmarks.landmark
            left_move = abs(float(lm[15].y) - float(lm[11].y))
            right_move = abs(float(lm[16].y) - float(lm[12].y))
            gesture = min(1.0, (left_move + right_move) / 2.0)

        return {
            "face_hits": face_hits,
            "eye_contact": eye_contact,
            "pose_hits": pose_hits,
            "gesture": gesture,
        }

    @staticmethod
    def _compute_motion_delta(cv2, np, previous_gray, gray) -> float:
        if previous_gray is None:
            return 0.0
        diff = cv2.absdiff(gray, previous_gray)
        return float(np.mean(diff) / 255.0)

    def _compute_frame_hash(self, video_file: str, chunk_id: int, start_sec: int, end_sec: int) -> str:
        """Deterministic hash for frame metadata consistency."""
        data = f"{video_file}|{chunk_id}|{start_sec}|{end_sec}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def _build_frame_dir(self, video_file: str, chunk_id: int) -> Path:
        safe_video = Path(video_file).name.replace(".", "_")
        return self.output_root / safe_video / f"chunk_{chunk_id:04d}"

    @staticmethod
    def _sample_timestamps(start_sec: int, end_sec: int, count: int) -> list[float]:
        if count <= 1:
            return [float(start_sec)]
        duration = max(0.1, float(end_sec - start_sec))
        step = duration / count
        return [float(start_sec) + (i + 0.5) * step for i in range(count)]

    @staticmethod
    def _load_face_detector(cv2_module):
        cascade_path = Path(cv2_module.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None
        detector = cv2_module.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return None
        return detector
