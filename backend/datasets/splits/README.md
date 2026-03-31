# Phase 6 Option A Dataset Format

Use JSONL files named `train.jsonl`, `val.jsonl`, and `test.jsonl` in this folder.

Each line must be one JSON object with this schema:

```json
{
  "video_id": "yc_pitch_001",
  "transcript": "Founder narration text...",
  "slide_text": "Optional slide OCR text...",
  "scores": [7.8, 8.1, 7.4, 6.9, 7.2, 8.0, 7.0, 7.5, 6.8, 7.1],
  "investment_band": "watchlist"
}
```

Rules:

- `scores` must contain 10 values in the same metric order used by inference.
- Score values are expected on a 0-10 scale.
- `investment_band` is optional. If missing, it is derived from average score:
  - `high-potential` for overall >= 7.5
  - `watchlist` for overall >= 5.0 and < 7.5
  - `early-risk` for overall < 5.0
- Legacy rows using `features` + `metric_targets` are still accepted for compatibility.

Recommended Option A workflow:

1. Collect real pitch transcripts and optional slide text.
2. Use an LLM rubric prompt to generate the 10 scores.
3. Spot-check a sample manually for label quality.
4. Save records to `train.jsonl` / `val.jsonl` / `test.jsonl`.
5. Run training with `python scripts/train.py --config models/config/training_cpu.yaml`.
