# Dataset Tools

## prepare_videos_with_whisper.py

Batch-transcribes videos locally using faster-whisper (no external API key) and exports two JSONL files:

- PitchInput JSONL for API/CLI inference pipelines
- Labeling JSONL (`video_id`, `transcript`, `slide_text`) for Option A rubric scoring

### Run

```powershell
python dataset_tools/prepare_videos_with_whisper.py --input-dir backend/outputs/batch_input
```

### Common flags

```powershell
python dataset_tools/prepare_videos_with_whisper.py `
  --input-dir backend/outputs/batch_input `
  --output-pitch-inputs backend/outputs/prepared/pitch_inputs.jsonl `
  --output-labeling backend/datasets/splits/labeling_input.jsonl `
  --model-size small `
  --device cpu `
  --compute-type int8
```

### Notes

- Install dependencies first: `pip install -r backend/requirements.txt`
- Supports common video formats: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`
- If video duration metadata cannot be read, script uses `--default-duration-sec` (default 60)
- `labeling_input.jsonl` can be fed to your LLM rubric prompt workflow to generate scored train/val/test JSONL files
