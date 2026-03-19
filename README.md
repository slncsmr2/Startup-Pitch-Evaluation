# Startup-Pitch-Evaluation

Multimodal AI backend for evaluating startup pitch quality using text, visual, and audio feature pipelines.

## What this project includes

- FastAPI service with evaluation and batch evaluation endpoints
- Modular pipeline with parallel feature extraction across modalities:
  - Text processing (NLP) with language detection
  - Visual analysis (CV) for delivery and confidence signals
  - Audio processing (DSP) for prosody and pace analysis
- Cross-modal fusion with attention weight tracking
- Hierarchical scoring with 10 quantitative metrics
- Risk flag detection and automated insights
- CLI and REST API with identical inference logic
- Jupyter notebook integration for interactive analysis

## Architecture

- **Input & Preprocessing**: Video, slides, and user details accepted; temporal synchronization in 5-second chunks
- **Feature Extraction** (parallel):
  - Text: Speech transcription, language detection, text embeddings
  - Visual: Frame analysis, delivery and confidence signals, visual embeddings
  - Audio: Prosody analysis, voice pace and energy signals, audio embeddings
- **Fusion & Scoring**: Cross-modal weighted fusion with 10 scoring metrics (0-10 scale)
- **Explainability**: Per-chunk visualizations with modality attention weights and risk distribution

## Project structure

```text
backend/
  app/
    core/config.py
    main.py
    pipeline.py
    schemas.py
    services/
      preprocessing.py
      extractors.py
      fusion.py
      scoring.py
      reporting.py
      audio_processor.py
      video_processor.py
      transcriber.py
  models/
    text_encoder.py
    visual_encoder.py
    audio_encoder.py
    fusion_head.py
    scoring_head.py
  tests/
    test_api.py
    test_pipeline.py
    test_transcriber.py
  scripts/
    infer_cli.py
    train.py
    evaluate.py
  requirements.txt

notebooks/
  Startup_Pitch_Evaluation.ipynb
```

## Quick start

### Running the API

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at:

- `http://127.0.0.1:8000/health` - Health check
- `http://127.0.0.1:8000/docs` - Interactive API documentation

### Running with Jupyter notebooks

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name startup-pitch-eval --display-name "Python (startup-pitch-eval)"
cd ..
jupyter lab
```

Then open `notebooks/Startup_Pitch_Evaluation.ipynb`.

## Sample evaluation request

```powershell
curl -X POST "http://127.0.0.1:8000/evaluate" ^
	-H "Content-Type: application/json" ^
	-d "{\"title\":\"Startup Example A\",\"transcript\":\"\",\"language_hint\":\"en-ta\",\"presenter_profile\":{\"experience\":\"5 years\"},\"video\":{\"file_name\":\"startup_example_a_pitch.mp4\",\"file_format\":\"mp4\",\"duration_sec\":120,\"transcript_text\":\"We solve an industry workflow problem with bilingual AI support. Our pilot improved adoption and operational outcomes.\"},\"slides\":[{\"title\":\"Problem\",\"content\":\"Current workflow creates measurable loss\"},{\"title\":\"Market\",\"content\":\"Large underserved customer segment\"},{\"title\":\"Traction\",\"content\":\"Pilot growth and repeat usage\"}],\"user_details\":{\"founder_name\":\"Founder Example\",\"startup_name\":\"Startup Example A\",\"sector\":\"Industry Segment\",\"stage\":\"Seed\"}}"
```

## Sample batch request

```powershell
curl -X POST "http://127.0.0.1:8000/evaluate/batch" ^
	-H "Content-Type: application/json" ^
	-d "{\"pitches\":[{\"title\":\"Startup Example A\",\"transcript\":\"We solve a high-frequency operational problem for a target segment.\",\"language_hint\":\"en-ta\",\"slide_text\":[\"Problem\",\"Solution\"]},{\"title\":\"Startup Example B\",\"transcript\":\"We improve planning quality and reduce waste for local businesses.\",\"language_hint\":\"en-ta\",\"slide_text\":[\"Problem\",\"Traction\"]}]}"
```

## Output highlights

- `summary.investment_band`: `high-potential`, `watchlist`, or `early-risk`
- `summary.language_detected`: detected language profile (`en`, `ta`, `ta-en`)
- `chunk_reports[].attention`: text/visual/audio contribution weights
- `chunk_reports[].risk_flags`: detected risk hints such as overclaim or weak traction evidence
- `dashboard.quantitative_scores`: 10 chart-ready metrics for investor UI

## Contributing

Contributions are welcome for feature improvements, bug fixes, tests, documentation, and model integration.

### Development setup

1. Fork the repository.
2. Clone your fork and move to the backend directory.
3. Create and activate a virtual environment.
4. Install dependencies and run tests.

```powershell
git clone <your-fork-url>
cd Startup-Pitch-Evaluation/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

### Contribution workflow

1. Create a feature branch from main.
2. Make focused changes with clear commit messages.
3. Add or update tests for all behavior changes.
4. Ensure all tests pass locally.
5. Open a pull request with a concise summary.

### Pull request checklist

- Code is readable and scoped to one logical change.
- New or changed behavior is covered by tests.
- README or docs are updated when API behavior changes.
- No sensitive data, secrets, or private files are committed.

### Coding guidelines

- Keep API schemas backward-compatible when possible.
- Prefer small, composable service functions.
- Use clear naming for scoring and reporting outputs.
- Keep sample data generic and free of personal identifiers.

## Notes

- Current feature extraction and fusion are deterministic placeholder implementations designed for fast iteration.
- You can replace individual service modules with real model inference (Whisper, wav2vec2, ViT, multimodal transformers) without changing API contracts.

## Run tests

```powershell
cd backend
pytest -q
```
