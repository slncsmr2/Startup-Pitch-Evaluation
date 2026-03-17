# Startup-Pitch-Evaluation

Multimodal AI backend starter project for evaluating startup pitch quality using text, visual, and audio feature pipelines.

## What this project includes

- FastAPI service with evaluation endpoint
- Batch evaluation endpoint for multiple pitches
- Modular pipeline matching your architecture:
- Input preprocessing and temporal chunk synchronization (5s windows)
- Parallel feature extraction for text (NLP), visual (CV), and audio (DSP) signals
- Fusion and hierarchical scoring heads (10 quantitative metrics)
- Explainability outputs with modality attention weights per chunk
- Heuristic risk flag detection per chunk
- Automated strengths, weaknesses, suggestions, and dashboard-ready series output
- Unit test for end-to-end pipeline execution

## Architecture coverage (diagram to code)

- Input and preprocessing:
- Video, slides, and user details accepted through request schema
- Temporal synchronizer and segmenter in 5-second chunks
- Feature extraction (parallel):
- Text path: speech text input, EN/TA language detection, bilingual normalization, text embeddings
- Visual path: frame-level proxy extraction, delivery and confidence signals, visual embeddings
- Audio path: waveform/prosody proxy extraction, voice pace and prosody signals, audio embeddings
- Advanced fusion and scoring:
- Cross-modal weighted fusion (text/visual/audio attention weights)
- 10 scoring heads: 6 text-focused + 4 AV-focused metrics, each 0-10
- Output and reporting:
- Weighted aggregate score, investment band, risk flags, and automated feedback
- Investor dashboard payload for charts (metric series, modality weights, risk distribution)

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
	tests/test_pipeline.py
	requirements.txt
```

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies.
3. Run the API.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API will be available at:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

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
