# Frontend Deployment

This folder is a standalone static frontend.

## Local run

```bash
cd frontend
python server.py
```

## Hugging Face Spaces

Use the Docker SDK and set the start command to:

```bash
python server.py
```

The app listens on port `7860` by default so Spaces can route traffic correctly.
