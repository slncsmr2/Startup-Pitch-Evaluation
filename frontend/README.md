# Frontend Deployment

This folder is a standalone static frontend for Vercel.

## Vercel

Set the project root to this folder and deploy it as a static site.

Do not add a Python `requirements.txt` here. That file makes Vercel try to treat the frontend as a Python app.

Use the site like this:

```text
https://your-vercel-site.vercel.app/?api=https://your-ngrok-url
```

## Local run

Open `index.html` directly or serve the folder with any static server.
