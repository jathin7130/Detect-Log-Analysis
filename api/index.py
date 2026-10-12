"""Vercel serverless entrypoint. Vercel's Python runtime looks for an ASGI
`app` object in files under api/ -- this just re-exports the real FastAPI
app from webapp/app.py, which only reads pre-computed JSON (no ML deps
needed at request time)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.app import app  # noqa: E402
