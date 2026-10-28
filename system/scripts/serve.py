"""Serve the shipped UI and API on one local port; no Node.js is required."""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from backend.app.main import app

frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if not (frontend_dist / "index.html").is_file():
    raise RuntimeError("缺少 frontend/dist/index.html，请在 frontend 目录执行 npm ci 和 npm run build")

# API routes are registered first. The UI uses /api/v1 on the same origin.
app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="csrag-ui")
