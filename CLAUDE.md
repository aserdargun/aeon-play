# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

aeon-play is an interactive time series machine learning playground built with Plotly Dash and powered by the aeon library. It provides a web-based educational interface for learning time series ML tasks (classification, regression, clustering, forecasting, anomaly detection, segmentation, similarity search).

## Common Commands

```bash
# Development setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run the application (starts at http://localhost:8050)
python -m aeon_play.app

# Run tests
pytest                          # all tests
pytest tests/test_converters.py -v  # single file
pytest --cov=aeon_play          # with coverage

# Linting and formatting
black src/ tests/
ruff check src/ tests/

# Docker
docker-compose up --build

# Production
gunicorn --bind 0.0.0.0:8050 --workers 4 --threads 2 aeon_play.app:server
```

## Architecture

### Multi-Page Dash Application
The app uses Dash's multi-page routing (`use_pages=True`). Pages in `src/aeon_play/pages/` are auto-registered. Each page module exports a `layout` and uses `dash.register_page(__name__, path="/...", name="...")`.

### Global State Management
Three stores defined in `app.py` manage cross-page state:
- `user-level` (localStorage): "beginner", "intermediate", or "advanced" - controls progressive disclosure
- `session-datasets` (sessionStorage): loaded dataset cache
- `current-dataset-key` (sessionStorage): active dataset reference

### Service Layer (`services/`)
- **DatasetService** (`datasets.py`): Loads aeon built-in datasets and custom uploads with caching
- **AeonDiscovery** (`aeon_discovery.py`): Introspects aeon for estimators, filters by type/level. Contains `BEGINNER_*` curated lists
- **DataConverter** (`converters.py`): Converts between pandas, numpy, and aeon data formats
- **FormGenerator** (`forms.py`): Dynamically generates parameter input forms from estimator signatures
- **TaskRunner** (`runners.py`): Executes ML tasks with cross-validation and metrics
- **CodeExporter** (`exporters.py`): Generates reproducible Python code snippets

### Component Layer (`components/`)
Reusable UI components:
- `level_gate`: Shows/hides content based on user expertise level
- `learning_section`: "What you're learning" educational blocks
- `param_form`: Dynamic estimator parameter forms
- `data_preview`: Dataset visualization widgets
- `metrics_display`: Results and performance metrics
- `code_export`: Modal for copying generated Python code

### Progressive Disclosure Pattern
The UI adapts based on the selected level (Beginner/Intermediate/Advanced):
- Beginner: Curated algorithms, simple presets, guided workflows
- Intermediate: Custom uploads, cross-validation options, pipelines
- Advanced: All estimators, benchmarking, full parameter control

Components check `user-level` store and use `level_gate()` to conditionally render features.

### Background Callbacks
Long-running ML tasks use Dash's `DiskcacheManager` for background callbacks. Configured in `app.py` with cache directory from `AEON_PLAY_CACHE_DIR` environment variable.

## Adding New Features

### New Task Page
Create `src/aeon_play/pages/tasks/my_task.py`:
```python
import dash
from dash import html, dcc, callback
dash.register_page(__name__, path="/tasks/my-task", name="My Task")
layout = html.Div([...])
```

### New Estimator Type
Add to `services/aeon_discovery.py`:
1. Add type filter in `TYPE_FILTERS` dict
2. Add beginner-friendly estimators to `BEGINNER_*` lists

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8050` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `AEON_PLAY_DEBUG` | `false` | Debug mode |
| `AEON_PLAY_CACHE_DIR` | `./.cache` | Cache directory |
| `AEON_PLAY_MAX_CASES` | `10000` | Max dataset cases |
| `AEON_PLAY_MAX_TIMEPOINTS` | `5000` | Max timepoints |
