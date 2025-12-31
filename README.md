# aeon-play

**Interactive Time Series Machine Learning Playground**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![aeon](https://img.shields.io/badge/powered%20by-aeon-orange)](https://www.aeon-toolkit.org/)

aeon-play is a production-ready, educational web application for exploring time series machine learning. Built with [Plotly Dash](https://dash.plotly.com/) and powered by the [aeon](https://www.aeon-toolkit.org/) library, it provides an interactive environment for learning and experimenting with:

- **Classification** - Assign labels to time series
- **Regression** - Predict continuous values
- **Clustering** - Group similar series
- **Forecasting** - Predict future values
- **Anomaly Detection** - Find unusual patterns
- **Segmentation** - Detect change points
- **Similarity Search** - Find matching series

## Features

### Progressive Disclosure Learning System

Three learning levels adapt the interface to your expertise:

| Level | Features |
|-------|----------|
| **Beginner** | Curated datasets, simple algorithms, guided workflows |
| **Intermediate** | Custom uploads, cross-validation, pipelines |
| **Advanced** | All algorithms, benchmarking, code export |

### Core Modules

- **Data Studio** - Load, preview, and prepare datasets
- **Visual Lab** - Explore series with ACF, lag plots, spectrograms
- **Transformations** - Apply Catch22, ROCKET, wavelets, and more
- **Distances Lab** - Compare DTW, Euclidean, edit-based distances
- **Task Modules** - Train and evaluate models with real-time results
- **Benchmarking** - Compare multiple algorithms side-by-side

### Educational Features

- "What you're learning" sections on every page
- Parameter tooltips with explanations
- Preset buttons for quick demos
- Generate reproducible Python code

## Architecture

```
aeon-play/
├── src/aeon_play/
│   ├── app.py              # Main Dash application
│   ├── pages/              # Multi-page routing
│   │   ├── home.py         # Landing page
│   │   ├── data.py         # Data Studio
│   │   ├── viz.py          # Visual Lab
│   │   ├── transform.py    # Transformations
│   │   ├── distances.py    # Distances Lab
│   │   ├── benchmark.py    # Benchmarking
│   │   └── tasks/          # Task modules
│   │       ├── classification.py
│   │       ├── regression.py
│   │       ├── clustering.py
│   │       ├── forecasting.py
│   │       ├── anomaly.py
│   │       ├── segmentation.py
│   │       └── similarity.py
│   ├── components/         # Reusable UI components
│   │   ├── learning_section.py
│   │   ├── param_form.py
│   │   ├── level_gate.py
│   │   ├── data_preview.py
│   │   ├── metrics_display.py
│   │   └── code_export.py
│   ├── services/           # Business logic
│   │   ├── datasets.py     # Data loading & caching
│   │   ├── aeon_discovery.py  # Estimator discovery
│   │   ├── converters.py   # Data format conversion
│   │   ├── forms.py        # Dynamic form generation
│   │   ├── runners.py      # Task execution
│   │   └── exporters.py    # Code generation
│   └── assets/             # CSS, favicon
├── tests/                  # Pytest test suite
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Container orchestration
└── nginx.conf              # Production reverse proxy
```

## Quick Start

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/aeon-toolkit/aeon-play.git
cd aeon-play

# Install uv if you haven't already
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run the application
python -m aeon_play.app

# Open http://localhost:8050 in your browser
```

### Using Docker

```bash
# Build and run
docker-compose up --build

# Open http://localhost:8050
```

### Using pip

```bash
pip install -e .
python -m aeon_play.app
```

## Configuration

Environment variables (set in `.env` or system environment):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8050` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `AEON_PLAY_DEBUG` | `false` | Enable debug mode |
| `AEON_PLAY_CACHE_DIR` | `./.cache` | Cache directory |
| `AEON_PLAY_MAX_CASES` | `10000` | Maximum dataset cases |
| `AEON_PLAY_MAX_TIMEPOINTS` | `5000` | Maximum timepoints |

## Deployment

### Production with Docker

```bash
# Build production image
docker build -t aeon-play:latest .

# Run with docker-compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Manual Production Deployment

```bash
# Install production dependencies
pip install .

# Run with gunicorn
gunicorn --bind 0.0.0.0:8050 --workers 4 --threads 2 aeon_play.app:server
```

### DNS & SSL Setup for aeon-play.org

1. **Point DNS** to your server's IP address
2. **Install SSL** using Let's Encrypt:
   ```bash
   certbot certonly --webroot -w /var/www/certbot -d aeon-play.org -d www.aeon-play.org
   ```
3. **Configure nginx** using the provided `nginx.conf`

## Development

### Running Tests

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=aeon_play

# Run specific test file
pytest tests/test_converters.py -v
```

### Adding a New Task Page

1. Create a new file in `src/aeon_play/pages/tasks/`:

```python
# my_task.py
import dash
from dash import html, dcc, callback

dash.register_page(__name__, path="/tasks/my-task", name="My Task")

layout = html.Div([
    # Your layout here
])

# Your callbacks here
```

2. The page will be automatically registered with the multi-page app.

### Adding a New Estimator Type

1. Add the type filter mapping in `services/aeon_discovery.py`:

```python
TYPE_FILTERS = {
    # ... existing types
    "my_type": "my-type-filter",
}
```

2. Add beginner-friendly estimators to the curated lists:

```python
BEGINNER_MY_TYPE = ["EstimatorA", "EstimatorB"]
```

## Technology Stack

- **Frontend**: Plotly Dash, Dash Bootstrap Components
- **Backend**: Flask (via Dash), Gunicorn
- **ML Library**: aeon (scikit-learn compatible)
- **Caching**: diskcache, Flask-Caching
- **Containerization**: Docker, docker-compose
- **Reverse Proxy**: Nginx

## Screenshots

*Screenshots coming soon*

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [aeon](https://www.aeon-toolkit.org/) - The time series ML toolkit powering this app
- [Plotly Dash](https://dash.plotly.com/) - The web framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning foundation

## Links

- **Documentation**: https://aeon-play.org/docs
- **aeon Library**: https://www.aeon-toolkit.org/
- **Bug Reports**: https://github.com/aeon-toolkit/aeon-play/issues
