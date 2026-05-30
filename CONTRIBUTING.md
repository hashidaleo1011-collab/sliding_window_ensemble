# Contributing to Sliding Window Ensemble

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sliding_window_ensemble.git
   cd sliding_window_ensemble
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
pytest
pytest --cov=sliding_window_ensemble  # With coverage
```

### Code Style

We use Black for code formatting:

```bash
black sliding_window_ensemble tests
```

Lint with flake8:

```bash
flake8 sliding_window_ensemble tests
```

### Type Checking

We use mypy for type checking:

```bash
mypy sliding_window_ensemble
```

## Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for your changes
4. Run tests and linters
5. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request

## Reporting Issues

Please include:
- Python version
- PyTorch version
- Transformers version
- Error message and traceback
- Reproducible code example

## Code of Conduct

Be respectful and constructive in all interactions.
