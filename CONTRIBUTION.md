# Contribution Guide

This document explains how to contribute to the `dSTORMQuant` project.

## 1. Scope and Principles

- Keep changes focused and easy to review.
- Prefer small, atomic commits with clear messages.
- Preserve reproducibility: configuration changes should be explicit and documented.
- Do not commit experimental data, demo archives, or local pipeline outputs.
- Keep `config/config.yaml`, Pydantic config models (`src/dSTORMQuant/core/config/models.py`), and user-facing docs in sync when behavior or parameters change.

## 2. Local Setup

**Requirements:** Python 3.10.12+, a C++ compiler (for the FINDER extension), and `pip`.

### Recommended: virtual environment + pip

This matches [README.md](README.md) and the GitHub Actions workflow.

From the project root:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Install the package, dev tools, and the FINDER C++ extension:

```bash
pip install -e .
pip install -e ".[dev]"

cd finder_cpp
pip install -r requirements-build.txt
pip install pybind11
pip install -e . --no-build-isolation
cd ..
```



### Alternative: uv

If you use [uv](https://docs.astral.sh/uv/), the repo includes `uv.lock`:

```bash
uv sync --extra dev

cd finder_cpp
pip install -e . --no-build-isolation
cd ..
```



### Demo data for manual testing

Do not commit input files. Download [examples.zip](https://github.com/BCMM-Bielefeld-University/dSTORMQuant/releases/download/v0.0.1/examples.zip) and place files in `data/input/` and `data/metadata/` as described in [README §3 Demo](README.md#3-demo).

## 3. Git Hooks and Code Quality

Ruff is configured in `ruff.toml`. Pre-commit hooks are defined in `.pre-commit-config.yaml`.

Install hooks once (after dev dependencies are installed):

```bash
pre-commit install
```

Run all hooks before opening a PR:

```bash
pre-commit run --all-files
```

The Ruff hook is configured to run via `uv run ruff check .`. Either install `uv`, or run Ruff directly while iterating:

```bash
ruff check .
ruff check . --fix
```

With uv:

```bash
uv run ruff check .
uv run ruff check . --fix
```



## 4. Testing

Pytest is configured in `pyproject.toml` (`testpaths = ["tests"]`), but the repository does not yet include a `tests/` suite.

When you add or change behavior:

```bash
pytest
```

Or with uv:

```bash
uv run pytest
```

Until automated tests cover your change, include a short manual validation note in your PR (for example: demo data run, config change exercised, or script output checked).

## 5. Data and Outputs

The repository tracks the `data/` folder layout, not its contents. `.gitignore` excludes:

- `examples.zip` and `examples/`
- `data/input/*` and `data/metadata/*` (`.gitkeep` files remain tracked)
- `data/output/` and `data/temp/`
- `*.log` files under `logs/`

Do not rely on pushing localization CSVs, metadata workbooks, ZIP archives, or scratch files. See [data/README.md](data/README.md) for the expected layout.

## 6. Documentation Requirements



### Python docstrings

Use **Google-style** docstrings on all public functions and methods:

- One-line summary, then optional longer description.
- `Args:` for each parameter (name, type context, meaning).
- `Returns:` for return value(s); `Raises:` when relevant.
- Do not repeat obvious inline comments for inputs/outputs already documented in the docstring.

Example:

```python
def spatiotemporal_grouping(df: pd.DataFrame, max_frame_gap: int = 2) -> pd.DataFrame:
    """Group localizations into spatiotemporal tracks.

    Args:
        df: Localization table with columns ``x``, ``y``, and ``frame``.
        max_frame_gap: Maximum frame gap to link neighbors.

    Returns:
        One row per track with aggregated coordinates and frame span.
    """
```

Update documentation whenever behavior changes:

- `README.md` for user-facing setup, demo, and run instructions.
- `docs/Supplementary matrial_UserGuide.pdf` for detailed usage.
- `config/config.yaml` inline comments for new or changed parameters.
- `src/dSTORMQuant/core/config/models.py` when adding or renaming config fields.



## 7. Continuous Integration

Pull requests and pushes to `main` / `master` run [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml):

- `ruff check .`
- Editable install smoke check (`import dSTORMQuant`, config loader import)
- Build artifacts for the main package and `finder_cpp`

Your PR should pass Ruff locally before review. CI does not currently run pytest.

## 8. Branching and Pull Requests

Recommended flow:

1. Create a branch from `main`.
2. Implement a focused change.
3. Run `pre-commit run --all-files` (or at least `ruff check .`).
4. Run `pytest` when tests apply to your change.
5. Open a PR with clear context and validation evidence.

PR checklist:

- [ ] Code compiles/runs locally.
- [ ] `ruff check .` passes (or `pre-commit run --all-files` passes).
- [ ] FINDER extension builds if you changed `finder_cpp/`.
- [ ] Tests pass, or manual validation is described.
- [ ] Docs and config models are updated for any user-visible or config-visible change.
- [ ] No sensitive data, large binaries, demo archives, or local outputs are included.



## 9. Commit Message Guidance

Use concise, descriptive commit messages. Example patterns:

- `fix: correct drift validation key usage in docs`
- `feat: add cluster-center kNN summary export`
- `docs: align README with zipped output workflow`



## 10. Reporting Issues

When opening an issue, include:

- Expected behavior
- Actual behavior
- Minimal reproducible input/config
- Error logs (stack trace)
- Environment details (OS, Python version, install method)

