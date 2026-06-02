# Contribution Guide

This document explains how to contribute to the `dSTORMQuant` project.

## 1. Scope and Principles

- Keep changes focused and easy to review.
- Prefer small, atomic commits with clear messages.
- Preserve reproducibility: configuration changes should be explicit and documented.
- Do not commit experimental data or local outputs.

## 2. Local Setup

From the project root:

```bash
uv sync --extra dev
cd finder_cpp
pip install -e . --no-build-isolation
cd ..
```

If you do not use `uv`, install dev tooling manually:

```bash
pip install -e .
pip install ruff black pre-commit pytest
```

## 3. Git Hooks and Code Quality

Install pre-commit hooks once:

```bash
uv run pre-commit install
```

Run all hooks before opening a PR:

```bash
uv run pre-commit run --all-files
```

Run Ruff directly when iterating quickly:

```bash
uv run ruff check .
```

Apply auto-fixes when appropriate:

```bash
uv run ruff check . --fix
```

## 4. Testing

Run tests when adding or changing behavior:

```bash
uv run pytest
```

If no tests exist yet for your feature, include a short manual validation note in your PR.

## 5. Data and Outputs

- The repository ignores `data/**`; do not rely on pushing data files.
- Keep metadata schema changes synchronized with docs and config models.
- Do not commit generated outputs, temporary files, or local notebooks unless explicitly needed.

## 6. Documentation Requirements

### Python docstrings

Use **Google-style** docstrings on all public functions and methods:

- One-line summary, then optional longer description.
- ``Args:`` for each parameter (name, type context, meaning).
- ``Returns:`` for return value(s); ``Raises:`` when relevant.
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

- `README.md` for user-facing setup/run instructions.
- `docs/dSTORMQuant_User_Guide.docx` for detailed usage (when updated).
- `docs/PIPELINE_DOCUMENTATION.md` for technical details and assumptions (when present).
- `config/config.yaml` inline comments for new parameters.

## 7. Branching and Pull Requests

Recommended flow:

1. Create a branch from `main`.
2. Implement a focused change.
3. Run `pre-commit` and tests.
4. Open a PR with clear context and validation evidence.

PR checklist:

- [ ] Code compiles/runs locally.
- [ ] `uv run pre-commit run --all-files` passes.
- [ ] Tests pass, or manual validation is described.
- [ ] Docs are updated for any user-visible or config-visible change.
- [ ] No sensitive data, large binaries, or local outputs are included.

## 8. Commit Message Guidance

Use concise, descriptive commit messages. Example patterns:

- `fix: correct drift validation key usage in docs`
- `feat: add cluster-center kNN summary export`
- `docs: align README with zipped output workflow`

## 9. Reporting Issues

When opening an issue, include:

- Expected behavior
- Actual behavior
- Minimal reproducible input/config
- Error logs (stack trace)
- Environment details (OS, Python version, install method)
