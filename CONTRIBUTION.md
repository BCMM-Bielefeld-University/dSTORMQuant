# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at
[https://github.com/BCMM-Bielefeld-University/dSTORMQuant/issues](https://github.com/BCMM-Bielefeld-University/dSTORMQuant/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Python version and how you installed the package (`pip`, `uv`, editable
install, etc.).
- Any details about your local setup that might be helpful in troubleshooting
(C++ compiler, whether FINDER built successfully, GUI/headless environment).
- The relevant parts of `config/config.yaml` (redact sensitive paths if needed).
- Expected behavior, actual behavior, and detailed steps to reproduce the bug.
- Error logs / stack traces when available.
- A minimal reproducible input when possible (tiny synthetic CSV / metadata is
preferred over large experimental files).

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and
`help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
`enhancement` and `help wanted` is open to whoever wants to implement it.

### Write Tests

Improving test coverage is especially valuable. Prefer fast unit tests with
tiny synthetic data under `tests/`. Full end-to-end runs on large demo datasets
remain mostly manual for now (see [Testing status](#testing-status)).

### Write Documentation

dSTORMQuant could always use more documentation, whether as part of the
official docs, in Google-style docstrings, README updates, config comments,
or even on the web in blog posts, articles, and such.

When behavior or parameters change, please update:

- `README.md` for user-facing setup, demo, and run instructions
- `docs/` user guide materials when relevant
- `config/config.yaml` inline comments for new or changed parameters
- `src/dSTORMQuant/core/config/models.py` when adding or renaming config fields

### Submit Feedback

The best way to send feedback is to file an issue at
[https://github.com/BCMM-Bielefeld-University/dSTORMQuant/issues](https://github.com/BCMM-Bielefeld-University/dSTORMQuant/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a research / volunteer-driven project, and that
contributions are welcome :)

## Scope and Principles

- Keep changes focused and easy to review.
- Prefer small, atomic commits with clear messages.
- Preserve reproducibility: configuration changes should be explicit and
documented.
- Do not commit experimental data, demo archives, or local pipeline outputs.
- Keep `config/config.yaml`, Pydantic config models
(`src/dSTORMQuant/core/config/models.py`), and user-facing docs in sync when
behavior or parameters change.

## Get Started!

Ready to contribute? Here's how to set up `dSTORMQuant` for local development.

**Requirements:** Python 3.10.12+, a C++ compiler (for the FINDER extension),
and `pip`.

1. Fork the `dSTORMQuant` repo on GitHub.
2. Clone your fork locally:
  ```sh
   git clone git@github.com:your_name_here/dSTORMQuant.git
   cd dSTORMQuant
  ```
3. Add the upstream remote (so you can keep your fork up to date):
  ```sh
   git remote add upstream https://github.com/BCMM-Bielefeld-University/dSTORMQuant.git
   git fetch upstream
   git checkout main
   git merge upstream/main
  ```
4. Create and activate a virtual environment, then install the package and
  developer tools:
   **Windows (PowerShell):**
   **Linux / macOS:**
   For tests only (lighter than full `dev`):
5. Build and install the FINDER C++ extension:
  ```sh
   cd finder_cpp
   pip install -r requirements-build.txt
   pip install pybind11
   pip install -e . --no-build-isolation
   cd ..
  ```
6. Install pre-commit hooks (once):
  ```sh
   pre-commit install
  ```
7. Create a branch for local development:
  ```sh
   git checkout -b name-of-your-bugfix-or-feature
  ```
   Now you can make your changes locally.
8. When you're done making changes, check that your changes pass linting and
  tests:
   Tip: while iterating quickly you can run:
   Note: the Ruff pre-commit hook is configured with `uv run ruff check .`.
   If you do not use `uv`, run `ruff check .` directly (as above).
9. Commit your changes and push your branch to GitHub:
  ```sh
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
  ```
   Prefer concise, descriptive commit messages, for example:
  - `fix: correct drift validation key usage in docs`
  - `feat: add cluster-center kNN summary export`
  - `docs: align README with zipped output workflow`
  - `test: cover temporal grouping edge cases`
10. Submit a pull request through the GitHub website against
  `BCMM-Bielefeld-University/dSTORMQuant` (`main`).

### Alternative: uv

If you use [uv](https://docs.astral.sh/uv/), the repo includes `uv.lock`:

```sh
uv sync --extra dev
cd finder_cpp
pip install -e . --no-build-isolation
cd ..
uv run ruff check .
uv run pytest
```

### Demo data for manual testing

Do not commit input files or pipeline outputs. Download
[examples.zip](https://github.com/BCMM-Bielefeld-University/dSTORMQuant/releases/download/v0.0.1/examples.zip)
and place files in `data/input/` and `data/metadata/` as described in
[README §3 Demo](README.md#3-demo).

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. Keep the change focused and easy to review. Prefer small, atomic commits.
2. The pull request should include tests when practical (especially for new or
  changed behavior under `tests/`). If automated tests are not yet feasible
   for your change, describe manual validation in the PR.
3. If the pull request adds functionality, the docs should be updated. Put new
  functionality into functions/methods with Google-style docstrings, and
   update `README.md` / config comments when user-visible behavior changes.
4. Keep `config/config.yaml` and Pydantic models in
  `src/dSTORMQuant/core/config/models.py` synchronized.
5. Do not include experimental data, demo archives, large binaries, or local
  outputs (`data/output/`, `data/temp/`, logs).
6. If you changed `finder_cpp/`, confirm the FINDER extension still builds.
7. The pull request should pass CI on GitHub Actions (lint, smoke check, unit
  tests, and package builds). Make sure locally that at least the following
   pass before opening the PR:

In the PR description, please include:

- What changed and why
- How you validated it (tests run, demo command, screenshots if relevant)
- Any related issue link (e.g. `Fixes #123`)
- Breaking changes or config migrations, if any

PR checklist:

- [ ] Code runs locally
- [ ] `ruff check .` passes (or `pre-commit run --all-files` passes)
- [ ] `pytest` passes, or manual validation is described
- [ ] FINDER extension builds if you changed `finder_cpp/`
- [ ] Docs / config models updated for user-visible or config-visible changes
- [ ] No sensitive data, large binaries, demo archives, or local outputs included

## Tips

To run a subset of tests:

```sh
pytest tests/test_filtering.py
pytest tests/test_config.py -q
```

### Testing status


| Layer                                                                | Status                                                     |
| -------------------------------------------------------------------- | ---------------------------------------------------------- |
| Unit tests (config, filtering, temporal grouping, NN helpers, utils) | Present under `tests/`; run in CI                          |
| Lint / smoke / package build                                         | Present in CI                                              |
| Full end-to-end pipeline on demo CSVs                                | Manual (see README demo)                                   |
| Napari / GUI visualization                                           | Not in default CI                                          |
| FINDER numerical regression                                          | Package wheel build in CI; dedicated numeric suite planned |


### Data and outputs

The repository tracks the `data/` folder layout, not its contents.
`.gitignore` excludes `examples.zip`, `examples/`, `data/input/*`,
`data/metadata/*` (`.gitkeep` files remain), `data/output/`, and
`data/temp/`. See [data/README.md](data/README.md).

### Documentation style (docstrings)

Use **Google-style** docstrings on public functions and methods:

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

## Continuous Integration

Pull requests and pushes to `main` / `master` run
[.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml):

- `ruff check .`
- Editable install smoke check
- `pytest` unit tests under `tests/`
- Build artifacts for the main package and `finder_cpp`
- On version tags (`v*`), create a GitHub Release with built artifacts

CI currently targets Python 3.12. Local development should work with Python
3.10.12+.

## Maintenance and Roadmap

dSTORMQuant is maintained by the Biochemistry and Molecular Medicine group
(Medical School OWL, Bielefeld University) and developed as an open-source
project on GitHub.

Future goals:

- Expand automated unit and integration tests
- Extending input parsing to additional SMLM software export formats (e.g. ThunderSTORM or NimOS column layouts) based on user needs
- Adding further analysis methods for individual pipeline stages (e.g., clustering and drift correction)

Bug reports, feature requests, and pull requests via GitHub are the preferred
channels for community involvement.

## Deploying

A reminder for maintainers on how to cut a release:

1. Make sure all changes are committed and CI is green on `main`.
2. Bump the package version in `pyproject.toml` if needed.
3. Create and push an annotated version tag, for example:
  ```sh
   git tag -a v0.0.2 -m "Release v0.0.2"
   git push origin v0.0.2
  ```
4. The GitHub Actions release job attaches built package artifacts to the
  GitHub Release for that tag.

Publishing to PyPI is optional and can be added later with a dedicated
workflow if needed.

## License

This project is released under the [MIT License](LICENSE). By contributing,
you agree that your contributions will be licensed under the same terms.

Parts of the FINDER C++ components may carry additional notices; see
`finder_cpp/` and [THRID-PARTY LICENSE.txt](THRID-PARTY%20LICENSE.txt) when
modifying third-party code.

## Questions

If you are unsure whether an idea fits, open a GitHub issue first and describe
your proposal briefly. For Code of Conduct concerns, contact the project team
as described in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Code of Conduct

Please note that this project is released with a
[Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this
project you agree to abide by its terms.