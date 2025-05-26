# AGENTS.md

These instructions apply to the entire repository.

## 1. Commit Discipline
- **Conventional Commits** are mandatory.
- Keep commits small and focused.
- Run the complete test suite before committing.

## 2. Environment Setup

### Initial Setup
```bash
# Create and set up the development environment
./setup_env.sh --dev

# Install pre-commit hooks
conda run --prefix dev_env pre-commit install
```

### Running Commands

#### Recommended: Using `conda run` (for scripts and one-off commands)
```bash
# Run a single command in the environment
conda run --prefix dev_env python -m askdata.ingest.door

# Run tests
conda run --prefix dev_env pytest tests/
```

#### Using `conda activate` (for interactive development)
```bash
# Activate the environment
conda activate ./dev_env

# Now you can run commands directly
python -m askdata.ingest.door
pytest tests/

# When done, deactivate
conda deactivate
```

### Best Practices
- Use `conda run --prefix dev_env` in:
  - CI/CD pipelines
  - Scripts and one-off commands
  - When you need a clean environment for each command
- Use `conda activate ./dev_env` for:
  - Interactive development sessions
  - When running multiple commands in sequence
  - When modifying the environment (e.g., `pip install`)

### Dependencies
- The environment is defined in `environment-dev.yml`
- Exact versions are pinned in `conda-lock.yml`
- Always update both files when adding/updating dependencies

## 3. Test‑Driven Development (TDD)
- Add failing tests first, then implement features until tests pass.
- Errors alone do not count as true failures; confirm with failing tests.
- Commit only after tests pass.

## 4. Pre‑commit Hooks
- Install hooks with `pre-commit install`.
- Hooks must include Ruff, Black, Isort, MyPy (`--strict`), Interrogate, and pytest.
- CI runs `pre-commit run --all-files`.

## 5. FAIR & Metadata
- Provide `CITATION.cff` and `codemeta.json` files.
- Maintain metadata CSVs in `metadata/`.
- Document data sources and usage in `README.md`.

## 6. Documentation & Releases
- Use MkDocs or Sphinx for docs.
- Version with Semantic Versioning and an auto-generated changelog.

Follow these rules to keep the project reproducible, FAIR-aligned, and easy to maintain.
