# Agent Instructions

## Before Committing or Pushing

Always run lint and format checks before committing code:

```bash
# Check for lint errors
uv run ruff check src/ tests/

# Fix lint errors automatically
uv run ruff check --fix src/ tests/

# Check formatting
uv run ruff format --check src/ tests/

# Apply formatting
uv run ruff format src/ tests/
```

Ensure both commands pass with no errors before committing.

## Quick One-Liner

```bash
uv run ruff check --fix src/ tests/ && uv run ruff format src/ tests/
```
