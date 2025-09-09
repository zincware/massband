# Massband Development Guidelines

> **Audience**: LLM-driven engineering agents and human developers

Massband is a Python library for molecular dynamics (MD) post-processing tools designed for large-scale simulation data analysis.


## Required Development Workflow

**CRITICAL**: Always run these commands in sequence before committing:

```bash
uv sync                              # Install dependencies
uvx pre-commit run --all-files       # Linting and Code formatting
uv run pytest                        # Run full test suite
```

**All three must pass** - this is enforced by CI.

## Repository Structure
| Path              | Purpose                                                |
|-------------------|--------------------------------------------------------|
| `massband/`       | Source code for the massband library                   |
| `tests/`          | Unit and integration tests for the massband library    |
| `docs/`           | Documentation for the massband library                 |
| `tmp/`            | Git ignored directory for testing.                     |


## Imports
All imports are made available through the `lazy-loader` from `__init__.pyi`.

### Code Standards

- Python ≥ 3.10 with full type annotations
- Follow existing patterns and maintain consistency
- **Prioritize readable, understandable code** - clarity over cleverness
- Avoid obfuscated or confusing patterns even if they're shorter
- Each feature needs corresponding tests


### Docstrings
- Docstrings are written in the numpy style
- Each Node must have a docstring describing its functionality.
- Docstrings must be concise and clear.
- Docstrings should not include unnecessary information or verbosity. E.g. A Node that runs a molecular dynamics simulation should not explain what molecular dynamics is. Expect the user to have expert knowledge in the field.


### Key Design Patterns

- **Batch Processing**: All trajectory processing uses batching to handle large datasets efficiently
- **JAX Integration**: Uses JAX for JIT compilation and GPU acceleration of numerical computations
