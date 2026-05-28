# SAPAT Development Guide

## Build Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Build package
python -m build

# Install locally
pip install dist/sapat-0.2.4-py3-none-any.whl  # Replace with latest version

# Run tests (when added)
# pytest

# Lint and format code
black sapat/
isort sapat/
```

## Code Style Guidelines
- **Imports**: Use isort for organizing imports
- **Formatting**: Follow PEP 8, use black formatter (line length 88)
- **Types**: Consider adding type hints to function signatures
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error Handling**: Use specific exceptions with informative messages
- **Documentation**: Docstrings for modules, classes, functions
- **Dependencies**: Minimize dependencies, document in requirements.txt

## Version Control
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `sapat/__version__.py`
- Add detailed notes to CHANGELOG.md for each release