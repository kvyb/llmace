# Publishing LLMACE to PyPI

This guide covers building and publishing LLMACE to PyPI.

## Prerequisites

1. **PyPI Account**: Register at https://pypi.org
2. **TestPyPI Account** (optional): Register at https://test.pypi.org
3. **API Token**: Generate from account settings
4. **Build Tools**:
```bash
pip install build twine
```

## Pre-Publishing Checklist

- [ ] All tests pass (`./run_tests.sh`)
- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] README.md accurate and complete
- [ ] LICENSE file present
- [ ] `.gitignore` configured
- [ ] No sensitive data in repository

## Building the Package

### 1. Clean Previous Builds
```bash
rm -rf dist/ build/ *.egg-info
```

### 2. Build Distribution
```bash
python -m build
```

This creates:
- `dist/llmace-X.Y.Z-py3-none-any.whl` (wheel)
- `dist/llmace-X.Y.Z.tar.gz` (source distribution)

### 3. Check Package
```bash
twine check dist/*
```

Should output: `Checking dist/...: PASSED`

## Testing on TestPyPI (Recommended)

### 1. Upload to TestPyPI
```bash
twine upload --repository testpypi dist/*
```

Enter your TestPyPI credentials when prompted.

### 2. Test Installation
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llmace
```

Note: `--extra-index-url` needed for dependencies not on TestPyPI.

### 3. Verify
```python
from llmace import LLMACE
print(LLMACE.__version__)
```

## Publishing to PyPI

### 1. Upload to PyPI
```bash
twine upload dist/*
```

Enter your PyPI credentials when prompted.

### 2. Verify Installation
```bash
pip install llmace
python -c "from llmace import LLMACE; print('Success!')"
```

### 3. Check PyPI Page
Visit https://pypi.org/project/llmace/

## Using API Tokens (Recommended)

### 1. Create `.pypirc`
```bash
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZw...
EOF

chmod 600 ~/.pypirc
```

### 2. Upload with Token
```bash
twine upload dist/*  # Uses token from .pypirc automatically
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

**Setup:**
1. Generate PyPI API token
2. Add as GitHub secret: `PYPI_API_TOKEN`
3. Create GitHub release to trigger workflow

## Version Bumping

### Semantic Versioning
- **Patch** (0.1.0 → 0.1.1): Bug fixes
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Major** (0.1.0 → 1.0.0): Breaking changes

### Update Version
Edit `pyproject.toml`:
```toml
[project]
version = "0.2.0"  # Update this
```

Tag in git:
```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

## Post-Publishing

1. **Verify Installation**:
```bash
pip install --upgrade llmace
```

2. **Update Documentation**:
- GitHub release notes
- CHANGELOG.md
- Announcement (Twitter, blog, etc.)

3. **Monitor**:
- PyPI download stats: https://pypistats.org/packages/llmace
- GitHub issues
- User feedback

## Troubleshooting

**Build Fails:**
```bash
# Check pyproject.toml syntax
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

**Upload Fails (Version Exists):**
- Bump version in `pyproject.toml`
- Rebuild: `python -m build`
- Upload again

**Missing Dependencies:**
```bash
# Ensure all imports work
python -c "import llmace; from llmace import *"
```

**Large Package Size:**
```bash
# Check what's included
tar -tzf dist/llmace-*.tar.gz | less
```

## Best Practices

1. **Never** commit API tokens
2. **Always** test on TestPyPI first
3. **Tag** releases in git
4. **Document** changes in CHANGELOG.md
5. **Verify** package locally before upload
6. **Announce** releases to users

## Resources

- PyPI: https://pypi.org
- TestPyPI: https://test.pypi.org  
- Packaging Guide: https://packaging.python.org
- Twine Docs: https://twine.readthedocs.io
- Build Docs: https://build.pypa.io

---

For issues, see [GitHub Issues](https://github.com/llmace/llmace/issues).

