# Smart Fork - Release Checklist

## Pre-Release Verification

### Code Quality
- [ ] All unit tests passing (run `pytest tests/`)
- [ ] All integration tests passing
- [ ] Code coverage >80% on core modules
- [ ] No critical linting errors (`ruff check src/`)
- [ ] Code formatted with black (`black src/ tests/`)
- [ ] Type checking passes (`mypy src/`)

### Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md updated with release notes
- [ ] All configuration options documented
- [ ] Installation instructions tested
- [ ] Example usage scenarios verified

### Packaging
- [ ] Version number updated in:
  - [ ] `pyproject.toml`
  - [ ] `src/smart_fork/__init__.py`
- [ ] `pyproject.toml` dependencies are up to date
- [ ] Entry point script works (`smart-fork --help`)
- [ ] Package builds successfully (`python -m build`)
- [ ] Wheel installs correctly (`pip install dist/*.whl`)
- [ ] All dependencies install correctly

### Testing
- [ ] Fresh install tested in clean environment
- [ ] MCP server starts correctly
- [ ] `/fork-detect` command works
- [ ] Search functionality tested with sample data
- [ ] Background indexing verified
- [ ] Initial setup flow tested
- [ ] Performance benchmarks meet targets (<3s search)

### Security
- [ ] No hardcoded credentials or secrets
- [ ] Dependencies scanned for vulnerabilities
- [ ] Server binds only to localhost (127.0.0.1)
- [ ] File paths validated and sanitized

## Release Process

### Build and Test
1. Clean build artifacts:
   ```bash
   rm -rf build/ dist/ src/smart_fork.egg-info/
   ```

2. Build distribution packages:
   ```bash
   python -m build
   ```

3. Test installation from wheel:
   ```bash
   pip install dist/smart_fork-*.whl
   ```

4. Verify entry point:
   ```bash
   smart-fork --version
   ```

### Git and GitHub
1. Commit all changes:
   ```bash
   git add .
   git commit -m "Release v0.1.0"
   ```

2. Create git tag:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   ```

3. Push to GitHub:
   ```bash
   git push origin main
   git push origin v0.1.0
   ```

### PyPI Publishing (when ready)
1. Install twine:
   ```bash
   pip install twine
   ```

2. Upload to Test PyPI first:
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. Test installation from Test PyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ smart-fork
   ```

4. If Test PyPI works, upload to PyPI:
   ```bash
   twine upload dist/*
   ```

### Post-Release
- [ ] Verify package available on PyPI
- [ ] Test fresh installation: `pip install smart-fork`
- [ ] Update documentation with PyPI badge
- [ ] Announce release (GitHub, social media, etc.)
- [ ] Monitor for installation issues
- [ ] Respond to bug reports promptly

## Version Numbering

Follow semantic versioning (semver.org):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

## Rollback Plan

If critical issues are found after release:

1. Yank the release from PyPI (makes it unavailable for new installs):
   ```bash
   twine upload --repository pypi --skip-existing dist/*
   ```

2. Fix the issue and release a patch version

3. Update documentation with migration guide if needed

## Notes
- Keep credentials secure (use PyPI API tokens, not passwords)
- Test on multiple Python versions (3.10, 3.11, 3.12)
- Consider GitHub Actions for automated releases
- Maintain CHANGELOG.md for user-facing changes
