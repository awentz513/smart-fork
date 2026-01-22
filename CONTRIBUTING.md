# Contributing to Smart Fork Detection

Thank you for considering contributing to Smart Fork! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building a welcoming community for developers using Claude Code.

## Ways to Contribute

### 1. Report Bugs

If you find a bug, please [open an issue](https://github.com/recursive-vibe/smart-fork/issues) with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs actual behavior
- Your environment (OS, Python version, Claude Code version)
- Relevant logs or error messages

### 2. Suggest Features

Have an idea? [Open a feature request](https://github.com/recursive-vibe/smart-fork/issues) describing:
- The problem you're trying to solve
- Your proposed solution
- Alternative approaches you considered
- Whether you'd be willing to implement it

### 3. Submit Pull Requests

#### Before You Start

1. Check existing issues and PRs to avoid duplicates
2. For major changes, open an issue first to discuss
3. Fork the repository and create a feature branch

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Smart-Fork.git
cd Smart-Fork

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

#### Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Write clean, documented code following existing patterns
3. Add tests for new functionality
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Update documentation as needed (README.md, docstrings)
6. Commit with clear, descriptive messages

#### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for function signatures
- Write docstrings for public methods
- Keep functions focused and small
- Add comments for complex logic

#### Testing

- Write unit tests for new features
- Ensure existing tests still pass
- Aim for >80% code coverage
- Test edge cases and error handling

#### Submitting

1. Push to your fork: `git push origin feature/your-feature-name`
2. Open a pull request against `main`
3. Fill out the PR template completely
4. Link related issues
5. Wait for review and address feedback

### 4. Improve Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples or usage scenarios
- Improve installation instructions
- Write tutorials or guides

### 5. Help Others

- Answer questions in [Discussions](https://github.com/recursive-vibe/smart-fork/discussions)
- Help triage and reproduce issues
- Review pull requests
- Share your Smart Fork setup and tips

## Areas for Contribution

### High Priority
- Performance optimizations for large session databases
- Additional embedding model support
- UI/UX improvements for search results
- Integration with more IDEs/editors
- Better error messages and debugging tools

### Medium Priority
- Advanced search filters (date ranges, regex, boolean operators)
- Session analytics and visualization
- Migration tools from other session formats
- Internationalization (i18n)

### Nice to Have
- Cloud sync with encryption
- Team/shared session libraries
- Browser extension for web-based Claude
- Mobile app for searching sessions on the go

## Development Workflow

```bash
# Always start from latest main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-feature

# Make changes, add tests
# ...

# Run tests locally
python -m pytest tests/ -v

# Commit changes
git add .
git commit -m "Add feature: description"

# Push and create PR
git push origin feature/my-feature
```

## Questions?

- Open a [Discussion](https://github.com/recursive-vibe/smart-fork/discussions) for general questions
- Use [Issues](https://github.com/recursive-vibe/smart-fork/issues) for bugs and features
- Tag maintainers in PRs if you need review

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Smart Fork better! 🚀
