#!/usr/bin/env python3
"""
Test script for Task 23: Package for distribution

This script tests all packaging requirements:
1. Configure pyproject.toml for pip installation
2. Add entry point script for MCP server
3. Test pip install from local source
4. Test pip install from built wheel
5. Verify all dependencies bundled correctly
6. Create release checklist
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import tempfile
import venv


def run_command(cmd: list[str], cwd: str = None, check: bool = True) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result.returncode, result.stdout, result.stderr


def test_pyproject_toml():
    """Test 1: Configure pyproject.toml for pip installation"""
    print("\n" + "="*80)
    print("TEST 1: Verify pyproject.toml configuration")
    print("="*80)

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ FAIL: pyproject.toml not found")
        return False

    content = pyproject_path.read_text()

    checks = [
        ('[build-system]', 'build-system section'),
        ('setuptools', 'setuptools backend'),
        ('[project]', 'project section'),
        ('name = "smart-fork"', 'project name'),
        ('version =', 'version'),
        ('dependencies =', 'dependencies list'),
        ('[project.scripts]', 'scripts section'),
        ('smart-fork =', 'entry point script'),
    ]

    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ {description} present")
        else:
            print(f"❌ {description} missing")
            all_passed = False

    if all_passed:
        print("\n✅ PASS: pyproject.toml properly configured")
    else:
        print("\n❌ FAIL: pyproject.toml configuration incomplete")

    return all_passed


def test_entry_point():
    """Test 2: Add entry point script for MCP server"""
    print("\n" + "="*80)
    print("TEST 2: Verify entry point script exists")
    print("="*80)

    server_path = Path("src/smart_fork/server.py")
    if not server_path.exists():
        print("❌ FAIL: server.py not found")
        return False

    content = server_path.read_text()

    if 'def main()' not in content:
        print("❌ FAIL: main() function not found in server.py")
        return False

    print("✓ main() function exists in server.py")

    # Check pyproject.toml has correct entry point
    pyproject_path = Path("pyproject.toml")
    pyproject_content = pyproject_path.read_text()

    if 'smart-fork = "smart_fork.server:main"' in pyproject_content:
        print("✓ Entry point 'smart-fork' points to smart_fork.server:main")
        print("\n✅ PASS: Entry point script properly configured")
        return True
    else:
        print("❌ FAIL: Entry point not properly configured in pyproject.toml")
        return False


def test_pip_install_local():
    """Test 3: Test pip install from local source"""
    print("\n" + "="*80)
    print("TEST 3: Test pip install from local source")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nCreating test virtual environment in {tmpdir}...")

        # Create virtual environment
        venv_dir = Path(tmpdir) / "test_venv"
        venv.create(venv_dir, with_pip=True)

        # Determine paths
        if sys.platform == "win32":
            pip_path = venv_dir / "Scripts" / "pip.exe"
            python_path = venv_dir / "Scripts" / "python.exe"
        else:
            pip_path = venv_dir / "bin" / "pip"
            python_path = venv_dir / "bin" / "python"

        # Upgrade pip
        print("\nUpgrading pip...")
        returncode, stdout, stderr = run_command(
            [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
            check=False
        )

        # Install from local source (editable mode)
        print("\nInstalling package from local source in editable mode...")
        returncode, stdout, stderr = run_command(
            [str(pip_path), "install", "-e", "."],
            check=False
        )

        if returncode != 0:
            print(f"❌ FAIL: pip install -e . failed")
            print(f"STDERR: {stderr}")
            return False

        print("✓ Package installed successfully")

        # Verify smart-fork command exists
        print("\nVerifying smart-fork command...")
        if sys.platform == "win32":
            smart_fork_path = venv_dir / "Scripts" / "smart-fork.exe"
        else:
            smart_fork_path = venv_dir / "bin" / "smart-fork"

        if smart_fork_path.exists():
            print(f"✓ smart-fork command exists at {smart_fork_path}")
        else:
            print(f"❌ FAIL: smart-fork command not found at {smart_fork_path}")
            return False

        # Verify package can be imported
        print("\nVerifying package can be imported...")
        returncode, stdout, stderr = run_command(
            [str(python_path), "-c", "import smart_fork; print(smart_fork.__version__)"],
            check=False
        )

        if returncode == 0:
            print(f"✓ Package imported successfully, version: {stdout.strip()}")
            print("\n✅ PASS: pip install from local source works")
            return True
        else:
            print(f"❌ FAIL: Package import failed")
            print(f"STDERR: {stderr}")
            return False


def test_build_wheel():
    """Test 4: Test pip install from built wheel"""
    print("\n" + "="*80)
    print("TEST 4: Build wheel and test installation")
    print("="*80)

    # Clean up any existing build artifacts
    print("\nCleaning up existing build artifacts...")
    for path in ["build", "dist", "src/smart_fork.egg-info"]:
        if Path(path).exists():
            shutil.rmtree(path)
            print(f"✓ Removed {path}")

    # Build the wheel
    print("\nBuilding wheel...")
    returncode, stdout, stderr = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "build"],
        check=False
    )

    returncode, stdout, stderr = run_command(
        [sys.executable, "-m", "build"],
        check=False
    )

    if returncode != 0:
        print(f"❌ FAIL: Wheel build failed")
        print(f"STDERR: {stderr}")
        return False

    print("✓ Wheel built successfully")

    # Find the wheel file
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ FAIL: dist directory not created")
        return False

    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ FAIL: No wheel file found in dist/")
        return False

    wheel_path = wheel_files[0]
    print(f"✓ Wheel file: {wheel_path.name}")

    # Test installation from wheel
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nCreating test virtual environment in {tmpdir}...")

        venv_dir = Path(tmpdir) / "test_venv_wheel"
        venv.create(venv_dir, with_pip=True)

        if sys.platform == "win32":
            pip_path = venv_dir / "Scripts" / "pip.exe"
            python_path = venv_dir / "Scripts" / "python.exe"
        else:
            pip_path = venv_dir / "bin" / "pip"
            python_path = venv_dir / "bin" / "python"

        # Install from wheel
        print(f"\nInstalling from wheel...")
        returncode, stdout, stderr = run_command(
            [str(pip_path), "install", str(wheel_path.absolute())],
            check=False
        )

        if returncode != 0:
            print(f"❌ FAIL: Installation from wheel failed")
            print(f"STDERR: {stderr}")
            return False

        print("✓ Package installed from wheel successfully")

        # Verify installation
        print("\nVerifying installation...")
        returncode, stdout, stderr = run_command(
            [str(python_path), "-c", "import smart_fork; print(smart_fork.__version__)"],
            check=False
        )

        if returncode == 0:
            print(f"✓ Package imported successfully, version: {stdout.strip()}")
            print("\n✅ PASS: pip install from wheel works")
            return True
        else:
            print(f"❌ FAIL: Package import failed")
            print(f"STDERR: {stderr}")
            return False


def test_dependencies():
    """Test 5: Verify all dependencies bundled correctly"""
    print("\n" + "="*80)
    print("TEST 5: Verify dependencies")
    print("="*80)

    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    required_deps = [
        "fastmcp",
        "chromadb",
        "sentence-transformers",
        "fastapi",
        "uvicorn",
        "watchdog",
        "psutil",
        "pydantic",
        "httpx",
    ]

    all_present = True
    for dep in required_deps:
        if dep in content:
            print(f"✓ {dep} listed in dependencies")
        else:
            print(f"❌ {dep} missing from dependencies")
            all_present = False

    if all_present:
        print("\n✅ PASS: All required dependencies listed")
        return True
    else:
        print("\n❌ FAIL: Some dependencies missing")
        return False


def create_release_checklist():
    """Test 6: Create release checklist"""
    print("\n" + "="*80)
    print("TEST 6: Create release checklist")
    print("="*80)

    checklist_content = """# Smart Fork - Release Checklist

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
"""

    checklist_path = Path("RELEASE_CHECKLIST.md")
    checklist_path.write_text(checklist_content)
    print(f"✓ Release checklist created at {checklist_path}")
    print(f"✓ Checklist contains {len(checklist_content.splitlines())} lines")
    print("\n✅ PASS: Release checklist created")

    return True


def main():
    """Run all packaging tests."""
    print("\n" + "="*80)
    print("SMART FORK - PACKAGING TESTS (Task 23)")
    print("="*80)
    print("\nThis script will test all packaging requirements:")
    print("1. Configure pyproject.toml for pip installation")
    print("2. Add entry point script for MCP server")
    print("3. Test pip install from local source")
    print("4. Test pip install from built wheel")
    print("5. Verify all dependencies bundled correctly")
    print("6. Create release checklist")

    results = {
        "pyproject.toml configuration": test_pyproject_toml(),
        "Entry point script": test_entry_point(),
        "pip install from local source": test_pip_install_local(),
        "Build and install from wheel": test_build_wheel(),
        "Dependencies verification": test_dependencies(),
        "Release checklist": create_release_checklist(),
    }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Package is ready for distribution!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Please review and fix issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
