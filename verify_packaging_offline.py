#!/usr/bin/env python3
"""
Offline verification for Task 23: Package for distribution

This script verifies packaging configuration without requiring network access.
"""

import sys
from pathlib import Path
import ast
import re


def verify_pyproject_toml():
    """Verify pyproject.toml is properly configured."""
    print("\n" + "="*80)
    print("TEST 1: Verify pyproject.toml configuration")
    print("="*80)

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ FAIL: pyproject.toml not found")
        return False

    content = pyproject_path.read_text()

    # Required sections and fields
    checks = {
        "Build system": [
            ('[build-system]', 'build-system section'),
            ('requires = ["setuptools>=61.0", "wheel"]', 'setuptools and wheel requirements'),
            ('build-backend = "setuptools.build_meta"', 'setuptools backend'),
        ],
        "Project metadata": [
            ('[project]', 'project section'),
            ('name = "smart-fork"', 'project name'),
            ('version = "0.1.0"', 'version'),
            ('description =', 'description'),
            ('readme = "README.md"', 'README reference'),
            ('requires-python = ">=3.10"', 'Python version requirement'),
            ('license =', 'license'),
        ],
        "Dependencies": [
            ('dependencies = [', 'dependencies list'),
            ('"fastmcp', 'fastmcp'),
            ('"chromadb', 'chromadb'),
            ('"sentence-transformers', 'sentence-transformers'),
            ('"fastapi', 'fastapi'),
            ('"uvicorn', 'uvicorn'),
            ('"watchdog', 'watchdog'),
            ('"psutil', 'psutil'),
            ('"pydantic', 'pydantic'),
            ('"httpx', 'httpx'),
        ],
        "Entry point": [
            ('[project.scripts]', 'scripts section'),
            ('smart-fork = "smart_fork.server:main"', 'entry point script'),
        ],
        "Dev dependencies": [
            ('[project.optional-dependencies]', 'optional dependencies'),
            ('dev = [', 'dev dependencies section'),
            ('"pytest', 'pytest'),
        ],
        "Package discovery": [
            ('[tool.setuptools.packages.find]', 'setuptools packages.find'),
            ('where = ["src"]', 'src directory specified'),
        ],
    }

    all_passed = True
    for category, items in checks.items():
        print(f"\n{category}:")
        for check_str, description in items:
            if check_str in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ❌ {description} missing")
                all_passed = False

    if all_passed:
        print("\n✅ PASS: pyproject.toml properly configured for pip installation")
    else:
        print("\n❌ FAIL: pyproject.toml configuration incomplete")

    return all_passed


def verify_entry_point():
    """Verify entry point script exists and is properly configured."""
    print("\n" + "="*80)
    print("TEST 2: Verify entry point script")
    print("="*80)

    # Check server.py exists
    server_path = Path("src/smart_fork/server.py")
    if not server_path.exists():
        print("❌ FAIL: src/smart_fork/server.py not found")
        return False
    print("✓ server.py exists")

    # Check for main() function
    content = server_path.read_text()
    if 'def main()' not in content:
        print("❌ FAIL: main() function not found in server.py")
        return False
    print("✓ main() function exists")

    # Verify main() has proper docstring or is callable
    if 'def main() -> None:' in content or 'def main():' in content:
        print("✓ main() function is properly defined")
    else:
        print("⚠ WARNING: main() function may not be properly typed")

    # Check pyproject.toml entry point
    pyproject_path = Path("pyproject.toml")
    pyproject_content = pyproject_path.read_text()

    if 'smart-fork = "smart_fork.server:main"' in pyproject_content:
        print("✓ Entry point 'smart-fork' correctly configured")
    else:
        print("❌ FAIL: Entry point not properly configured")
        return False

    print("\n✅ PASS: Entry point script properly configured")
    return True


def verify_package_structure():
    """Verify package structure is correct."""
    print("\n" + "="*80)
    print("TEST 3: Verify package structure")
    print("="*80)

    required_files = [
        "src/smart_fork/__init__.py",
        "src/smart_fork/server.py",
        "pyproject.toml",
        "README.md",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False

    # Check __init__.py has version
    init_path = Path("src/smart_fork/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        if '__version__' in content:
            # Extract version
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                version = match.group(1)
                print(f"✓ __version__ defined: {version}")
            else:
                print("⚠ WARNING: __version__ found but could not parse value")
        else:
            print("❌ __version__ not defined in __init__.py")
            all_exist = False

    # Check for all core modules
    core_modules = [
        "session_parser.py",
        "chunking_service.py",
        "embedding_service.py",
        "vector_db_service.py",
        "session_registry.py",
        "scoring_service.py",
        "search_service.py",
        "background_indexer.py",
        "api_server.py",
        "selection_ui.py",
        "fork_generator.py",
        "initial_setup.py",
        "memory_extractor.py",
        "config_manager.py",
    ]

    print("\nCore modules:")
    for module in core_modules:
        path = Path(f"src/smart_fork/{module}")
        if path.exists():
            print(f"  ✓ {module}")
        else:
            print(f"  ❌ {module} missing")
            all_exist = False

    if all_exist:
        print("\n✅ PASS: Package structure is correct")
    else:
        print("\n❌ FAIL: Package structure incomplete")

    return all_exist


def verify_dependencies():
    """Verify all required dependencies are listed."""
    print("\n" + "="*80)
    print("TEST 4: Verify dependencies")
    print("="*80)

    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Required runtime dependencies
    runtime_deps = [
        ("fastmcp", "MCP protocol support"),
        ("chromadb", "Vector database"),
        ("sentence-transformers", "Embedding model"),
        ("fastapi", "REST API framework"),
        ("uvicorn", "ASGI server"),
        ("watchdog", "File system monitoring"),
        ("psutil", "Memory monitoring"),
        ("pydantic", "Data validation"),
        ("httpx", "HTTP client"),
    ]

    # Required dev dependencies
    dev_deps = [
        ("pytest", "Testing framework"),
        ("pytest-asyncio", "Async test support"),
        ("pytest-cov", "Coverage reporting"),
    ]

    print("\nRuntime dependencies:")
    all_runtime = True
    for dep, description in runtime_deps:
        if f'"{dep}' in content:
            print(f"  ✓ {dep} - {description}")
        else:
            print(f"  ❌ {dep} - {description} MISSING")
            all_runtime = False

    print("\nDevelopment dependencies:")
    all_dev = True
    for dep, description in dev_deps:
        if f'"{dep}' in content:
            print(f"  ✓ {dep} - {description}")
        else:
            print(f"  ❌ {dep} - {description} MISSING")
            all_dev = False

    if all_runtime and all_dev:
        print("\n✅ PASS: All dependencies properly listed")
        return True
    elif all_runtime:
        print("\n⚠ WARNING: Runtime dependencies complete, but some dev dependencies missing")
        return True
    else:
        print("\n❌ FAIL: Some runtime dependencies missing")
        return False


def verify_readme():
    """Verify README.md exists and has required sections."""
    print("\n" + "="*80)
    print("TEST 5: Verify README documentation")
    print("="*80)

    readme_path = Path("README.md")
    if not readme_path.exists():
        print("❌ FAIL: README.md not found")
        return False

    content = readme_path.read_text()

    required_sections = [
        ("# Smart Fork", "Title"),
        ("## Installation", "Installation instructions"),
        ("## Usage", "Usage instructions"),
        ("## Configuration", "Configuration documentation"),
        ("pip install", "Installation command"),
    ]

    all_present = True
    for section, description in required_sections:
        if section in content:
            print(f"✓ {description} present")
        else:
            print(f"❌ {description} missing")
            all_present = False

    # Check README length (should be comprehensive)
    lines = content.splitlines()
    print(f"\n✓ README has {len(lines)} lines")

    if len(lines) < 50:
        print("⚠ WARNING: README seems short (< 50 lines)")

    if all_present:
        print("\n✅ PASS: README properly documented")
    else:
        print("\n❌ FAIL: README missing required sections")

    return all_present


def verify_release_checklist():
    """Verify release checklist was created."""
    print("\n" + "="*80)
    print("TEST 6: Verify release checklist")
    print("="*80)

    checklist_path = Path("RELEASE_CHECKLIST.md")
    if not checklist_path.exists():
        print("❌ FAIL: RELEASE_CHECKLIST.md not found")
        return False

    content = checklist_path.read_text()

    required_sections = [
        ("# Smart Fork - Release Checklist", "Title"),
        ("## Pre-Release Verification", "Pre-release section"),
        ("## Release Process", "Release process section"),
        ("### Build and Test", "Build instructions"),
        ("### Git and GitHub", "Git instructions"),
        ("## Version Numbering", "Versioning guide"),
        ("## Rollback Plan", "Rollback procedures"),
    ]

    all_present = True
    for section, description in required_sections:
        if section in content:
            print(f"✓ {description} present")
        else:
            print(f"❌ {description} missing")
            all_present = False

    lines = content.splitlines()
    print(f"\n✓ Checklist has {len(lines)} lines")

    if all_present:
        print("\n✅ PASS: Release checklist properly created")
    else:
        print("\n❌ FAIL: Release checklist incomplete")

    return all_present


def main():
    """Run all offline packaging verification tests."""
    print("\n" + "="*80)
    print("SMART FORK - PACKAGING VERIFICATION (Task 23)")
    print("OFFLINE MODE - No network required")
    print("="*80)
    print("\nVerifying packaging configuration:")
    print("1. pyproject.toml configuration")
    print("2. Entry point script")
    print("3. Package structure")
    print("4. Dependencies")
    print("5. README documentation")
    print("6. Release checklist")

    results = {
        "pyproject.toml configuration": verify_pyproject_toml(),
        "Entry point script": verify_entry_point(),
        "Package structure": verify_package_structure(),
        "Dependencies": verify_dependencies(),
        "README documentation": verify_readme(),
        "Release checklist": verify_release_checklist(),
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
        print("\nNote: Network-dependent tests (pip install, wheel build) require")
        print("network access and should be tested in a connected environment.")
        print("\nThe package structure and configuration are correct.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Please review and fix issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
