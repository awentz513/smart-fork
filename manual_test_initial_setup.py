#!/usr/bin/env python3
"""
Manual test script for InitialSetup class.

This script tests the initial database setup flow without requiring pytest.
"""

import sys
import time
import json
from pathlib import Path
from tempfile import TemporaryDirectory

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from smart_fork.initial_setup import InitialSetup, SetupProgress, SetupState


def test_setup_progress():
    """Test SetupProgress dataclass."""
    print("Test 1: SetupProgress creation...")
    progress = SetupProgress(
        total_files=100,
        processed_files=50,
        current_file="session.jsonl",
        total_chunks=1000,
        elapsed_time=120.0,
        estimated_remaining=120.0
    )
    assert progress.total_files == 100
    assert progress.processed_files == 50
    assert progress.is_complete is False
    print("✓ SetupProgress creation works")

    print("\nTest 2: SetupProgress with completion...")
    progress_complete = SetupProgress(
        total_files=100,
        processed_files=100,
        current_file="",
        total_chunks=2000,
        elapsed_time=300.0,
        estimated_remaining=0.0,
        is_complete=True
    )
    assert progress_complete.is_complete is True
    print("✓ SetupProgress completion flag works")


def test_setup_state():
    """Test SetupState dataclass."""
    print("\nTest 3: SetupState creation...")
    state = SetupState(
        total_files=100,
        processed_files=["file1.jsonl", "file2.jsonl"],
        started_at=time.time(),
        last_updated=time.time()
    )
    assert state.total_files == 100
    assert len(state.processed_files) == 2
    print("✓ SetupState creation works")

    print("\nTest 4: SetupState serialization...")
    data = state.to_dict()
    restored = SetupState.from_dict(data)
    assert restored.total_files == state.total_files
    assert restored.processed_files == state.processed_files
    print("✓ SetupState serialization round-trip works")


def test_initial_setup_init():
    """Test InitialSetup initialization."""
    print("\nTest 5: InitialSetup initialization...")
    setup = InitialSetup()
    assert setup.storage_dir == Path("~/.smart-fork").expanduser()
    assert setup.claude_dir == Path("~/.claude").expanduser()
    assert setup.progress_callback is None
    print("✓ InitialSetup initialization with defaults works")

    print("\nTest 6: InitialSetup with custom paths...")
    with TemporaryDirectory() as tmpdir:
        setup = InitialSetup(
            storage_dir=f"{tmpdir}/storage",
            claude_dir=f"{tmpdir}/claude"
        )
        assert setup.storage_dir == Path(f"{tmpdir}/storage")
        assert setup.claude_dir == Path(f"{tmpdir}/claude")
    print("✓ InitialSetup custom paths work")


def test_first_run_detection():
    """Test first-run detection."""
    print("\nTest 7: First-run detection...")
    with TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir) / "storage"
        setup = InitialSetup(storage_dir=str(storage_dir))

        # Should be first run (dir doesn't exist)
        assert setup.is_first_run() is True

        # Create directory
        storage_dir.mkdir()

        # Should not be first run anymore
        assert setup.is_first_run() is False
    print("✓ First-run detection works")


def test_incomplete_setup_detection():
    """Test incomplete setup detection."""
    print("\nTest 8: Incomplete setup detection...")
    with TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir) / "storage"
        storage_dir.mkdir()
        setup = InitialSetup(storage_dir=str(storage_dir))

        # No state file yet
        assert setup.has_incomplete_setup() is False

        # Create state file
        state_file = storage_dir / "setup_state.json"
        state_file.write_text("{}")

        # Should detect incomplete setup
        assert setup.has_incomplete_setup() is True
    print("✓ Incomplete setup detection works")


def test_find_session_files():
    """Test finding session files."""
    print("\nTest 9: Finding session files...")
    with TemporaryDirectory() as tmpdir:
        claude_dir = Path(tmpdir) / "claude"
        claude_dir.mkdir()

        # Create test files
        (claude_dir / "session1.jsonl").write_text("x" * 200)
        (claude_dir / "session2.jsonl").write_text("x" * 200)
        (claude_dir / "small.jsonl").write_text("x")  # Too small
        (claude_dir / "other.txt").write_text("x" * 200)  # Wrong extension

        setup = InitialSetup(claude_dir=str(claude_dir))
        files = setup._find_session_files()

        assert len(files) == 2
        assert all(f.suffix == ".jsonl" for f in files)
    print("✓ Finding session files works")

    print("\nTest 10: Finding session files recursively...")
    with TemporaryDirectory() as tmpdir:
        claude_dir = Path(tmpdir) / "claude"
        projects_dir = claude_dir / "projects" / "myproject"
        projects_dir.mkdir(parents=True)

        (claude_dir / "session1.jsonl").write_text("x" * 200)
        (projects_dir / "session2.jsonl").write_text("x" * 200)

        setup = InitialSetup(claude_dir=str(claude_dir))
        files = setup._find_session_files()

        assert len(files) == 2
    print("✓ Recursive session file search works")


def test_state_management():
    """Test state save/load/delete."""
    print("\nTest 11: State management...")
    with TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir) / "storage"
        storage_dir.mkdir()
        setup = InitialSetup(storage_dir=str(storage_dir))

        # Create and save state
        started = time.time()
        state = SetupState(
            total_files=100,
            processed_files=["file1.jsonl", "file2.jsonl"],
            started_at=started,
            last_updated=started + 10
        )
        setup._save_state(state)

        # Load state
        loaded = setup._load_state()
        assert loaded is not None
        assert loaded.total_files == 100
        assert len(loaded.processed_files) == 2
        print("✓ State save/load works")

        # Delete state
        setup._delete_state()
        assert not setup.state_file.exists()
    print("✓ State deletion works")


def test_project_extraction():
    """Test project extraction from paths."""
    print("\nTest 12: Project extraction...")
    setup = InitialSetup()

    # With projects directory
    path1 = Path("/home/user/.claude/projects/myproject/sessions/session1.jsonl")
    assert setup._extract_project(path1) == "myproject"

    # Without projects directory
    path2 = Path("/home/user/.claude/session1.jsonl")
    assert setup._extract_project(path2) == "unknown"

    print("✓ Project extraction works")


def test_time_estimation():
    """Test time estimation."""
    print("\nTest 13: Time estimation...")
    setup = InitialSetup()

    # Zero processed
    assert setup._estimate_remaining_time(0, 100, 0.0) == 0.0

    # Half complete
    remaining = setup._estimate_remaining_time(50, 100, 100.0)
    assert remaining == 100.0

    # Nearly complete
    remaining = setup._estimate_remaining_time(99, 100, 99.0)
    assert 0.9 <= remaining <= 1.1

    print("✓ Time estimation works")


def test_progress_notification():
    """Test progress notification."""
    print("\nTest 14: Progress notification...")

    # With callback
    called = []
    def callback(progress):
        called.append(progress)

    setup = InitialSetup(progress_callback=callback)
    setup._notify_progress(
        total=100,
        processed=50,
        current_file="session.jsonl",
        total_chunks=1000,
        start_time=time.time() - 60
    )

    assert len(called) == 1
    assert isinstance(called[0], SetupProgress)
    assert called[0].total_files == 100
    print("✓ Progress notification with callback works")

    # Without callback (should not crash)
    setup_no_callback = InitialSetup(progress_callback=None)
    setup_no_callback._notify_progress(
        total=100,
        processed=50,
        current_file="session.jsonl",
        total_chunks=1000,
        start_time=time.time()
    )
    print("✓ Progress notification without callback works")


def test_interruption():
    """Test graceful interruption."""
    print("\nTest 15: Interruption handling...")
    setup = InitialSetup()
    assert setup._interrupted is False

    setup.interrupt()
    assert setup._interrupted is True
    print("✓ Interruption flag works")


def test_run_setup_no_files():
    """Test running setup with no files."""
    print("\nTest 16: Running setup with no files...")
    with TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir) / "storage"
        claude_dir = Path(tmpdir) / "claude"
        claude_dir.mkdir()

        setup = InitialSetup(
            storage_dir=str(storage_dir),
            claude_dir=str(claude_dir)
        )

        result = setup.run_setup()
        assert result['success'] is True
        assert result['files_processed'] == 0
        assert 'No session files found' in result['message']
    print("✓ Setup with no files works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Manual Tests for InitialSetup")
    print("=" * 60)

    tests = [
        ("SetupProgress dataclass", test_setup_progress),
        ("SetupState dataclass", test_setup_state),
        ("InitialSetup initialization", test_initial_setup_init),
        ("First-run detection", test_first_run_detection),
        ("Incomplete setup detection", test_incomplete_setup_detection),
        ("Finding session files", test_find_session_files),
        ("State management", test_state_management),
        ("Project extraction", test_project_extraction),
        ("Time estimation", test_time_estimation),
        ("Progress notification", test_progress_notification),
        ("Interruption handling", test_interruption),
        ("Running setup", test_run_setup_no_files),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
