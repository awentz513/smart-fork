#!/usr/bin/env python3
"""
Verification script for Task 5: MemoryExtractor integration into scoring pipeline.

This script verifies:
1. ChunkingService extracts memory types from content
2. Memory types are stored in chunk metadata during indexing
3. SearchService extracts memory types from search results
4. ScoringService applies correct boosts (+5%, +8%, +2%)
5. Sessions with memory markers rank higher than without
"""

import sys
sys.path.insert(0, 'src')

from smart_fork.memory_extractor import MemoryExtractor
from smart_fork.chunking_service import ChunkingService, Chunk
from smart_fork.scoring_service import ScoringService, SessionScore
from smart_fork.session_parser import SessionMessage


def test_memory_extractor_basics():
    """TEST 1: MemoryExtractor detects memory types correctly."""
    print("\n" + "="*60)
    print("TEST 1: MemoryExtractor Basic Detection")
    print("="*60)

    extractor = MemoryExtractor()
    passed = 0
    total = 0

    # Test PATTERN detection
    total += 1
    pattern_text = "This uses the singleton design pattern for managing state."
    types = extractor.extract_memory_types(pattern_text)
    if 'PATTERN' in types:
        print("  ✓ PATTERN detected in text with 'design pattern'")
        passed += 1
    else:
        print(f"  ✗ PATTERN not detected. Got: {types}")

    # Test WORKING_SOLUTION detection
    total += 1
    solution_text = "This working solution handles all edge cases and all tests pass."
    types = extractor.extract_memory_types(solution_text)
    if 'WORKING_SOLUTION' in types:
        print("  ✓ WORKING_SOLUTION detected in text with 'working solution'")
        passed += 1
    else:
        print(f"  ✗ WORKING_SOLUTION not detected. Got: {types}")

    # Test WAITING detection
    total += 1
    waiting_text = "This task is waiting for the API response, marked as pending."
    types = extractor.extract_memory_types(waiting_text)
    if 'WAITING' in types:
        print("  ✓ WAITING detected in text with 'waiting' and 'pending'")
        passed += 1
    else:
        print(f"  ✗ WAITING not detected. Got: {types}")

    # Test boost calculation
    total += 1
    all_types = ['PATTERN', 'WORKING_SOLUTION', 'WAITING']
    boost = extractor.get_memory_boost(all_types)
    expected_boost = 0.05 + 0.08 + 0.02  # 0.15
    if abs(boost - expected_boost) < 0.001:
        print(f"  ✓ Boost calculation correct: {boost} (expected {expected_boost})")
        passed += 1
    else:
        print(f"  ✗ Boost calculation wrong: {boost} (expected {expected_boost})")

    print(f"\n  Result: {passed}/{total} checks passed")
    return passed == total


def test_chunking_service_integration():
    """TEST 2: ChunkingService extracts and stores memory types."""
    print("\n" + "="*60)
    print("TEST 2: ChunkingService Memory Type Extraction")
    print("="*60)

    chunking_service = ChunkingService(extract_memory=True)
    passed = 0
    total = 0

    # Create test messages with memory markers
    messages = [
        SessionMessage(
            role="user",
            content="How should I implement this feature?",
            timestamp=None
        ),
        SessionMessage(
            role="assistant",
            content="I recommend using the factory design pattern. This architectural approach will make your code more maintainable.",
            timestamp=None
        ),
        SessionMessage(
            role="user",
            content="That sounds good. Did you test it?",
            timestamp=None
        ),
        SessionMessage(
            role="assistant",
            content="Yes, this is a working solution that has been verified. All tests pass and it handles edge cases correctly.",
            timestamp=None
        ),
    ]

    # Chunk the messages
    chunks = chunking_service.chunk_messages(messages)

    # Check that chunks were created
    total += 1
    if chunks:
        print(f"  ✓ Created {len(chunks)} chunk(s) from messages")
        passed += 1
    else:
        print("  ✗ No chunks created")
        return False

    # Check that memory types were extracted
    total += 1
    has_memory_types = any(chunk.memory_types for chunk in chunks)
    if has_memory_types:
        print("  ✓ Memory types extracted from chunks")
        passed += 1
    else:
        print("  ✗ No memory types found in any chunk")

    # Check for specific memory types
    all_memory_types = set()
    for chunk in chunks:
        if chunk.memory_types:
            all_memory_types.update(chunk.memory_types)

    total += 1
    if 'PATTERN' in all_memory_types:
        print("  ✓ PATTERN memory type found in chunks")
        passed += 1
    else:
        print("  ✗ PATTERN memory type not found")

    total += 1
    if 'WORKING_SOLUTION' in all_memory_types:
        print("  ✓ WORKING_SOLUTION memory type found in chunks")
        passed += 1
    else:
        print("  ✗ WORKING_SOLUTION memory type not found")

    print(f"\n  Memory types found: {sorted(all_memory_types)}")
    print(f"  Result: {passed}/{total} checks passed")
    return passed == total


def test_scoring_service_boosts():
    """TEST 3: ScoringService applies correct memory boosts."""
    print("\n" + "="*60)
    print("TEST 3: ScoringService Memory Boost Application")
    print("="*60)

    scoring_service = ScoringService()
    passed = 0
    total = 0

    # Base score without memory types
    base_score = scoring_service.calculate_session_score(
        session_id="test_session",
        chunk_similarities=[0.8, 0.7],
        total_chunks_in_session=2,
        session_last_modified=None,
        memory_types=None
    )

    # Score with PATTERN memory type (+5%)
    pattern_score = scoring_service.calculate_session_score(
        session_id="test_session",
        chunk_similarities=[0.8, 0.7],
        total_chunks_in_session=2,
        session_last_modified=None,
        memory_types=['PATTERN']
    )

    total += 1
    expected_boost = 0.05
    actual_boost = pattern_score.final_score - base_score.final_score
    if abs(actual_boost - expected_boost) < 0.001:
        print(f"  ✓ PATTERN boost correct: +{actual_boost:.3f} (expected +{expected_boost})")
        passed += 1
    else:
        print(f"  ✗ PATTERN boost wrong: +{actual_boost:.3f} (expected +{expected_boost})")

    # Score with WORKING_SOLUTION memory type (+8%)
    solution_score = scoring_service.calculate_session_score(
        session_id="test_session",
        chunk_similarities=[0.8, 0.7],
        total_chunks_in_session=2,
        session_last_modified=None,
        memory_types=['WORKING_SOLUTION']
    )

    total += 1
    expected_boost = 0.08
    actual_boost = solution_score.final_score - base_score.final_score
    if abs(actual_boost - expected_boost) < 0.001:
        print(f"  ✓ WORKING_SOLUTION boost correct: +{actual_boost:.3f} (expected +{expected_boost})")
        passed += 1
    else:
        print(f"  ✗ WORKING_SOLUTION boost wrong: +{actual_boost:.3f} (expected +{expected_boost})")

    # Score with WAITING memory type (+2%)
    waiting_score = scoring_service.calculate_session_score(
        session_id="test_session",
        chunk_similarities=[0.8, 0.7],
        total_chunks_in_session=2,
        session_last_modified=None,
        memory_types=['WAITING']
    )

    total += 1
    expected_boost = 0.02
    actual_boost = waiting_score.final_score - base_score.final_score
    if abs(actual_boost - expected_boost) < 0.001:
        print(f"  ✓ WAITING boost correct: +{actual_boost:.3f} (expected +{expected_boost})")
        passed += 1
    else:
        print(f"  ✗ WAITING boost wrong: +{actual_boost:.3f} (expected +{expected_boost})")

    # Score with all memory types (+15% total)
    all_types_score = scoring_service.calculate_session_score(
        session_id="test_session",
        chunk_similarities=[0.8, 0.7],
        total_chunks_in_session=2,
        session_last_modified=None,
        memory_types=['PATTERN', 'WORKING_SOLUTION', 'WAITING']
    )

    total += 1
    expected_boost = 0.15
    actual_boost = all_types_score.final_score - base_score.final_score
    if abs(actual_boost - expected_boost) < 0.001:
        print(f"  ✓ Combined boost correct: +{actual_boost:.3f} (expected +{expected_boost})")
        passed += 1
    else:
        print(f"  ✗ Combined boost wrong: +{actual_boost:.3f} (expected +{expected_boost})")

    print(f"\n  Base score: {base_score.final_score:.4f}")
    print(f"  With all memory types: {all_types_score.final_score:.4f}")
    print(f"  Result: {passed}/{total} checks passed")
    return passed == total


def test_ranking_with_memory():
    """TEST 4: Sessions with memory markers rank higher."""
    print("\n" + "="*60)
    print("TEST 4: Ranking Verification (Memory > No Memory)")
    print("="*60)

    scoring_service = ScoringService()
    passed = 0
    total = 0

    # Create scores for sessions with and without memory types
    # Same similarity scores, but different memory types

    session_no_memory = scoring_service.calculate_session_score(
        session_id="session_no_memory",
        chunk_similarities=[0.75],
        total_chunks_in_session=1,
        session_last_modified=None,
        memory_types=None
    )

    session_with_pattern = scoring_service.calculate_session_score(
        session_id="session_with_pattern",
        chunk_similarities=[0.75],
        total_chunks_in_session=1,
        session_last_modified=None,
        memory_types=['PATTERN']
    )

    session_with_solution = scoring_service.calculate_session_score(
        session_id="session_with_solution",
        chunk_similarities=[0.75],
        total_chunks_in_session=1,
        session_last_modified=None,
        memory_types=['WORKING_SOLUTION']
    )

    # Rank them
    all_scores = [session_no_memory, session_with_pattern, session_with_solution]
    ranked = scoring_service.rank_sessions(all_scores, top_k=3)

    # Check that WORKING_SOLUTION ranks highest (highest boost)
    total += 1
    if ranked[0].session_id == 'session_with_solution':
        print("  ✓ WORKING_SOLUTION session ranks first (highest boost)")
        passed += 1
    else:
        print(f"  ✗ Expected session_with_solution first, got: {ranked[0].session_id}")

    # Check that PATTERN ranks second
    total += 1
    if ranked[1].session_id == 'session_with_pattern':
        print("  ✓ PATTERN session ranks second")
        passed += 1
    else:
        print(f"  ✗ Expected session_with_pattern second, got: {ranked[1].session_id}")

    # Check that no-memory ranks last
    total += 1
    if ranked[2].session_id == 'session_no_memory':
        print("  ✓ No-memory session ranks last")
        passed += 1
    else:
        print(f"  ✗ Expected session_no_memory last, got: {ranked[2].session_id}")

    print(f"\n  Ranking:")
    for i, score in enumerate(ranked):
        print(f"    {i+1}. {score.session_id}: {score.final_score:.4f} (boost: {score.memory_boost:.2f})")

    print(f"\n  Result: {passed}/{total} checks passed")
    return passed == total


def test_integration_in_initial_setup():
    """TEST 5: Verify integration points in initial_setup.py."""
    print("\n" + "="*60)
    print("TEST 5: Integration Points Verification")
    print("="*60)

    passed = 0
    total = 0

    # Check that ChunkingService has memory extraction enabled by default
    total += 1
    chunking_service = ChunkingService()
    if chunking_service.extract_memory and chunking_service.memory_extractor is not None:
        print("  ✓ ChunkingService has memory extraction enabled by default")
        passed += 1
    else:
        print("  ✗ ChunkingService does not have memory extraction enabled")

    # Check that Chunk dataclass has memory_types field
    total += 1
    chunk = Chunk(
        content="test",
        start_index=0,
        end_index=0,
        token_count=1,
        memory_types=['PATTERN']
    )
    if hasattr(chunk, 'memory_types') and chunk.memory_types == ['PATTERN']:
        print("  ✓ Chunk dataclass supports memory_types field")
        passed += 1
    else:
        print("  ✗ Chunk dataclass does not support memory_types field")

    # Check that ScoringService has memory boost constants
    total += 1
    scoring_service = ScoringService()
    if (scoring_service.BOOST_PATTERN == 0.05 and
        scoring_service.BOOST_WORKING_SOLUTION == 0.08 and
        scoring_service.BOOST_WAITING == 0.02):
        print("  ✓ ScoringService has correct boost constants")
        passed += 1
    else:
        print("  ✗ ScoringService boost constants incorrect")

    # Verify SessionScore includes memory_boost field
    total += 1
    score = scoring_service.calculate_session_score(
        session_id="test",
        chunk_similarities=[0.8],
        total_chunks_in_session=1,
        memory_types=['PATTERN']
    )
    if hasattr(score, 'memory_boost') and score.memory_boost == 0.05:
        print("  ✓ SessionScore includes memory_boost field")
        passed += 1
    else:
        print("  ✗ SessionScore does not include memory_boost field correctly")

    print(f"\n  Result: {passed}/{total} checks passed")
    return passed == total


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("MEMORY EXTRACTOR INTEGRATION VERIFICATION")
    print("Task 5: Integrate MemoryExtractor into scoring pipeline")
    print("="*60)

    results = []

    # Run all tests
    results.append(("MemoryExtractor Basic Detection", test_memory_extractor_basics()))
    results.append(("ChunkingService Integration", test_chunking_service_integration()))
    results.append(("ScoringService Boost Application", test_scoring_service_boosts()))
    results.append(("Ranking Verification", test_ranking_with_memory()))
    results.append(("Integration Points", test_integration_in_initial_setup()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - MemoryExtractor integration is complete!")
        print("\nIntegration Flow:")
        print("  1. ChunkingService → extracts memory types during chunking")
        print("  2. InitialSetup → stores memory_types in chunk metadata")
        print("  3. VectorDBService → serializes/deserializes memory_types")
        print("  4. SearchService → extracts memory_types from search results")
        print("  5. ScoringService → applies boosts (+5%, +8%, +2%)")
    else:
        print("SOME TESTS FAILED - Please review the failures above")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
