<!--
  plan2.md - Gap remediation tasks for Smart Fork Detection project

  CONTEXT:
  - plan.md tasks are all complete (23/23 MVP tasks)
  - This plan addresses gaps identified from PRD vs implementation comparison
  - Gaps discovered during validation testing on 2026-01-21

  STRUCTURE:
  - Each task has: category, description, steps array, priority, and passes boolean
  - priority: P0 = critical/blocking, P1 = high, P2 = medium, P3 = low
  - passes: false = task not yet completed
  - passes: true  = task completed and verified

  USAGE:
  - The ralph.sh automation loop reads this file via PROMPT.md
  - Claude picks the first task where passes is false
  - After completing a task, Claude updates passes to true

  CATEGORIES:
  - bugfix: Fixes for issues found during validation
  - enhancement: Improvements to existing functionality
  - integration: Connecting components that exist but aren't wired together
  - ux: User experience improvements
  - testing: Additional test coverage

  CHANGE LOG:
  - 2026-01-21: Created plan2.md based on PRD gap analysis
-->

# Smart Fork Detection - Gap Remediation Plan (Phase 2)

## Overview
This plan addresses gaps identified between the PRD specification and current implementation. All MVP tasks from plan.md are complete, but validation testing revealed several integration issues and missing UX features.

**Reference:**
- [PRD-Smart-Fork-Detection-2026-01-20.md](PRD-Smart-Fork-Detection-2026-01-20.md)
- [plan.md](plan.md) (Phase 1 - Complete)

**Validation Results:**
- 540 sessions successfully indexed
- Search returns relevant results
- Fork commands generate correctly
- Core architecture is solid (~80% MVP complete)

---

## Gap Summary

| Gap | Priority | Status |
|-----|----------|--------|
| Session parser nested message handling | P0 | Fixed during validation |
| Missing einops dependency | P0 | Fixed during validation |
| Datetime serialization in initial_setup | P0 | Fixed during validation |
| Initial setup progress display | P1 | Pending |
| MCP server tool verification | P1 | Pending |
| Background indexer watchdog integration | P1 | Pending |
| Selection UI integration | P2 | Pending |
| Memory extractor scoring integration | P2 | Pending |
| Graceful setup interruption/resume | P2 | Pending |
| Error states UX | P3 | Pending |

---

## Task List

```json
[
  {
    "category": "bugfix",
    "priority": "P1",
    "description": "Add progress display to initial setup",
    "steps": [
      "Modify InitialSetup.run_setup() to accept progress_callback parameter",
      "Display 'Indexing session X of Y...' during processing",
      "Show file size being processed for large files",
      "Add elapsed time tracking",
      "Add estimated time remaining based on average processing rate",
      "Test progress display with actual session files"
    ],
    "passes": true
  },
  {
    "category": "integration",
    "priority": "P1",
    "description": "Verify and fix MCP server tool registration",
    "steps": [
      "Review server.py MCP tool registration",
      "Ensure search_sessions tool is properly exposed",
      "Ensure get_fork_command tool is properly exposed",
      "Test MCP server responds to tool list request",
      "Test tool invocation returns expected results",
      "Update README if tool names differ from /fork-detect"
    ],
    "passes": true
  },
  {
    "category": "integration",
    "priority": "P1",
    "description": "Verify background indexer watchdog integration",
    "steps": [
      "Review BackgroundIndexer watchdog setup",
      "Verify file monitoring is actually started on server init",
      "Test creating a new session file triggers indexing",
      "Test modifying a session file triggers re-indexing",
      "Verify debounce delay (5 seconds) works correctly",
      "Add logging to confirm watchdog events are captured"
    ],
    "passes": true
  },
  {
    "category": "integration",
    "priority": "P2",
    "description": "Integrate SelectionUI with MCP server flow",
    "steps": [
      "Review selection_ui.py current implementation",
      "Wire SelectionUI into search results presentation",
      "Implement 5-option display (top 3 + None + Type something)",
      "Mark highest-scoring result as 'Recommended'",
      "Handle user selection and return appropriate action",
      "Test selection flow via MCP tool invocation"
    ],
    "passes": true
  },
  {
    "category": "integration",
    "priority": "P2",
    "description": "Integrate MemoryExtractor into scoring pipeline",
    "steps": [
      "Review memory_extractor.py current implementation",
      "Wire MemoryExtractor into chunking/indexing flow",
      "Store detected memory types in chunk metadata",
      "Pass memory types to ScoringService during search",
      "Apply PATTERN (+5%), WORKING_SOLUTION (+8%), WAITING (+2%) boosts",
      "Test that sessions with memory markers rank higher"
    ],
    "passes": true
  },
  {
    "category": "enhancement",
    "priority": "P2",
    "description": "Add graceful interruption and resume to initial setup",
    "steps": [
      "Save setup progress to setup_state.json after each batch",
      "Track which session files have been processed",
      "On startup, check for incomplete setup state",
      "Resume from last processed file if interrupted",
      "Handle Ctrl+C gracefully with state save",
      "Test interruption and resume workflow"
    ],
    "passes": true
  },
  {
    "category": "enhancement",
    "priority": "P2",
    "description": "Add timeout handling for large session processing",
    "steps": [
      "Add configurable timeout per session (default 30s)",
      "Skip sessions that timeout and log warning",
      "Track timed-out sessions for later retry",
      "Add --retry-timeouts flag to initial setup",
      "Test with known large session files",
      "Update README with timeout configuration"
    ],
    "passes": true
  },
  {
    "category": "ux",
    "priority": "P3",
    "description": "Improve error state handling and messages",
    "steps": [
      "Handle 'no sessions found' with helpful message",
      "Handle 'database not initialized' with setup prompt",
      "Handle 'search timeout' with status update",
      "Add suggested actions for each error state",
      "Test all error paths",
      "Update troubleshooting docs with error codes"
    ],
    "passes": true
  },
  {
    "category": "testing",
    "priority": "P2",
    "description": "Add integration tests for MCP tool flow",
    "steps": [
      "Create test fixture simulating MCP client",
      "Test search_sessions tool invocation",
      "Test get_fork_command tool invocation",
      "Test tool error handling",
      "Verify response format matches MCP spec",
      "Add test for full search-select-fork workflow"
    ],
    "passes": true
  },
  {
    "category": "enhancement",
    "priority": "P3",
    "description": "Add session preview capability",
    "steps": [
      "Implement get_session_preview(session_id, length) method",
      "Return first N characters of session content",
      "Include message count and date range",
      "Expose via MCP tool or API endpoint",
      "Allow user to view session before forking",
      "Test preview with various session sizes"
    ],
    "passes": false
  },
  {
    "category": "bugfix",
    "priority": "P1",
    "description": "Fix README MCP configuration accuracy",
    "steps": [
      "Verify actual MCP tool names in server.py",
      "Update README Quick Start with correct tool invocation",
      "Remove /fork-detect references if not implemented as slash command",
      "Document actual MCP tool interface",
      "Add example MCP tool usage",
      "Test README instructions on fresh setup"
    ],
    "passes": true
  }
]
```

---

## Architecture Notes

The core architecture from plan.md remains unchanged. These tasks focus on:

1. **Integration gaps**: Components exist but aren't properly wired together
2. **UX polish**: Progress indicators, error handling, user feedback
3. **Robustness**: Timeout handling, graceful interruption, resume capability

```
Current State:
┌─────────────────────────────────────────────────────────────┐
│  MCP Server (server.py)                                      │
│    └── Tools registered but integration unclear              │
│                                                              │
│  Background Indexer (background_indexer.py)                  │
│    └── Watchdog setup exists but may not be running          │
│                                                              │
│  Selection UI (selection_ui.py)                              │
│    └── Exists but not wired into MCP flow                    │
│                                                              │
│  Memory Extractor (memory_extractor.py)                      │
│    └── Exists but not integrated into scoring pipeline       │
│                                                              │
│  Initial Setup (initial_setup.py)                            │
│    └── Works but lacks progress display and resume           │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Criteria for Phase 2

- [ ] Initial setup shows progress: "Indexing session X of Y (Z%)"
- [ ] MCP tools can be invoked from Claude Code
- [ ] New session files are automatically indexed within 10 seconds
- [ ] Selection UI presents 5 options with recommended badge
- [ ] Memory markers boost session rankings correctly
- [ ] Setup can be interrupted and resumed without data loss
- [ ] Large sessions timeout gracefully instead of hanging
- [ ] All error states have helpful user-facing messages

---

## Validation Fixes Already Applied

The following issues were discovered and fixed during validation testing:

1. **Session parser nested message handling** (session_parser.py:231-232)
   - Issue: Parser didn't handle Claude Code's nested `message.role` and `message.content` structure
   - Fix: Added proper nested dict handling with content extraction

2. **Missing einops dependency** (pyproject.toml)
   - Issue: nomic-embed-text model requires einops package
   - Fix: Added `einops>=0.7.0` to dependencies

3. **Datetime serialization** (initial_setup.py:271)
   - Issue: datetime objects passed to SessionMetadata instead of ISO strings
   - Fix: Convert datetime to `.isoformat()` before storing

---

## Estimated Effort

| Priority | Tasks | Est. Complexity |
|----------|-------|-----------------|
| P1 | 4 tasks | Medium |
| P2 | 5 tasks | Medium-High |
| P3 | 2 tasks | Low |

**Total: 11 tasks**

---

## Notes

- Tasks are ordered by priority (P1 first, then P2, then P3)
- Each task should be completable in a single ralph.sh iteration
- Test validation should be run after completing all P1 tasks
- P2 and P3 tasks can be deferred to v1.1 if time constrained
