<!--
  plan3.md - Phase 3 Feature Plan for Smart Fork Detection

  CONTEXT:
  - plan.md tasks are all complete (23/23 MVP tasks)
  - plan2.md tasks are all complete (11/11 gap remediation tasks)
  - Phase 3 focuses on enhancements and new capabilities

  STRUCTURE:
  - Each task has: category, description, steps array, priority
  - priority: P1 = high value/low effort, P2 = medium, P3 = nice-to-have
  - Tasks are grouped by theme for easier planning

  CATEGORIES:
  - performance: Speed and efficiency improvements
  - ux: User experience enhancements
  - intelligence: Smart features and learning
  - integration: External system connections
  - organization: Session management features
-->

# Smart Fork Detection - Phase 3 Feature Plan

## Overview
Phase 3 focuses on enhancing the core Smart Fork Detection system with performance optimizations, user experience improvements, and intelligent features. These features build on the solid MVP foundation from Phase 1 and 2.

**Reference:**
- [PRD-Smart-Fork-Detection-2026-01-20.md](PRD-Smart-Fork-Detection-2026-01-20.md)
- [plan.md](plan.md) (Phase 1 - Complete)
- [plan2.md](plan2.md) (Phase 2 - Complete)

**Current State:**
- 540+ sessions indexed
- Search, fork, and preview working via MCP
- Background indexing operational
- Progress display and error handling complete

---

## Feature Categories

### P1: High Value / Low Effort

These features provide significant user value with relatively straightforward implementation.

```json
[
  {
    "category": "performance",
    "priority": "P1",
    "description": "Add query result caching",
    "rationale": "Same queries shouldn't re-compute embeddings and search",
    "steps": [
      "Add LRU cache for query embeddings",
      "Add search result cache with TTL",
      "Clear cache on database updates",
      "Add cache hit/miss statistics",
      "Configure cache size via settings"
    ],
    "passes": true
  },
  {
    "category": "ux",
    "priority": "P1",
    "description": "Add fork history tracking",
    "rationale": "Users often want to return to recently forked sessions",
    "steps": [
      "Create fork_history.json storage",
      "Record session_id, timestamp, query on each fork",
      "Add get-fork-history MCP tool",
      "Display last 10 forks on request",
      "Add option to re-fork from history"
    ],
    "passes": true
  },
  {
    "category": "ux",
    "priority": "P1",
    "description": "Add project-scoped search filter",
    "rationale": "Users often want to search within current project only",
    "steps": [
      "Add optional 'project' parameter to fork-detect tool",
      "Auto-detect project from current working directory",
      "Filter search results by project metadata",
      "Show project scope in results display",
      "Allow toggling between all/project scope"
    ],
    "passes": true
  },
  {
    "category": "performance",
    "priority": "P1",
    "description": "Add embedding cache for indexed sessions",
    "rationale": "Re-indexing shouldn't recompute embeddings for unchanged content",
    "steps": [
      "Store chunk hashes with embeddings",
      "Skip embedding computation for unchanged chunks",
      "Implement content-addressable embedding storage",
      "Add cache statistics to initial setup output",
      "Test with large session re-indexing"
    ],
    "passes": true
  }
]
```

### P2: Medium Value / Medium Effort

These features add meaningful capabilities with moderate implementation complexity.

```json
[
  {
    "category": "intelligence",
    "priority": "P2",
    "description": "Learn from user fork selections",
    "rationale": "Improve ranking based on what users actually choose",
    "steps": [
      "Track which result users select (position 1, 2, 3, or custom)",
      "Store selection patterns with query context",
      "Add lightweight preference model",
      "Boost sessions user has forked before",
      "Add user_preference_weight to scoring pipeline"
    ],
    "passes": true
  },
  {
    "category": "ux",
    "priority": "P2",
    "description": "Add temporal search filters",
    "rationale": "Users often remember when they worked on something",
    "steps": [
      "Add 'time_range' parameter (today, this_week, this_month, custom)",
      "Parse natural language dates ('last Tuesday', '2 weeks ago')",
      "Filter results by session created_at timestamp",
      "Show relative time in search results",
      "Add recency boost for temporal queries"
    ],
    "passes": true
  },
  {
    "category": "organization",
    "priority": "P2",
    "description": "Add session tagging capability",
    "rationale": "Let users organize sessions with custom tags",
    "steps": [
      "Add tags field to session metadata",
      "Create add-session-tag MCP tool",
      "Create list-session-tags MCP tool",
      "Enable searching by tag",
      "Auto-suggest tags based on content"
    ],
    "passes": true
  },
  {
    "category": "performance",
    "priority": "P2",
    "description": "Add multi-threaded indexing",
    "rationale": "Initial setup can be slow with many sessions",
    "steps": [
      "Add --workers parameter to initial setup",
      "Implement thread pool for session processing",
      "Add progress tracking across threads",
      "Handle thread-safe database writes",
      "Test with 1000+ sessions"
    ],
    "passes": true
  },
  {
    "category": "intelligence",
    "priority": "P2",
    "description": "Add duplicate session detection",
    "rationale": "Multiple sessions on same topic should be flagged",
    "steps": [
      "Compute session-level embeddings (average of chunks)",
      "Find similar session pairs above threshold",
      "Flag potential duplicates in search results",
      "Add get-similar-sessions MCP tool",
      "Suggest session merging for high similarity"
    ],
    "passes": true
  }
]
```

### P3: Nice-to-Have / Higher Effort

These features are valuable but require more significant implementation work.

```json
[
  {
    "category": "organization",
    "priority": "P3",
    "description": "Add automatic topic clustering",
    "rationale": "Group related sessions for easier browsing",
    "steps": [
      "Implement k-means or HDBSCAN clustering on session embeddings",
      "Auto-generate cluster labels from content",
      "Create get-session-clusters MCP tool",
      "Allow browsing sessions by cluster",
      "Update clusters on new session indexing"
    ],
    "passes": false
  },
  {
    "category": "integration",
    "priority": "P3",
    "description": "Add VS Code extension integration",
    "rationale": "Native IDE experience for fork detection",
    "steps": [
      "Create VS Code extension scaffold",
      "Add command palette integration",
      "Show search results in sidebar panel",
      "One-click fork from extension",
      "Display fork history in activity bar"
    ],
    "passes": false
  },
  {
    "category": "intelligence",
    "priority": "P3",
    "description": "Add session summarization",
    "rationale": "Quick overview without reading full session",
    "steps": [
      "Implement extractive summarization using key sentences",
      "Add summary field to session metadata",
      "Generate summaries during indexing",
      "Show summary in search results",
      "Add get-session-summary MCP tool"
    ],
    "passes": false
  },
  {
    "category": "ux",
    "priority": "P3",
    "description": "Add session diff tool",
    "rationale": "Compare two sessions to understand differences",
    "steps": [
      "Create compare-sessions MCP tool",
      "Compute semantic similarity between messages",
      "Highlight unique content in each session",
      "Show conversation flow comparison",
      "Display technology/topic differences"
    ],
    "passes": false
  },
  {
    "category": "organization",
    "priority": "P3",
    "description": "Add session archiving",
    "rationale": "Keep database performant by archiving old sessions",
    "steps": [
      "Add archive threshold setting (e.g., sessions > 1 year old)",
      "Create archive database separate from active",
      "Move old sessions to archive on schedule",
      "Add --include-archive flag to search",
      "Implement archive restore capability"
    ],
    "passes": false
  }
]
```

---

## Architecture Considerations

### For P1 Performance Features

```
Cache Layer Architecture:
┌─────────────────────────────────────────────────────────────┐
│  Query Request                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                       │
│  │  Query Cache    │──hit──▶ Return cached results          │
│  │  (LRU, TTL=5m)  │                                       │
│  └─────────────────┘                                       │
│       │ miss                                                │
│       ▼                                                     │
│  ┌─────────────────┐                                       │
│  │ Embedding Cache │──hit──▶ Use cached embedding           │
│  │ (content hash)  │                                       │
│  └─────────────────┘                                       │
│       │ miss                                                │
│       ▼                                                     │
│  ┌─────────────────┐                                       │
│  │ Compute & Store │                                       │
│  └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### For Intelligence Features

```
Learning Feedback Loop:
┌─────────────────────────────────────────────────────────────┐
│  User Query ──▶ Search Results ──▶ User Selection           │
│                      │                     │                │
│                      ▼                     │                │
│              ┌──────────────┐              │                │
│              │ Preference   │◀─────────────┘                │
│              │ Tracker      │                               │
│              └──────────────┘                               │
│                      │                                      │
│                      ▼                                      │
│              ┌──────────────┐                               │
│              │ Scoring      │ preference_boost applied      │
│              │ Pipeline     │ to future searches            │
│              └──────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

| Feature | Success Criteria |
|---------|------------------|
| Query caching | 50%+ cache hit rate after warmup |
| Fork history | Used at least once per day |
| Project filter | 30%+ of searches use project scope |
| Embedding cache | 80%+ cache hit on re-index |
| Preference learning | Selection position improves over time |
| Temporal search | Natural language dates parse correctly |
| Session tagging | Users create 2+ tags per week |
| Multi-threaded indexing | 3x+ speedup with 4 workers |

---

## Implementation Order

Recommended order for Phase 3:

1. **Query result caching** (P1) - Immediate perf improvement
2. **Fork history tracking** (P1) - High user value, simple
3. **Project-scoped search** (P1) - Frequently requested
4. **Embedding cache** (P1) - Reduces initial setup time
5. **Temporal search** (P2) - Natural user workflow
6. **Session tagging** (P2) - Organization foundation
7. **Learning from selections** (P2) - Continuous improvement
8. **Multi-threaded indexing** (P2) - Large repo support

---

## Notes

- P1 tasks are designed to be completable in 1-2 iterations each
- P2 tasks may require 2-3 iterations
- P3 tasks are stretch goals for future consideration
- All features should maintain backwards compatibility
- Consider feature flags for gradual rollout
