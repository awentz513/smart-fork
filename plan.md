# Smart Fork Detection - Project Plan

## Overview
An MCP server for Claude Code that enables semantic search of past session transcripts and intelligent session forking. The system maintains a local vector database of all Claude Code sessions, allowing developers to quickly find and resume work from the most relevant previous conversation.

**Reference:** [PRD-Smart-Fork-Detection-2026-01-20.md](PRD-Smart-Fork-Detection-2026-01-20.md)

---

## Task List

```json
[
  {
    "category": "setup",
    "description": "Initialize project structure and dependencies",
    "steps": [
      "Create project directory structure (src/, tests/, configs/)",
      "Initialize Python project with pyproject.toml",
      "Install core dependencies: fastmcp, chromadb, sentence-transformers, fastapi, watchdog, psutil",
      "Create virtual environment and verify imports",
      "Set up storage directory structure (~/.smart-fork/)"
    ],
    "passes": true
  },
  {
    "category": "setup",
    "description": "Configure MCP server boilerplate",
    "steps": [
      "Create main MCP server entry point using FastMCP",
      "Register placeholder /fork-detect tool",
      "Test server starts and responds to MCP protocol",
      "Verify Claude Code can connect to the server"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement session file parser (JSONL reader)",
    "steps": [
      "Create SessionParser class for reading .jsonl files",
      "Handle UTF-8 encoding and malformed JSON lines gracefully",
      "Extract message content, timestamps, and roles",
      "Add support for incomplete/crashed sessions",
      "Write unit tests for parser edge cases"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement semantic chunking algorithm",
    "steps": [
      "Create ChunkingService class",
      "Implement token counting (target 750 tokens per chunk)",
      "Ensure code blocks are never split mid-block",
      "Keep conversation turns together (user + assistant)",
      "Add 150-token overlap between adjacent chunks",
      "Write unit tests verifying chunk boundaries"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement Nomic embedding integration",
    "steps": [
      "Create EmbeddingService class wrapping sentence-transformers",
      "Load nomic-embed-text-v1.5 model (768 dimensions)",
      "Implement adaptive batch sizing based on available RAM",
      "Add memory monitoring and garbage collection per batch",
      "Verify embeddings generate without system lockup",
      "Write integration test with sample text"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Create ChromaDB wrapper with chunk storage",
    "steps": [
      "Create VectorDBService class wrapping ChromaDB",
      "Initialize persistent collection at ~/.smart-fork/vector_db/",
      "Implement add_chunks() method with metadata",
      "Implement search_chunks() method returning top k results",
      "Implement delete_session_chunks() for re-indexing",
      "Write integration tests for CRUD operations"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Build session registry manager",
    "steps": [
      "Create SessionRegistry class",
      "Implement JSON storage at ~/.smart-fork/session-registry.json",
      "Track session metadata: project, timestamps, chunk count, tags",
      "Implement get/update/delete session methods",
      "Add last_synced timestamp tracking",
      "Write unit tests for registry operations"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement composite scoring algorithm",
    "steps": [
      "Create ScoringService class",
      "Implement best_similarity calculation (40% weight)",
      "Implement avg_similarity calculation (20% weight)",
      "Implement chunk_ratio calculation (5% weight)",
      "Implement recency decay with 30-day constant (25% weight)",
      "Add chain_quality placeholder at 0.5 (10% weight)",
      "Implement memory type boosting (PATTERN, WORKING_SOLUTION, WAITING)",
      "Write unit tests verifying score calculations"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Build search service with ranking",
    "steps": [
      "Create SearchService class orchestrating embedding + vector search",
      "Perform k-nearest neighbors search (k=200 chunks)",
      "Group chunks by session_id",
      "Calculate composite scores per session",
      "Return top 5 sessions with metadata and previews",
      "Ensure search completes in <3 seconds",
      "Write integration tests with sample indexed data"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement background indexing service",
    "steps": [
      "Create BackgroundIndexer class with async support",
      "Implement file monitoring using watchdog for ~/.claude/",
      "Add debouncing (5-second delay after last modification)",
      "Process sessions in background thread pool",
      "Implement checkpoint indexing every 10-20 messages",
      "Add graceful handling of rapid successive changes",
      "Write integration tests simulating file changes"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Build local REST API server",
    "steps": [
      "Create FastAPI app binding to localhost:8741",
      "Implement POST /chunks/search endpoint",
      "Implement POST /sessions/index endpoint",
      "Implement GET /sessions/{session_id} endpoint",
      "Implement GET /stats endpoint",
      "Ensure server only accessible from 127.0.0.1",
      "Write API endpoint tests"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement /fork-detect MCP command handler",
    "steps": [
      "Register /fork-detect as MCP tool",
      "Prompt user: 'What would you like to do?'",
      "Accept natural language description input",
      "Call SearchService to find relevant sessions",
      "Return formatted results within 3 seconds",
      "Handle empty/no-results case gracefully"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Create interactive selection UI",
    "steps": [
      "Display exactly 5 options: top 3 results + 'None' + 'Type something'",
      "Mark highest-scoring result as 'Recommended'",
      "Show session ID, date, topic, preview snippet, and score %",
      "Implement 'Chat about this' discussion option",
      "Handle keyboard navigation (Enter, arrows, Esc)",
      "Test selection flow end-to-end"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Build fork command generator (dual modes)",
    "steps": [
      "Generate new terminal command: claude --resume [id] --fork-session",
      "Generate in-session command: /fork [id] [path]",
      "Display session metadata before forking",
      "Ensure commands are copy-paste ready",
      "Show success confirmation with execution time",
      "Test both forking modes"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement initial database setup flow",
    "steps": [
      "Detect first-run (no ~/.smart-fork/ directory)",
      "Scan ~/.claude/ for all existing session files",
      "Display progress: 'Indexing session X of Y...'",
      "Show estimated time remaining",
      "Support graceful interruption and resume",
      "Create session registry on completion",
      "Test full initial setup flow"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement Claude memory marker extraction",
    "steps": [
      "Create MemoryExtractor class",
      "Detect PATTERN markers in session content",
      "Detect WORKING_SOLUTION markers",
      "Detect WAITING markers",
      "Store memory types in chunk metadata",
      "Apply memory-based score boosting",
      "Write unit tests for marker detection"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Add configuration management",
    "steps": [
      "Create config.json schema at ~/.smart-fork/config.json",
      "Implement default configuration values",
      "Support embedding model selection",
      "Support batch size and memory limits",
      "Support search parameters (threshold, recency weight)",
      "Support server port configuration",
      "Write config validation tests"
    ],
    "passes": true
  },
  {
    "category": "testing",
    "description": "Unit test coverage for core services",
    "steps": [
      "Test SessionParser with valid/invalid JSONL",
      "Test ChunkingService boundary detection",
      "Test EmbeddingService output dimensions",
      "Test ScoringService calculations match formula",
      "Test SessionRegistry CRUD operations",
      "Achieve 80%+ code coverage on core modules"
    ],
    "passes": true
  },
  {
    "category": "testing",
    "description": "Integration tests for search flow",
    "steps": [
      "Create test fixture with 10 sample sessions",
      "Index test sessions into ChromaDB",
      "Run search queries and verify ranking order",
      "Verify results return within 3-second target",
      "Test memory boost affects ranking correctly",
      "Test recency factor affects ranking correctly"
    ],
    "passes": true
  },
  {
    "category": "testing",
    "description": "End-to-end testing of /fork-detect workflow",
    "steps": [
      "Simulate user invoking /fork-detect",
      "Provide test query and verify results display",
      "Select a result and verify fork command generated",
      "Test 'None - start fresh' option",
      "Test 'Type something' refinement option",
      "Verify all error states handled gracefully"
    ],
    "passes": true
  },
  {
    "category": "testing",
    "description": "Performance and stress testing",
    "steps": [
      "Test indexing 1000+ messages without RAM exhaustion",
      "Test search with 10,000 chunks in database",
      "Verify search latency <3s at 95th percentile",
      "Test concurrent indexing and searching",
      "Monitor memory usage stays under 2GB",
      "Test database size scaling (~500KB per 1000 messages)"
    ],
    "passes": false
  },
  {
    "category": "documentation",
    "description": "Write user documentation",
    "steps": [
      "Create README with installation instructions",
      "Document /fork-detect command usage",
      "Document configuration options",
      "Add troubleshooting guide for common issues",
      "Include privacy best practices",
      "Add example usage scenarios"
    ],
    "passes": false
  },
  {
    "category": "deployment",
    "description": "Package for distribution",
    "steps": [
      "Configure pyproject.toml for pip installation",
      "Add entry point script for MCP server",
      "Test pip install from local source",
      "Test pip install from built wheel",
      "Verify all dependencies bundled correctly",
      "Create release checklist"
    ],
    "passes": false
  }
]
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code Client                    │
└────────────────────┬────────────────────────────────────┘
                     │ MCP Protocol
┌────────────────────▼────────────────────────────────────┐
│              Smart Fork MCP Server                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Command Handler (/fork-detect)           │   │
│  └───────────┬──────────────────────────────────────┘   │
│              │                                           │
│  ┌───────────▼──────────────┐  ┌────────────────────┐   │
│  │   EmbeddingService       │  │  SearchService     │   │
│  │   (Nomic 768-dim)        │  │  (Composite Score) │   │
│  └───────────┬──────────────┘  └─────────┬──────────┘   │
│              │                           │              │
│  ┌───────────▼───────────────────────────▼──────────┐   │
│  │           Local REST API (localhost:8741)        │   │
│  └───────────┬──────────────────────────────────────┘   │
└──────────────┼──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│                  Storage Layer                           │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────┐   │
│  │  ChromaDB      │  │   Session    │  │   Config   │   │
│  │ vector_db/     │  │   Registry   │  │    JSON    │   │
│  └────────────────┘  └──────────────┘  └────────────┘   │
│                 ~/.smart-fork/                           │
└─────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.10+ | Best ecosystem for ML/embeddings |
| MCP Framework | FastMCP | Official Python SDK |
| Vector DB | ChromaDB | Local, lightweight, excellent Python integration |
| Embedding Model | nomic-embed-text-v1.5 | 768-dim, high quality for technical text, runs locally |
| API Framework | FastAPI | Async, high performance |
| File Monitoring | watchdog | Cross-platform file system events |

---

## Scoring Formula

```
Final Score = (best_similarity × 0.40)
            + (avg_similarity × 0.20)
            + (chunk_ratio × 0.05)
            + (recency × 0.25)
            + (chain_quality × 0.10)
```

**Memory Type Boosts:**
- PATTERN: +5%
- WORKING_SOLUTION: +8%
- WAITING: +2%

---

## Success Criteria

- [ ] Search latency < 3 seconds for 95th percentile
- [ ] Indexing throughput > 100 messages/second
- [ ] Memory usage < 2GB with 50,000 chunks
- [ ] Top-1 accuracy: First result relevant 80% of time
- [ ] Top-3 accuracy: At least one relevant 95% of time
- [ ] Zero data loss from indexing failures

---

## Phase 1 MVP Scope

**In Scope:**
- `/fork-detect` command with search and selection
- Background session indexing with file monitoring
- Composite scoring with 5 factors
- Dual forking modes (new terminal + in-session)
- Initial database setup flow
- Basic configuration

**Out of Scope (v1.1+):**
- Chain quality tracking (uses placeholder 0.5)
- Search filtering by project/date/tags
- Session preview viewer
- Team sharing features
