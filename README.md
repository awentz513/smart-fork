# Smart Fork Detection

An MCP (Model Context Protocol) server for Claude Code that enables semantic search of past session transcripts and intelligent session forking. Never lose context again - find and resume from the most relevant previous conversation instantly.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Smart Fork Detection solves the "context loss" problem in AI-assisted development by maintaining a searchable vector database of all your Claude Code sessions. When you need to work on a similar task or continue where you left off, simply search your conversation history and fork from the most relevant session - with full context preserved.

**Key Benefits:**
- **Overcome Context Limits**: Break free from the 200,000 token limit by intelligent session forking
- **Instant Context Recovery**: Find relevant past conversations in seconds instead of re-explaining everything
- **Knowledge Reuse**: Transform hundreds of isolated sessions into connected, searchable knowledge
- **Productivity Boost**: Reduce context rebuilding time from minutes to seconds

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [/fork-detect Command](#fork-detect-command)
  - [Selecting a Session](#selecting-a-session)
  - [Forking a Session](#forking-a-session)
- [Configuration](#configuration)
  - [Configuration Options](#configuration-options)
  - [Configuration File](#configuration-file)
- [How It Works](#how-it-works)
  - [Background Indexing](#background-indexing)
  - [Semantic Search](#semantic-search)
  - [Composite Scoring](#composite-scoring)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Performance Tuning](#performance-tuning)
- [Privacy & Security](#privacy--security)
- [Example Usage Scenarios](#example-usage-scenarios)
- [Advanced Topics](#advanced-topics)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.10 or higher
- Claude Code (with MCP support)
- 2GB+ RAM recommended for embedding model
- 500MB+ disk space for vector database

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/your-org/smart-fork.git
cd smart-fork
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

### Install from PyPI (coming soon)

```bash
pip install smart-fork
```

### Verify Installation

Run the verification script to ensure everything is set up correctly:
```bash
python -c "import smart_fork; print(smart_fork.__version__)"
```

### Configure Claude Code MCP

Add Smart Fork to your Claude Code MCP configuration file (`~/.claude/mcp_config.json`):

```json
{
  "mcpServers": {
    "smart-fork": {
      "command": "smart-fork",
      "args": [],
      "env": {}
    }
  }
}
```

Restart Claude Code to load the MCP server.

## Quick Start

1. **Start Claude Code** - The Smart Fork server will automatically start in the background and begin indexing your existing sessions.

2. **First Run Setup** - On first launch, Smart Fork will scan `~/.claude/` for existing session files and build the initial database. This may take a few minutes depending on how many sessions you have.

3. **Invoke the Tool** - In any Claude Code session, type:
   ```
   /fork-detect
   ```

4. **Describe Your Task** - Enter a natural language description:
   ```
   I want to add real-time dashboard updates with WebSocket support
   ```

5. **Select a Session** - Choose from the top 5 most relevant past sessions, or start fresh.

6. **Fork and Continue** - Copy the generated command and paste it in a new terminal to continue from that session with full context.

## Usage

### /fork-detect Command

The `/fork-detect` command is your main interface to Smart Fork. It searches your session history and helps you find the most relevant previous conversation.

**Syntax:**
```
/fork-detect
```

When invoked, you'll be prompted to describe what you want to do in natural language.

**Example queries:**
- "implement user authentication with JWT"
- "fix database connection pooling issues"
- "add dark mode to the settings page"
- "refactor API error handling"
- "optimize React component rendering"

### Selecting a Session

After searching, Smart Fork displays exactly 5 options:

1. **Top 3 Results** - The most relevant sessions based on composite scoring
2. **None - start fresh** - Begin a new session without forking
3. **Type something else** - Refine your search with a different query

Each result shows:
- **Session ID**: Unique identifier for the session
- **Date**: When the session was created
- **Project**: Project name (extracted from file path)
- **Score**: Relevance percentage (0-100%)
- **Preview**: Snippet from the most relevant part of the conversation
- **⭐ Recommended**: The highest-scoring result

**Example output:**
```
Found 5 relevant sessions:

⭐ [1] Session abc123 (92% match) - Recommended
   Date: 2026-01-15
   Project: my-dashboard
   Preview: "Implemented real-time updates using WebSocket connection with
            automatic reconnection logic..."

[2] Session def456 (81% match)
   Date: 2026-01-10
   Project: my-dashboard
   Preview: "Added dashboard component with live data updates and polling
            fallback..."

[3] Session ghi789 (67% match)
   Date: 2025-12-20
   Project: admin-portal
   Preview: "Created WebSocket handler for server-sent events with proper
            error handling..."

[4] None - start fresh

[5] Type something else
```

### Forking a Session

When you select a session, Smart Fork generates two types of fork commands:

**1. New Terminal Fork** (Recommended)
```bash
claude --resume abc123 --fork-session
```
Opens a new Claude Code session continuing from the selected conversation.

**2. In-Session Fork** (Advanced)
```bash
/fork abc123 /path/to/project
```
Forks within the current session (if supported by your Claude Code version).

Simply copy the command and paste it in a new terminal to continue with full context from that session.

## Configuration

Smart Fork works out-of-the-box with sensible defaults, but you can customize its behavior.

### Configuration Options

Smart Fork uses a configuration file at `~/.smart-fork/config.json`. The file is created automatically with default values on first run.

#### Embedding Model Settings

```json
"embedding": {
  "model_name": "nomic-ai/nomic-embed-text-v1.5",
  "dimension": 768,
  "batch_size": 32,
  "max_batch_size": 128,
  "min_batch_size": 8
}
```

- **model_name**: HuggingFace model identifier for embeddings
- **dimension**: Embedding vector dimensions (must match model)
- **batch_size**: Default batch size for embedding generation
- **max_batch_size**: Maximum batch size (auto-adjusted based on RAM)
- **min_batch_size**: Minimum batch size (prevents too-small batches)

#### Search Parameters

```json
"search": {
  "k_chunks": 200,
  "top_n_sessions": 5,
  "preview_length": 200,
  "similarity_threshold": 0.3,
  "recency_weight": 0.25
}
```

- **k_chunks**: Number of chunks to retrieve from vector database
- **top_n_sessions**: Number of session results to display
- **preview_length**: Character limit for preview snippets
- **similarity_threshold**: Minimum similarity score (0.0-1.0)
- **recency_weight**: Weight given to recent sessions in scoring

#### Chunking Settings

```json
"chunking": {
  "target_tokens": 750,
  "overlap_tokens": 150,
  "max_tokens": 1000
}
```

- **target_tokens**: Target size for each chunk
- **overlap_tokens**: Overlap between adjacent chunks (maintains context)
- **max_tokens**: Maximum chunk size (hard limit)

#### Background Indexing

```json
"indexing": {
  "debounce_delay": 5.0,
  "checkpoint_interval": 15,
  "enabled": true
}
```

- **debounce_delay**: Seconds to wait after file modification before indexing
- **checkpoint_interval**: Index after this many new messages (prevents loss)
- **enabled**: Enable/disable background indexing

#### Server Settings

```json
"server": {
  "host": "127.0.0.1",
  "port": 8741
}
```

- **host**: Bind address (always localhost for security)
- **port**: Port for local REST API server

#### Memory Management

```json
"memory": {
  "max_memory_mb": 2000,
  "gc_between_batches": true
}
```

- **max_memory_mb**: Maximum memory usage target in megabytes
- **gc_between_batches**: Run garbage collection between embedding batches

#### Storage Directory

```json
"storage_dir": "~/.smart-fork"
```

- **storage_dir**: Directory for database and registry files

### Configuration File

Create or edit `~/.smart-fork/config.json`:

```json
{
  "embedding": {
    "model_name": "nomic-ai/nomic-embed-text-v1.5",
    "dimension": 768,
    "batch_size": 32,
    "max_batch_size": 128,
    "min_batch_size": 8
  },
  "search": {
    "k_chunks": 200,
    "top_n_sessions": 5,
    "preview_length": 200,
    "similarity_threshold": 0.3,
    "recency_weight": 0.25
  },
  "chunking": {
    "target_tokens": 750,
    "overlap_tokens": 150,
    "max_tokens": 1000
  },
  "indexing": {
    "debounce_delay": 5.0,
    "checkpoint_interval": 15,
    "enabled": true
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8741
  },
  "memory": {
    "max_memory_mb": 2000,
    "gc_between_batches": true
  },
  "storage_dir": "~/.smart-fork"
}
```

Changes take effect after restarting Claude Code.

## How It Works

### Background Indexing

Smart Fork continuously monitors `~/.claude/` for new or modified session files:

1. **File Monitoring**: Uses the `watchdog` library to detect file system changes
2. **Debouncing**: Waits 5 seconds after the last modification before indexing (configurable)
3. **Checkpoint Indexing**: Indexes sessions every 10-20 messages to prevent data loss
4. **Graceful Processing**: Handles rapid successive changes without duplication

Session files are parsed, chunked, embedded, and stored in the vector database automatically.

### Semantic Search

When you invoke `/fork-detect`, Smart Fork:

1. **Embeds Your Query**: Converts your natural language description to a 768-dimensional vector
2. **Vector Search**: Finds the 200 most similar chunks using ChromaDB's k-NN search
3. **Groups by Session**: Aggregates chunks by their parent session
4. **Scores Sessions**: Calculates composite scores for each session
5. **Ranks Results**: Returns the top 5 sessions, sorted by relevance

### Composite Scoring

Each session receives a composite score based on multiple factors:

```
Final Score = (best_similarity × 0.40)
            + (avg_similarity × 0.20)
            + (chunk_ratio × 0.05)
            + (recency × 0.25)
            + (chain_quality × 0.10)
```

**Scoring Components:**

- **Best Similarity (40%)**: Highest similarity score among matched chunks
- **Average Similarity (20%)**: Mean similarity across all matched chunks
- **Chunk Ratio (5%)**: Proportion of session chunks that matched
- **Recency (25%)**: Exponential decay based on session age (30-day half-life)
- **Chain Quality (10%)**: Placeholder for future conversation quality metrics (currently 0.5)

**Memory Type Boosts:**

Sessions containing Claude memory markers receive bonus scores:
- **PATTERN** (e.g., "design pattern", "approach", "architecture"): +5%
- **WORKING_SOLUTION** (e.g., "tested", "verified", "successful"): +8%
- **WAITING** (e.g., "todo", "pending", "in progress"): +2%

These boosts help prioritize sessions with proven solutions and documented patterns.

## Troubleshooting

### Common Issues

#### "No sessions found" error

**Cause**: Database is empty or hasn't finished initial indexing.

**Solutions:**
1. Wait for initial indexing to complete (check `~/.smart-fork/session-registry.json`)
2. Verify session files exist in `~/.claude/`
3. Check logs for indexing errors

#### Search returns irrelevant results

**Cause**: Query may be too vague or database needs more sessions.

**Solutions:**
1. Use more specific queries with technical terms
2. Try different phrasing
3. Use the "Type something else" option to refine
4. Adjust `similarity_threshold` in config (lower = more results)

#### High memory usage

**Cause**: Embedding model or large batches consuming RAM.

**Solutions:**
1. Reduce `max_batch_size` in config (e.g., to 64 or 32)
2. Lower `max_memory_mb` to trigger more aggressive batch sizing
3. Close other applications to free memory
4. Enable `gc_between_batches` if disabled

#### Slow search performance

**Cause**: Large database or insufficient resources.

**Solutions:**
1. Reduce `k_chunks` in config (e.g., to 100)
2. Increase `similarity_threshold` to filter more aggressively
3. Check system resources (CPU, RAM, disk I/O)
4. Consider using a faster machine for large databases

#### Background indexing not working

**Cause**: File monitoring may have failed or is disabled.

**Solutions:**
1. Check that `indexing.enabled` is `true` in config
2. Verify `~/.claude/` directory exists and is readable
3. Restart Claude Code to reinitialize the MCP server
4. Check logs for watchdog errors

#### Config changes not taking effect

**Cause**: Configuration is loaded once at startup.

**Solutions:**
1. Restart Claude Code after changing config
2. Verify config file has valid JSON syntax
3. Check file permissions on `~/.smart-fork/config.json`

### Performance Tuning

**For systems with limited RAM (< 8GB):**
```json
{
  "embedding": {
    "batch_size": 16,
    "max_batch_size": 32
  },
  "memory": {
    "max_memory_mb": 1000,
    "gc_between_batches": true
  }
}
```

**For high-performance systems (16GB+ RAM):**
```json
{
  "embedding": {
    "batch_size": 64,
    "max_batch_size": 256
  },
  "search": {
    "k_chunks": 300
  },
  "memory": {
    "max_memory_mb": 4000
  }
}
```

**For faster search at the cost of accuracy:**
```json
{
  "search": {
    "k_chunks": 100,
    "similarity_threshold": 0.5
  }
}
```

## Privacy & Security

### Data Storage

All data is stored locally on your machine:
- **Vector Database**: `~/.smart-fork/vector_db/` (ChromaDB)
- **Session Registry**: `~/.smart-fork/session-registry.json`
- **Configuration**: `~/.smart-fork/config.json`

**No data is ever sent to external servers** (except for downloading the embedding model on first run).

### Network Security

- The REST API server binds exclusively to `127.0.0.1` (localhost)
- No external network access is possible
- Only processes on your local machine can access the API

### Session Privacy

- Session files in `~/.claude/` may contain sensitive information
- The vector database stores embeddings (semantic representations) but not full text
- Session metadata (project, timestamps, chunk counts) is stored in the registry
- Preview snippets are generated on-demand from matched chunks

### Best Practices

1. **Secure Your Machine**: Use full-disk encryption and strong user passwords
2. **Backup Carefully**: If backing up `~/.smart-fork/`, treat it as sensitive data
3. **Review Before Forking**: Check preview snippets to avoid leaking sensitive context
4. **Clean Old Sessions**: Periodically delete session files you no longer need
5. **Environment Variables**: Avoid storing secrets in sessions (use `.env` files instead)

### Data Deletion

To completely remove Smart Fork data:

```bash
# Stop Claude Code first
rm -rf ~/.smart-fork/
```

This deletes the vector database, session registry, and configuration. Your original session files in `~/.claude/` remain untouched.

## Example Usage Scenarios

### Scenario 1: Implementing a Similar Feature

**Context**: You previously built a real-time dashboard with WebSockets. Now you need to add live notifications to a different project.

```
You: /fork-detect
Smart Fork: What would you like to do?
You: implement real-time notifications with WebSocket

Results:
⭐ [1] Session abc123 (94% match) - Recommended
   Date: 2026-01-10
   Project: my-dashboard
   Preview: "Set up WebSocket connection with automatic reconnection,
            heartbeat ping/pong, and exponential backoff..."

You: (select option 1)

Smart Fork: Fork command generated!
Terminal command: claude --resume abc123 --fork-session

You: (paste command in new terminal)
Claude: (continues with full context from the dashboard implementation)
```

**Result**: You save 20+ minutes of re-explaining WebSocket setup, reconnection logic, error handling, etc.

### Scenario 2: Debugging a Recurring Issue

**Context**: You fixed a memory leak in your app 3 weeks ago. A similar issue appeared in a different component.

```
You: /fork-detect
Smart Fork: What would you like to do?
You: fix memory leak in React components

Results:
⭐ [1] Session def456 (89% match) - Recommended
   Date: 2025-12-28
   Project: web-app
   Preview: "Identified memory leak caused by missing cleanup in useEffect.
            Added return function to cancel subscriptions and clear intervals..."

You: (select option 1)
```

**Result**: Claude remembers the exact debugging approach, tools used, and solution pattern from the previous fix.

### Scenario 3: Continuing After Context Limit

**Context**: You're in a long refactoring session that hit the 200k token limit.

```
You: /fork-detect
Smart Fork: What would you like to do?
You: continue refactoring authentication module

Results:
⭐ [1] Session ghi789 (98% match) - Recommended
   Date: Today, 2 hours ago
   Project: api-server
   Preview: "Refactoring auth module: completed user service, working on
            token validation. Next: implement refresh token rotation..."

You: (select option 1)
```

**Result**: Seamlessly continue refactoring with all architectural decisions and progress context intact.

### Scenario 4: Testing Multiple Approaches

**Context**: You want to try different UI frameworks for the same feature.

```
# First approach
You: /fork-detect
You: implement settings page with form validation
You: (select session with React + Formik)

# Later, try another approach
You: /fork-detect
You: implement settings page with form validation
You: (select same baseline session)
You: Actually, let's use Vue 3 with Vuelidate instead
```

**Result**: Test multiple approaches from the same baseline without losing context or duplicating setup work.

### Scenario 5: Onboarding to a New Project

**Context**: A new team member needs to understand your project's patterns.

```
You: /fork-detect
You: understand project structure and coding patterns

Results:
⭐ [1] Session jkl012 (85% match) - Recommended
   Date: 2025-11-15
   Project: api-server
   Preview: "Explained project architecture: 3-tier design with controllers,
            services, and repositories. Error handling uses custom exception
            classes..."
```

**Result**: New developers can fork from architecture discussions to get context-aware guidance.

## Advanced Topics

### Manual Session Indexing

To manually trigger indexing of a specific session:

```python
from smart_fork.background_indexer import BackgroundIndexer
from smart_fork.embedding_service import EmbeddingService
from smart_fork.vector_db_service import VectorDBService
from smart_fork.session_registry import SessionRegistry

# Initialize services
embedding_service = EmbeddingService()
vector_db = VectorDBService()
registry = SessionRegistry()

# Create indexer
indexer = BackgroundIndexer(
    embedding_service=embedding_service,
    vector_db=vector_db,
    registry=registry
)

# Index a specific file
indexer.index_file("/path/to/session.jsonl")
```

### Querying the Database Programmatically

```python
from smart_fork.search_service import SearchService

# Initialize search service
search_service = SearchService(
    embedding_service=embedding_service,
    vector_db=vector_db,
    scoring_service=scoring_service,
    registry=registry
)

# Search for sessions
results = search_service.search(
    query="implement user authentication",
    k_chunks=200,
    top_n_sessions=5
)

for result in results:
    print(f"{result.session_id}: {result.score:.2%} - {result.preview[:100]}")
```

### Custom Embedding Models

To use a different embedding model, update your config:

```json
{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384
  }
}
```

**Note**: Changing the model requires re-indexing all sessions (the dimension must match).

### Database Statistics

Check database statistics:

```python
from smart_fork.vector_db_service import VectorDBService

vector_db = VectorDBService()
stats = vector_db.get_stats()

print(f"Total chunks: {stats['total_chunks']}")
print(f"Total sessions: {stats['total_sessions']}")
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Additional embedding models support
- Performance optimizations
- UI/UX improvements
- Documentation enhancements
- Bug fixes and testing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Troubleshooting?** Check the [Troubleshooting](#troubleshooting) section above or [open an issue](https://github.com/your-org/smart-fork/issues).

**Questions?** Join our [discussions](https://github.com/your-org/smart-fork/discussions) or reach out to the team.
