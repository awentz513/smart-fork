"""Main MCP server entry point for Smart Fork."""

import json
import sys
import os
import logging
import signal
import atexit
from typing import Any, Dict, List, Optional
from pathlib import Path

from .embedding_service import EmbeddingService
from .vector_db_service import VectorDBService
from .scoring_service import ScoringService
from .session_registry import SessionRegistry
from .search_service import SearchService
from .selection_ui import SelectionUI
from .background_indexer import BackgroundIndexer
from .session_parser import SessionParser
from .chunking_service import ChunkingService
from .fork_generator import ForkGenerator
from .cache_service import CacheService
from .config_manager import ConfigManager
from .fork_history_service import ForkHistoryService
from .preference_service import PreferenceService
from .session_tag_service import SessionTagService
from .duplicate_detection_service import DuplicateDetectionService
from .session_clustering_service import SessionClusteringService
from .session_summary_service import SessionSummaryService
from .session_diff_service import SessionDiffService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


class MCPServer:
    """Basic MCP server implementing JSON-RPC 2.0 over stdio."""

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        background_indexer: Optional[BackgroundIndexer] = None
    ) -> None:
        """Initialize the MCP server."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.server_info = {
            "name": "smart-fork",
            "version": "0.1.0"
        }
        self.search_service = search_service
        self.background_indexer = background_indexer

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Any
    ) -> None:
        """Register a tool with the MCP server."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler
        }

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": self.server_info
        }

    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools_list = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"]
            }
            for tool in self.tools.values()
        ]
        return {"tools": tools_list}

    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self.tools[tool_name]["handler"]
        result = handler(arguments)

        return {
            "content": [
                {
                    "type": "text",
                    "text": result
                }
            ]
        }

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle a JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            elif method == "notifications/initialized":
                # Notification, no response needed
                return None
            else:
                raise ValueError(f"Unknown method: {method}")

            if request_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            return None

        except Exception as e:
            if request_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
            return None

    def run(self) -> None:
        """Run the MCP server on stdio."""
        # Write server started message to stderr for debugging
        print("Smart Fork MCP Server started", file=sys.stderr)
        print(f"Server info: {self.server_info}", file=sys.stderr)
        print(f"Registered tools: {list(self.tools.keys())}", file=sys.stderr)

        # Process requests from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)

                if response is not None:
                    print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error handling request: {e}", file=sys.stderr)


def format_search_results_with_selection(
    query: str,
    results: List[Any],
    claude_dir: Optional[str] = None,
    session_registry: Optional[Any] = None,
    project_scope: Optional[str] = None
) -> str:
    """
    Format search results with interactive selection UI.

    Args:
        query: Search query
        results: List of search results
        claude_dir: Optional path to Claude directory (for ForkGenerator)
        session_registry: Optional SessionRegistry for database stats
        project_scope: Optional project scope description to display

    Returns:
        Formatted selection prompt
    """
    # Create ForkGenerator for generating fork commands
    fork_generator = ForkGenerator(claude_sessions_dir=claude_dir or "~/.claude")
    selection_ui = SelectionUI(fork_generator=fork_generator)

    if not results:
        # Get database stats if available
        stats_info = ""
        setup_command = "python -m smart_fork.initial_setup"

        if session_registry:
            try:
                stats = session_registry.get_stats()
                total_sessions = stats.get('total_sessions', 0)
                if total_sessions == 0:
                    stats_info = f"\n⚠️  Database Status: Empty (0 sessions indexed)\n"
                else:
                    stats_info = f"\n📊 Database Status: {total_sessions} sessions indexed\n"
            except Exception:
                pass

        # Show no results message with options to refine or start fresh
        scope_info = f"\nScope: {project_scope}\n" if project_scope else ""
        return f"""Fork Detection - No Results Found

Your query: {query}{scope_info}
{stats_info}
No relevant sessions were found in the database.

This could mean:
- The database is empty or not yet indexed
- Your query doesn't match any existing sessions
- Try rephrasing your query with different keywords

💡 Suggested Actions:
1. If database is empty, run: {setup_command}
2. Try broader search terms (e.g., "authentication" instead of "OAuth JWT middleware")
3. Search for technologies used (e.g., "React", "FastAPI", "TypeScript")
4. Search for problem types (e.g., "bug fix", "performance", "testing")

Options:
1. ❌ None of these - start fresh
2. 🔍 Type something else

Tip: The system searches through all your past Claude Code sessions to find relevant work.
"""

    # Display selection UI
    selection_data = selection_ui.display_selection(
        results,
        query,
        project_scope=project_scope
    )
    return selection_data['prompt']


def detect_project_from_cwd(cwd: Optional[str] = None) -> Optional[str]:
    """
    Detect project name from current working directory.

    Converts a file path to Claude's project naming scheme.
    Example: /Users/foo/Documents/MyProject -> -Users-foo-Documents-MyProject

    Args:
        cwd: Current working directory (defaults to os.getcwd())

    Returns:
        Project name in Claude's format, or None if detection fails
    """
    if cwd is None:
        cwd = os.getcwd()

    try:
        # Convert path to absolute and resolve symlinks
        abs_path = Path(cwd).resolve()

        # Convert path to Claude's project naming format
        # Replace path separators with hyphens, remove leading slash
        path_str = str(abs_path)

        # For macOS/Linux: /Users/foo/Documents/Project -> -Users-foo-Documents-Project
        # For Windows: C:\Users\foo\Documents\Project -> -C-Users-foo-Documents-Project
        project_name = path_str.replace(os.sep, '-')

        # Ensure it starts with a hyphen
        if not project_name.startswith('-'):
            project_name = '-' + project_name

        logger.debug(f"Detected project from CWD: {cwd} -> {project_name}")
        return project_name

    except Exception as e:
        logger.warning(f"Failed to detect project from CWD '{cwd}': {e}")
        return None


def create_fork_detect_handler(
    search_service: Optional[SearchService],
    claude_dir: Optional[str] = None,
    session_registry: Optional[Any] = None
):
    """
    Create the fork-detect handler with access to search service.

    Args:
        search_service: SearchService instance for performing searches
        claude_dir: Optional path to Claude directory (for ForkGenerator)
        session_registry: Optional SessionRegistry for database stats
    """
    def fork_detect_handler(arguments: Dict[str, Any]) -> str:
        """Handler for /fork-detect tool."""
        query = arguments.get("query", "")
        project_param = arguments.get("project")
        scope = arguments.get("scope", "all")
        time_range = arguments.get("time_range")
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")
        tags = arguments.get("tags")
        include_archive = arguments.get("include_archive", False)

        if not query:
            return "Error: Please provide a query describing what you want to do."

        if search_service is None:
            setup_command = "python -m smart_fork.initial_setup"
            return f"""Fork Detection (Service Not Initialized)

Your query: {query}

⚠️  The search service is not yet initialized.

Common Causes:
- Vector database is not set up (needs initial indexing)
- Required dependencies are not installed
- Database files are corrupted or missing

💡 Suggested Actions:
1. Run initial setup to index your sessions: {setup_command}
2. Check that dependencies are installed: pip install -e .
3. Verify database files exist: ~/.smart-fork/chroma-db/
4. Check logs for specific errors

Need help? See: README.md > Troubleshooting
"""

        try:
            # Determine project filter based on parameters
            filter_metadata = None
            project_display = None

            # Handle project parameter
            if project_param:
                if project_param.lower() == "current":
                    # Auto-detect from CWD
                    detected_project = detect_project_from_cwd()
                    if detected_project:
                        filter_metadata = {"project": detected_project}
                        project_display = f"Current Project ({detected_project})"
                    else:
                        logger.warning("Failed to auto-detect project from CWD")
                        project_display = "All Projects (auto-detection failed)"
                else:
                    # Use explicit project name
                    filter_metadata = {"project": project_param}
                    project_display = f"Project: {project_param}"
            elif scope == "project":
                # scope=project means auto-detect current project
                detected_project = detect_project_from_cwd()
                if detected_project:
                    filter_metadata = {"project": detected_project}
                    project_display = f"Current Project ({detected_project})"
                else:
                    logger.warning("Failed to auto-detect project from CWD for scope=project")
                    project_display = "All Projects (auto-detection failed)"
            else:
                project_display = "All Projects"

            # Perform search with temporal parameters
            temporal_desc = ""
            if time_range:
                temporal_desc = f", time: {time_range}"
            elif start_date or end_date:
                temporal_desc = f", time: {start_date or ''} to {end_date or ''}"

            # Add tags info to description if provided
            tags_desc = ""
            if tags:
                tags_list = [t.strip() for t in tags.split(",")] if isinstance(tags, str) else tags
                tags_desc = f", tags: {', '.join(tags_list)}"

            logger.info(f"Processing fork-detect query: {query} (scope: {project_display}{temporal_desc}{tags_desc}, include_archive: {include_archive})")
            results = search_service.search(
                query,
                top_n=5,
                filter_metadata=filter_metadata,
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
                include_archive=include_archive
            )

            # Filter by tags if provided
            if tags:
                # Parse tags (can be comma-separated string or list)
                if isinstance(tags, str):
                    tags_list = [t.strip().lower() for t in tags.split(",")]
                else:
                    tags_list = [t.strip().lower() for t in tags]

                # Filter results to only include sessions with matching tags
                filtered_results = []
                for result in results:
                    if result.metadata and result.metadata.tags:
                        session_tags = [t.lower() for t in result.metadata.tags]
                        # Session must have at least one matching tag
                        if any(tag in session_tags for tag in tags_list):
                            filtered_results.append(result)

                results = filtered_results
                logger.info(f"Filtered to {len(results)} results matching tags: {tags_list}")

            # Format and return results with selection UI (including fork commands)
            formatted_output = format_search_results_with_selection(
                query,
                results,
                claude_dir=claude_dir,
                session_registry=session_registry,
                project_scope=project_display
            )
            logger.info(f"Returned {len(results)} results for query with selection UI")

            return formatted_output

        except TimeoutError as e:
            logger.warning(f"Search timeout for query: {query}")
            return f"""Fork Detection - Search Timeout

Your query: {query}

⏱️  The search operation timed out.

This usually happens when:
- The database is very large (>10,000 sessions)
- The query is too complex or ambiguous
- System resources are constrained

💡 Suggested Actions:
1. Try a simpler, more specific query
2. Use exact technology names (e.g., "React useState hook" not "state management")
3. Search for specific file types or patterns
4. Check system resources (CPU, memory)

The search was stopped to prevent hanging. Try refining your query.
"""
        except Exception as e:
            logger.error(f"Error in fork-detect handler: {e}", exc_info=True)
            error_type = type(e).__name__
            return f"""Fork Detection - Error ({error_type})

Your query: {query}

❌ An error occurred while searching:
{str(e)}

💡 Suggested Actions:
1. Check the server logs for detailed error information
2. Verify the database is not corrupted: ls ~/.smart-fork/chroma-db/
3. Try restarting the MCP server
4. If the error persists, try re-running initial setup

Need help? See: README.md > Troubleshooting
"""

    return fork_detect_handler


def initialize_services(storage_dir: Optional[str] = None) -> tuple[Optional[SearchService], Optional[BackgroundIndexer]]:
    """
    Initialize all required services for the MCP server.

    Args:
        storage_dir: Directory for storing database and registry (default: ~/.smart-fork)

    Returns:
        Tuple of (SearchService, BackgroundIndexer) if initialization succeeds, (None, None) otherwise
    """
    try:
        # Determine storage directory
        if storage_dir is None:
            home = Path.home()
            storage_dir = str(home / ".smart-fork")

        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)

        vector_db_path = storage_path / "vector_db"
        registry_path = storage_path / "session-registry.json"

        logger.info(f"Initializing services with storage: {storage_dir}")

        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load()

        # Initialize cache service
        cache_service = None
        if config.cache.enabled:
            cache_service = CacheService(
                embedding_cache_size=config.cache.embedding_cache_size,
                embedding_ttl_seconds=config.cache.embedding_ttl_seconds,
                result_cache_size=config.cache.result_cache_size,
                result_ttl_seconds=config.cache.result_ttl_seconds
            )
            logger.info("Cache service initialized")

        # Initialize services
        embedding_service = EmbeddingService()
        vector_db_service = VectorDBService(
            persist_directory=str(vector_db_path),
            cache_service=cache_service
        )
        scoring_service = ScoringService()
        session_registry = SessionRegistry(registry_path=str(registry_path))
        chunking_service = ChunkingService()
        session_parser = SessionParser()
        preference_service = PreferenceService()
        logger.info("Preference service initialized")

        # Create search service
        search_service = SearchService(
            embedding_service=embedding_service,
            vector_db_service=vector_db_service,
            scoring_service=scoring_service,
            session_registry=session_registry,
            k_chunks=config.search.k_chunks,
            top_n_sessions=config.search.top_n_sessions,
            preview_length=config.search.preview_length,
            cache_service=cache_service,
            enable_cache=config.cache.enabled,
            preference_service=preference_service,
            enable_preferences=True
        )

        # Get Claude directory to monitor
        claude_dir = Path.home() / ".claude"

        # Initialize summary service
        summary_service = SessionSummaryService()

        # Create background indexer
        background_indexer = BackgroundIndexer(
            claude_dir=claude_dir,
            vector_db=vector_db_service,
            session_registry=session_registry,
            embedding_service=embedding_service,
            chunking_service=chunking_service,
            session_parser=session_parser,
            debounce_seconds=5.0,
            checkpoint_interval=15,
            summary_service=summary_service
        )

        logger.info("Services initialized successfully")
        return search_service, background_indexer

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        return None, None


def create_session_preview_handler(
    search_service: Optional[SearchService],
    claude_dir: Optional[str] = None
):
    """
    Create the get-session-preview handler.

    Args:
        search_service: SearchService instance for accessing session data
        claude_dir: Optional path to Claude directory
    """
    def session_preview_handler(arguments: Dict[str, Any]) -> str:
        """Handler for get-session-preview tool."""
        session_id = arguments.get("session_id", "")
        length = arguments.get("length", 500)

        if not session_id:
            return "Error: Please provide a session_id."

        if search_service is None:
            return "Error: Search service is not initialized. Run initial setup first."

        try:
            preview_data = search_service.get_session_preview(
                session_id,
                length,
                claude_dir=claude_dir
            )

            if preview_data is None:
                return f"Error: Session '{session_id}' not found or could not be read."

            # Format the preview response
            date_info = ""
            if preview_data.get('date_range'):
                date_range = preview_data['date_range']
                date_info = f"\nDate Range: {date_range.get('start', 'Unknown')} to {date_range.get('end', 'Unknown')}"

            message_count = preview_data.get('message_count', 0)
            preview_text = preview_data.get('preview', '')

            return f"""Session Preview: {session_id}

Messages: {message_count}{date_info}

Preview:
{preview_text}

---
Use this information to decide if you want to fork from this session.
"""

        except Exception as e:
            logger.error(f"Error in session-preview handler: {e}", exc_info=True)
            return f"Error: Failed to get session preview: {str(e)}"

    return session_preview_handler


def create_record_fork_handler(
    fork_history_service: Optional[ForkHistoryService],
    preference_service: Optional[PreferenceService]
):
    """
    Create the record-fork handler for tracking fork history and preferences.

    Args:
        fork_history_service: ForkHistoryService instance for tracking forks
        preference_service: PreferenceService instance for learning preferences
    """
    def record_fork_handler(arguments: Dict[str, Any]) -> str:
        """Handler for record-fork tool."""
        session_id = arguments.get("session_id", "")
        query = arguments.get("query", "")
        position = arguments.get("position", -1)

        if not session_id:
            return "Error: Please provide a session_id."

        if fork_history_service is None:
            return "Error: Fork history service is not initialized."

        try:
            # Record in fork history service
            fork_history_service.record_fork(
                session_id=session_id,
                query=query,
                position=position
            )

            # Also record in preference service for learning
            if preference_service is not None:
                preference_service.record_selection(
                    session_id=session_id,
                    query=query,
                    position=position
                )

            return f"Fork recorded successfully for session {session_id}"

        except Exception as e:
            logger.error(f"Error recording fork: {e}", exc_info=True)
            return f"Error: Failed to record fork: {str(e)}"

    return record_fork_handler


def create_fork_history_handler(fork_history_service: Optional[ForkHistoryService]):
    """
    Create the get-fork-history handler for retrieving fork history.

    Args:
        fork_history_service: ForkHistoryService instance for accessing history
    """
    def fork_history_handler(arguments: Dict[str, Any]) -> str:
        """Handler for get-fork-history tool."""
        limit = arguments.get("limit", 10)

        if fork_history_service is None:
            return "Error: Fork history service is not initialized."

        try:
            recent_forks = fork_history_service.get_recent_forks(limit=limit)
            stats = fork_history_service.get_stats()

            if not recent_forks:
                return """Fork History - No History Yet

You haven't forked any sessions yet.

When you fork from a session, it will be recorded here for easy reference.
This helps you:
- Quickly return to recently used sessions
- See what queries led to successful forks
- Track your most useful sessions over time
"""

            # Format the history output
            output = f"""Fork History - Last {len(recent_forks)} Forks

Total forks recorded: {stats['total_forks']}
Unique sessions forked: {stats['unique_sessions']}

Recent Forks:
"""
            for i, entry in enumerate(recent_forks, 1):
                # Parse and format timestamp
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = entry.timestamp

                # Format position
                if entry.position == -1:
                    position_str = "custom"
                elif entry.position >= 1:
                    position_str = f"#{entry.position}"
                else:
                    position_str = "unknown"

                # Truncate query if too long
                query_display = entry.query[:60] + "..." if len(entry.query) > 60 else entry.query

                output += f"""
{i}. {entry.session_id}
   Time: {time_str}
   Query: {query_display}
   Position: {position_str}
"""

            output += """
---
Use these session IDs with get-session-preview to see their content,
or fork from them again using the generated fork commands.
"""

            return output

        except Exception as e:
            logger.error(f"Error getting fork history: {e}", exc_info=True)
            return f"Error: Failed to get fork history: {str(e)}"

    return fork_history_handler


def create_add_tag_handler(tag_service: Optional[SessionTagService]):
    """
    Create the add-session-tag handler.

    Args:
        tag_service: SessionTagService instance for managing tags
    """
    def add_tag_handler(arguments: Dict[str, Any]) -> str:
        """Handler for add-session-tag tool."""
        session_id = arguments.get("session_id", "")
        tag = arguments.get("tag", "")

        if not session_id:
            return "Error: Please provide a session_id."

        if not tag:
            return "Error: Please provide a tag."

        if tag_service is None:
            return "Error: Tag service is not initialized."

        try:
            success = tag_service.add_tag(session_id, tag)

            if success:
                # Get current tags to show updated list
                current_tags = tag_service.get_session_tags(session_id)
                tags_display = ", ".join(current_tags) if current_tags else "none"
                return f"Tag '{tag}' added successfully to session {session_id}.\nCurrent tags: {tags_display}"
            else:
                # Check if session exists
                if tag_service.get_session_tags(session_id) is None:
                    return f"Error: Session '{session_id}' not found."
                else:
                    return f"Tag '{tag}' is already on session {session_id}."

        except Exception as e:
            logger.error(f"Error adding tag: {e}", exc_info=True)
            return f"Error: Failed to add tag: {str(e)}"

    return add_tag_handler


def create_remove_tag_handler(tag_service: Optional[SessionTagService]):
    """
    Create the remove-session-tag handler.

    Args:
        tag_service: SessionTagService instance for managing tags
    """
    def remove_tag_handler(arguments: Dict[str, Any]) -> str:
        """Handler for remove-session-tag tool."""
        session_id = arguments.get("session_id", "")
        tag = arguments.get("tag", "")

        if not session_id:
            return "Error: Please provide a session_id."

        if not tag:
            return "Error: Please provide a tag."

        if tag_service is None:
            return "Error: Tag service is not initialized."

        try:
            success = tag_service.remove_tag(session_id, tag)

            if success:
                # Get current tags to show updated list
                current_tags = tag_service.get_session_tags(session_id)
                tags_display = ", ".join(current_tags) if current_tags else "none"
                return f"Tag '{tag}' removed successfully from session {session_id}.\nCurrent tags: {tags_display}"
            else:
                # Check if session exists
                if tag_service.get_session_tags(session_id) is None:
                    return f"Error: Session '{session_id}' not found."
                else:
                    return f"Tag '{tag}' not found on session {session_id}."

        except Exception as e:
            logger.error(f"Error removing tag: {e}", exc_info=True)
            return f"Error: Failed to remove tag: {str(e)}"

    return remove_tag_handler


def create_list_tags_handler(tag_service: Optional[SessionTagService]):
    """
    Create the list-session-tags handler.

    Args:
        tag_service: SessionTagService instance for managing tags
    """
    def list_tags_handler(arguments: Dict[str, Any]) -> str:
        """Handler for list-session-tags tool."""
        session_id = arguments.get("session_id")
        show_all = arguments.get("show_all", False)

        if tag_service is None:
            return "Error: Tag service is not initialized."

        try:
            if session_id:
                # List tags for a specific session
                tags = tag_service.get_session_tags(session_id)

                if tags is None:
                    return f"Error: Session '{session_id}' not found."

                if not tags:
                    # Suggest some tags
                    suggestions = tag_service.suggest_tags(session_id)
                    if suggestions:
                        suggestions_display = ", ".join(suggestions)
                        return f"""Session {session_id} has no tags yet.

Suggested tags (based on common tags in other sessions):
{suggestions_display}

Use add-session-tag to add tags to this session."""
                    else:
                        return f"Session {session_id} has no tags yet. Use add-session-tag to add tags."

                tags_display = ", ".join(tags)
                return f"Session {session_id} tags: {tags_display}"

            else:
                # List all tags across all sessions
                if show_all:
                    all_tags = tag_service.list_all_tags()

                    if not all_tags:
                        return """No tags found in the system yet.

Tags help organize and categorize your sessions. Use add-session-tag to start tagging sessions."""

                    output = "All Tags (with usage counts):\n\n"
                    for tag_info in all_tags:
                        tag_name = tag_info["tag"]
                        count = tag_info["count"]
                        output += f"  {tag_name}: {count} session(s)\n"

                    # Add stats
                    stats = tag_service.get_stats()
                    output += f"""
Statistics:
  Total sessions: {stats['total_sessions']}
  Tagged sessions: {stats['tagged_sessions']}
  Untagged sessions: {stats['untagged_sessions']}
  Unique tags: {stats['unique_tags']}
  Avg tags per session: {stats['avg_tags_per_session']:.1f}
"""

                    return output

                else:
                    # Just show top tags
                    stats = tag_service.get_stats()
                    top_tags = stats.get("top_tags", [])

                    if not top_tags:
                        return """No tags found in the system yet.

Tags help organize and categorize your sessions. Use add-session-tag to start tagging sessions."""

                    output = "Top Tags:\n\n"
                    for tag_info in top_tags:
                        tag_name = tag_info["tag"]
                        count = tag_info["count"]
                        output += f"  {tag_name}: {count} session(s)\n"

                    output += f"\nTotal: {stats['unique_tags']} unique tags across {stats['tagged_sessions']} sessions"
                    output += "\n\nUse list-session-tags with show_all=true to see all tags."

                    return output

        except Exception as e:
            logger.error(f"Error listing tags: {e}", exc_info=True)
            return f"Error: Failed to list tags: {str(e)}"

    return list_tags_handler


def create_similar_sessions_handler(
    duplicate_service: Optional[DuplicateDetectionService]
):
    """
    Create the get-similar-sessions handler.

    Args:
        duplicate_service: DuplicateDetectionService instance for finding similar sessions
    """
    def similar_sessions_handler(arguments: Dict[str, Any]) -> str:
        """Handler for get-similar-sessions tool."""
        session_id = arguments.get("session_id", "")
        top_k = arguments.get("top_k", 5)
        include_scores = arguments.get("include_scores", True)

        if not session_id:
            return "Error: Please provide a session_id."

        if duplicate_service is None:
            return "Error: Duplicate detection service is not initialized."

        try:
            # Get similar sessions
            similar_sessions = duplicate_service.get_similar_sessions(
                session_id=session_id,
                top_k=top_k,
                include_metadata=True
            )

            if not similar_sessions:
                return f"""Similar Sessions for {session_id}

No similar sessions found above the similarity threshold ({duplicate_service.similarity_threshold}).

This session appears to be unique in your database. This is normal if:
- The session covers a unique topic or problem
- You have relatively few sessions indexed
- The session has very few chunks (< {duplicate_service.min_chunks_for_comparison})

💡 Tip: Use fork-detect to search for sessions by topic instead."""

            # Format the output
            output = f"""Similar Sessions for {session_id}

Found {len(similar_sessions)} similar session(s) (threshold: {duplicate_service.similarity_threshold}):

"""
            for i, similar in enumerate(similar_sessions, 1):
                # Format similarity score
                similarity_pct = f"{similar.similarity * 100:.1f}%"

                # Get metadata info
                metadata = similar.metadata or {}
                created_at = metadata.get('created_at', 'Unknown')
                num_messages = metadata.get('message_count', '?')

                # Format created date
                if created_at != 'Unknown':
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        date_str = dt.strftime("%Y-%m-%d")
                    except:
                        date_str = created_at
                else:
                    date_str = "Unknown"

                output += f"{i}. {similar.session_id}"

                if include_scores:
                    output += f"\n   Similarity: {similarity_pct}"

                output += f"\n   Date: {date_str}"
                output += f"\n   Messages: {num_messages}\n"

            output += """
---
These sessions may be:
- Duplicates (very high similarity)
- Related work on the same topic
- Sessions that could be merged or archived

Use get-session-preview to compare their content.
"""

            return output

        except Exception as e:
            logger.error(f"Error getting similar sessions: {e}", exc_info=True)
            return f"Error: Failed to get similar sessions: {str(e)}"

    return similar_sessions_handler


def create_cluster_sessions_handler(
    clustering_service: Optional[SessionClusteringService]
):
    """
    Create the cluster-sessions handler.

    Args:
        clustering_service: SessionClusteringService instance for clustering sessions
    """
    def cluster_sessions_handler(arguments: Dict[str, Any]) -> str:
        """Handler for cluster-sessions tool."""
        num_clusters = arguments.get("num_clusters")

        if clustering_service is None:
            return "Error: Session clustering service is not initialized."

        try:
            # Run clustering
            result = clustering_service.cluster_sessions(num_clusters=num_clusters)

            if result.num_clusters == 0:
                return """Session Clustering Complete

No sessions were eligible for clustering.

This happens when:
- No sessions have enough chunks (minimum 3 required)
- The database is empty or very sparse
- Sessions are too short to generate meaningful embeddings

💡 Tip: Index more sessions to enable clustering."""

            # Format output
            output = f"""Session Clustering Complete

Clustered {result.total_sessions} sessions into {result.num_clusters} topic groups
"""

            if result.overall_silhouette_score is not None:
                quality_desc = "excellent" if result.overall_silhouette_score > 0.5 else "good" if result.overall_silhouette_score > 0.25 else "fair"
                output += f"Clustering Quality: {result.overall_silhouette_score:.3f} ({quality_desc})\n"

            output += "\n"

            # Sort clusters by size (largest first)
            sorted_clusters = sorted(result.clusters, key=lambda c: c.size, reverse=True)

            # Show cluster summaries
            for cluster in sorted_clusters[:10]:  # Show top 10
                output += f"\nCluster {cluster.cluster_id}: {cluster.label}\n"
                output += f"  Sessions: {cluster.size}\n"
                if cluster.silhouette_score:
                    output += f"  Quality: {cluster.silhouette_score:.3f}\n"

            if len(result.clusters) > 10:
                output += f"\n... and {len(result.clusters) - 10} more clusters\n"

            output += """
---
Use get-session-clusters to browse clusters and their sessions.
Use get-cluster-sessions <cluster_id> to see sessions in a specific cluster.
"""

            return output

        except Exception as e:
            logger.error(f"Error clustering sessions: {e}", exc_info=True)
            return f"Error: Failed to cluster sessions: {str(e)}"

    return cluster_sessions_handler


def create_get_clusters_handler(
    clustering_service: Optional[SessionClusteringService]
):
    """
    Create the get-session-clusters handler.

    Args:
        clustering_service: SessionClusteringService instance for retrieving clusters
    """
    def get_clusters_handler(arguments: Dict[str, Any]) -> str:
        """Handler for get-session-clusters tool."""
        if clustering_service is None:
            return "Error: Session clustering service is not initialized."

        try:
            result = clustering_service.get_all_clusters()

            if result is None:
                return """No Clustering Available

Clusters have not been computed yet.

Run cluster-sessions to generate topic-based clusters from your sessions.

Example:
  cluster-sessions num_clusters=10
"""

            # Format output
            output = f"""Session Clusters

Total: {result.num_clusters} clusters, {result.total_sessions} sessions
"""

            if result.overall_silhouette_score:
                output += f"Overall Quality: {result.overall_silhouette_score:.3f}\n"

            output += "\n"

            # Sort clusters by size
            sorted_clusters = sorted(result.clusters, key=lambda c: c.size, reverse=True)

            for cluster in sorted_clusters:
                output += f"\nCluster {cluster.cluster_id}: {cluster.label}\n"
                output += f"  Sessions: {cluster.size}\n"

                # Show first few session IDs
                preview_count = min(3, len(cluster.session_ids))
                for sid in cluster.session_ids[:preview_count]:
                    output += f"    - {sid}\n"

                if len(cluster.session_ids) > preview_count:
                    output += f"    ... and {len(cluster.session_ids) - preview_count} more\n"

            output += """
---
Use get-cluster-sessions <cluster_id> to see all sessions in a cluster.
Use fork-detect to search for sessions by topic.
"""

            return output

        except Exception as e:
            logger.error(f"Error getting clusters: {e}", exc_info=True)
            return f"Error: Failed to get clusters: {str(e)}"

    return get_clusters_handler


def create_get_cluster_sessions_handler(
    clustering_service: Optional[SessionClusteringService]
):
    """
    Create the get-cluster-sessions handler.

    Args:
        clustering_service: SessionClusteringService instance for retrieving cluster sessions
    """
    def get_cluster_sessions_handler(arguments: Dict[str, Any]) -> str:
        """Handler for get-cluster-sessions tool."""
        cluster_id = arguments.get("cluster_id")

        if cluster_id is None:
            return "Error: Please provide a cluster_id."

        if clustering_service is None:
            return "Error: Session clustering service is not initialized."

        try:
            cluster = clustering_service.get_cluster_by_id(cluster_id)

            if cluster is None:
                return f"""Cluster Not Found

No cluster with ID {cluster_id} exists.

Use get-session-clusters to see available clusters.
"""

            # Format output
            output = f"""Cluster {cluster.cluster_id}: {cluster.label}

Sessions in this cluster ({cluster.size}):

"""

            for i, session_id in enumerate(cluster.session_ids, 1):
                output += f"{i}. {session_id}\n"

            output += """
---
Use get-session-preview <session_id> to view session content.
Use fork-detect to search for similar sessions.
"""

            return output

        except Exception as e:
            logger.error(f"Error getting cluster sessions: {e}", exc_info=True)
            return f"Error: Failed to get cluster sessions: {str(e)}"

    return get_cluster_sessions_handler


def create_session_summary_handler(
    session_registry: Optional[SessionRegistry]
):
    """
    Create the get-session-summary handler.

    Args:
        session_registry: SessionRegistry instance for accessing session summaries
    """
    def session_summary_handler(arguments: Dict[str, Any]) -> str:
        """Handler for get-session-summary tool."""
        session_id = arguments.get("session_id", "")

        if not session_id:
            return "Error: Please provide a session_id."

        if session_registry is None:
            return "Error: Session registry is not initialized."

        try:
            # Get session metadata
            session = session_registry.get_session(session_id)

            if session is None:
                return f"""Session Not Found

No session with ID '{session_id}' exists in the database.

Use fork-detect to search for sessions, or check that the session ID is correct.
"""

            # Check if summary exists
            if not session.summary:
                return f"""Session Summary: {session_id}

No summary available for this session.

This can happen if:
- The session was indexed before summarization was enabled
- The session is very short (< 20 characters of content)
- Summary generation failed during indexing

Session Info:
- Messages: {session.message_count}
- Chunks: {session.chunk_count}
- Project: {session.project or 'Unknown'}

Use get-session-preview to view the full session content instead.
"""

            # Format the output
            output = f"""Session Summary: {session_id}

Summary:
{session.summary}

Session Info:
- Messages: {session.message_count}
- Chunks: {session.chunk_count}
- Project: {session.project or 'Unknown'}
"""

            if session.created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(session.created_at.replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                    output += f"- Created: {date_str}\n"
                except:
                    pass

            if session.tags:
                tags_str = ", ".join(session.tags)
                output += f"- Tags: {tags_str}\n"

            output += """
---
Use get-session-preview for full content, or fork-detect to search for related sessions.
"""

            return output

        except Exception as e:
            logger.error(f"Error getting session summary: {e}", exc_info=True)
            return f"Error: Failed to get session summary: {str(e)}"

    return session_summary_handler


def create_compare_sessions_handler(
    diff_service: Optional[SessionDiffService]
):
    """
    Create the compare-sessions handler.

    Args:
        diff_service: SessionDiffService instance for comparing sessions
    """
    def compare_sessions_handler(arguments: Dict[str, Any]) -> str:
        """Handler for compare-sessions tool."""
        session_id_1 = arguments.get("session_id_1", "")
        session_id_2 = arguments.get("session_id_2", "")
        include_content = arguments.get("include_content", False)

        if not session_id_1:
            return "Error: Please provide session_id_1."

        if not session_id_2:
            return "Error: Please provide session_id_2."

        if diff_service is None:
            return "Error: Session diff service is not initialized."

        try:
            # Compare the sessions
            diff = diff_service.compare_sessions(session_id_1, session_id_2)

            if diff is None:
                return f"""Session Comparison Failed

Could not compare sessions '{session_id_1}' and '{session_id_2}'.

Possible reasons:
- One or both sessions do not exist
- One or both sessions have no content/chunks
- Embeddings could not be retrieved

Use get-session-preview to verify the sessions exist and have content.
"""

            # Format the output
            similarity_pct = f"{diff.similarity_score * 100:.1f}%"

            output = f"""Session Comparison: {session_id_1} vs {session_id_2}

Overall Similarity: {similarity_pct}
"""

            # Common messages section
            if diff.common_messages:
                output += f"\nCommon Content ({len(diff.common_messages)} matching messages):\n"

                # Show top 5 matches
                for i, match in enumerate(diff.common_messages[:5], 1):
                    similarity = f"{match.similarity * 100:.0f}%"
                    output += f"  {i}. Match (similarity: {similarity})\n"

                    if include_content:
                        # Show truncated content
                        content_1_preview = match.content_1[:80] + "..." if len(match.content_1) > 80 else match.content_1
                        content_2_preview = match.content_2[:80] + "..." if len(match.content_2) > 80 else match.content_2
                        output += f"     Session 1: {content_1_preview}\n"
                        output += f"     Session 2: {content_2_preview}\n"

                if len(diff.common_messages) > 5:
                    output += f"  ... and {len(diff.common_messages) - 5} more matches\n"
            else:
                output += "\nCommon Content: None\n"

            # Unique content section
            output += f"\nUnique to {session_id_1}: {len(diff.unique_to_1)} messages\n"
            if include_content and diff.unique_to_1:
                unique_content_1 = diff_service.get_message_content(
                    session_id_1,
                    diff.unique_to_1[:3],  # Show first 3
                    max_length=100
                )
                for i, content in enumerate(unique_content_1, 1):
                    output += f"  {i}. {content}\n"
                if len(diff.unique_to_1) > 3:
                    output += f"  ... and {len(diff.unique_to_1) - 3} more\n"

            output += f"\nUnique to {session_id_2}: {len(diff.unique_to_2)} messages\n"
            if include_content and diff.unique_to_2:
                unique_content_2 = diff_service.get_message_content(
                    session_id_2,
                    diff.unique_to_2[:3],  # Show first 3
                    max_length=100
                )
                for i, content in enumerate(unique_content_2, 1):
                    output += f"  {i}. {content}\n"
                if len(diff.unique_to_2) > 3:
                    output += f"  ... and {len(diff.unique_to_2) - 3} more\n"

            # Topics/Technologies section
            if diff.common_topics:
                topics_str = ", ".join(diff.common_topics[:10])
                output += f"\nCommon Topics: {topics_str}\n"
                if len(diff.common_topics) > 10:
                    output += f"  ... and {len(diff.common_topics) - 10} more\n"
            else:
                output += "\nCommon Topics: None\n"

            if diff.topics_1:
                topics_str = ", ".join(diff.topics_1[:10])
                output += f"\nUnique to {session_id_1}: {topics_str}\n"
                if len(diff.topics_1) > 10:
                    output += f"  ... and {len(diff.topics_1) - 10} more\n"

            if diff.topics_2:
                topics_str = ", ".join(diff.topics_2[:10])
                output += f"\nUnique to {session_id_2}: {topics_str}\n"
                if len(diff.topics_2) > 10:
                    output += f"  ... and {len(diff.topics_2) - 10} more\n"

            output += """
---
This comparison helps you understand:
- How similar two sessions are (duplicate detection)
- What unique work was done in each session
- Which technologies/topics differ between sessions

Use get-session-preview to view full session content.
Use include_content=true to see message snippets in the comparison.
"""

            return output

        except Exception as e:
            logger.error(f"Error comparing sessions: {e}", exc_info=True)
            return f"Error: Failed to compare sessions: {str(e)}"

    return compare_sessions_handler


def create_server(
    search_service: Optional[SearchService] = None,
    background_indexer: Optional[BackgroundIndexer] = None,
    claude_dir: Optional[str] = None,
    session_registry: Optional[Any] = None,
    fork_history_service: Optional[ForkHistoryService] = None,
    preference_service: Optional[PreferenceService] = None,
    tag_service: Optional[SessionTagService] = None,
    duplicate_service: Optional[DuplicateDetectionService] = None,
    clustering_service: Optional[SessionClusteringService] = None,
    diff_service: Optional[SessionDiffService] = None
) -> MCPServer:
    """
    Create and configure the MCP server.

    Args:
        search_service: Optional SearchService instance
        background_indexer: Optional BackgroundIndexer instance
        claude_dir: Optional path to Claude directory (for ForkGenerator)
        session_registry: Optional SessionRegistry for database stats
        fork_history_service: Optional ForkHistoryService for tracking fork history
        preference_service: Optional PreferenceService for learning from selections
        tag_service: Optional SessionTagService for managing session tags
        duplicate_service: Optional DuplicateDetectionService for finding similar sessions
        clustering_service: Optional SessionClusteringService for automatic topic clustering
        diff_service: Optional SessionDiffService for comparing sessions
    """
    server = MCPServer(
        search_service=search_service,
        background_indexer=background_indexer
    )

    # Register /fork-detect tool
    server.register_tool(
        name="fork-detect",
        description="Search for relevant past Claude Code sessions to fork from",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you want to do"
                },
                "project": {
                    "type": "string",
                    "description": "Optional project name to filter results. Use 'current' to auto-detect from working directory, or specify a project name explicitly."
                },
                "scope": {
                    "type": "string",
                    "enum": ["all", "project"],
                    "description": "Search scope: 'all' searches all sessions (default), 'project' searches only current project"
                },
                "time_range": {
                    "type": "string",
                    "description": "Predefined time range (today, yesterday, this_week, last_week, this_month, last_month, this_year) or natural language ('last Tuesday', '2 weeks ago', '3d')"
                },
                "start_date": {
                    "type": "string",
                    "description": "Custom start date for filtering (ISO format: 2026-01-01 or natural language)"
                },
                "end_date": {
                    "type": "string",
                    "description": "Custom end date for filtering (ISO format: 2026-01-21 or natural language)"
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated list of tags to filter results (e.g., 'bug-fix,urgent' or 'react'). Sessions must have at least one matching tag."
                },
                "include_archive": {
                    "type": "boolean",
                    "description": "Whether to include archived sessions in search results (default: false)",
                    "default": false
                }
            },
            "required": ["query"]
        },
        handler=create_fork_detect_handler(
            search_service,
            claude_dir=claude_dir,
            session_registry=session_registry
        )
    )

    # Register get-session-preview tool
    server.register_tool(
        name="get-session-preview",
        description="Get a preview of a session's content before forking",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to preview"
                },
                "length": {
                    "type": "integer",
                    "description": "Maximum preview length in characters (default: 500)",
                    "default": 500
                }
            },
            "required": ["session_id"]
        },
        handler=create_session_preview_handler(search_service, claude_dir=claude_dir)
    )

    # Register record-fork tool (for tracking fork events)
    server.register_tool(
        name="record-fork",
        description="Record a fork event for history tracking (internal tool, usually auto-called)",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID that was forked"
                },
                "query": {
                    "type": "string",
                    "description": "The query that led to this fork"
                },
                "position": {
                    "type": "integer",
                    "description": "Result position (1-5 for displayed results, -1 for custom)",
                    "default": -1
                }
            },
            "required": ["session_id", "query"]
        },
        handler=create_record_fork_handler(fork_history_service, preference_service)
    )

    # Register get-fork-history tool
    server.register_tool(
        name="get-fork-history",
        description="Get history of recently forked sessions",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of history entries to return (default: 10)",
                    "default": 10
                }
            }
        },
        handler=create_fork_history_handler(fork_history_service)
    )

    # Register add-session-tag tool
    server.register_tool(
        name="add-session-tag",
        description="Add a tag to a session for organization and categorization",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to tag"
                },
                "tag": {
                    "type": "string",
                    "description": "Tag name (will be normalized to lowercase)"
                }
            },
            "required": ["session_id", "tag"]
        },
        handler=create_add_tag_handler(tag_service)
    )

    # Register remove-session-tag tool
    server.register_tool(
        name="remove-session-tag",
        description="Remove a tag from a session",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to remove tag from"
                },
                "tag": {
                    "type": "string",
                    "description": "Tag name to remove (case-insensitive)"
                }
            },
            "required": ["session_id", "tag"]
        },
        handler=create_remove_tag_handler(tag_service)
    )

    # Register list-session-tags tool
    server.register_tool(
        name="list-session-tags",
        description="List tags for a session or all tags in the system",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID to list tags for. If not provided, lists all tags."
                },
                "show_all": {
                    "type": "boolean",
                    "description": "If true, shows all tags with counts. If false, shows top tags only. (default: false)",
                    "default": False
                }
            }
        },
        handler=create_list_tags_handler(tag_service)
    )

    # Register get-similar-sessions tool
    server.register_tool(
        name="get-similar-sessions",
        description="Find sessions similar to a given session (for detecting duplicates or related work)",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to find similar sessions for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of similar sessions to return (default: 5)",
                    "default": 5
                },
                "include_scores": {
                    "type": "boolean",
                    "description": "Whether to include similarity scores in output (default: true)",
                    "default": True
                }
            },
            "required": ["session_id"]
        },
        handler=create_similar_sessions_handler(duplicate_service)
    )

    # Register get-session-summary tool
    server.register_tool(
        name="get-session-summary",
        description="Get a quick summary of a session's content without reading the full session",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to get summary for"
                }
            },
            "required": ["session_id"]
        },
        handler=create_session_summary_handler(session_registry)
    )

    # Register compare-sessions tool
    server.register_tool(
        name="compare-sessions",
        description="Compare two sessions to identify common content, unique messages, and differences in topics/technologies",
        input_schema={
            "type": "object",
            "properties": {
                "session_id_1": {
                    "type": "string",
                    "description": "First session ID to compare"
                },
                "session_id_2": {
                    "type": "string",
                    "description": "Second session ID to compare"
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Whether to include message content snippets in the output (default: false)",
                    "default": False
                }
            },
            "required": ["session_id_1", "session_id_2"]
        },
        handler=create_compare_sessions_handler(diff_service)
    )

    # Register cluster-sessions tool
    server.register_tool(
        name="cluster-sessions",
        description="Automatically cluster sessions by topic using k-means on session embeddings",
        input_schema={
            "type": "object",
            "properties": {
                "num_clusters": {
                    "type": "integer",
                    "description": "Number of clusters to create (default: 10, auto-adjusted based on available sessions)"
                }
            }
        },
        handler=create_cluster_sessions_handler(clustering_service)
    )

    # Register get-session-clusters tool
    server.register_tool(
        name="get-session-clusters",
        description="Get all session clusters and their metadata",
        input_schema={
            "type": "object",
            "properties": {}
        },
        handler=create_get_clusters_handler(clustering_service)
    )

    # Register get-cluster-sessions tool
    server.register_tool(
        name="get-cluster-sessions",
        description="Get all sessions in a specific cluster",
        input_schema={
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "integer",
                    "description": "Cluster ID to retrieve sessions for"
                }
            },
            "required": ["cluster_id"]
        },
        handler=create_get_cluster_sessions_handler(clustering_service)
    )

    return server


def main() -> None:
    """Main entry point for the Smart Fork MCP server."""
    # Initialize services (may be None if initialization fails)
    search_service, background_indexer = initialize_services()

    if search_service is None:
        logger.warning("Services not initialized - server will run with limited functionality")

    # Get Claude directory path
    claude_dir = str(Path.home() / ".claude")

    # Initialize fork history and preference services
    fork_history_service = ForkHistoryService()
    logger.info("Fork history service initialized")

    preference_service = PreferenceService()
    logger.info("Preference service initialized")

    # Start background indexer if initialized
    if background_indexer is not None:
        background_indexer.start()
        logger.info("Background indexer started")

        # Register cleanup handlers
        def cleanup():
            if background_indexer is not None and background_indexer.is_running():
                logger.info("Stopping background indexer...")
                background_indexer.stop()

        atexit.register(cleanup)

        # Handle SIGTERM and SIGINT gracefully
        def signal_handler(signum, frame):
            cleanup()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    # Get session_registry from search_service if available
    session_registry = None
    if search_service is not None:
        session_registry = getattr(search_service, 'session_registry', None)

    # Initialize session tag service
    tag_service = None
    if session_registry is not None:
        tag_service = SessionTagService(session_registry)
        logger.info("Session tag service initialized")

    # Initialize duplicate detection service
    duplicate_service = None
    if search_service is not None:
        vector_db_service = getattr(search_service, 'vector_db_service', None)
        if vector_db_service is not None and session_registry is not None:
            duplicate_service = DuplicateDetectionService(
                vector_db_service=vector_db_service,
                session_registry=session_registry
            )
            logger.info("Duplicate detection service initialized")

    # Initialize session clustering service
    clustering_service = None
    if search_service is not None:
        vector_db_service = getattr(search_service, 'vector_db_service', None)
        if vector_db_service is not None and session_registry is not None:
            clustering_service = SessionClusteringService(
                vector_db_service=vector_db_service,
                session_registry=session_registry
            )
            logger.info("Session clustering service initialized")

    # Initialize session diff service
    diff_service = None
    if search_service is not None:
        vector_db_service = getattr(search_service, 'vector_db_service', None)
        embedding_service = getattr(search_service, 'embedding_service', None)
        if vector_db_service is not None and session_registry is not None and embedding_service is not None:
            diff_service = SessionDiffService(
                vector_db_service=vector_db_service,
                session_registry=session_registry,
                embedding_service=embedding_service
            )
            logger.info("Session diff service initialized")

    # Create and run server
    server = create_server(
        search_service=search_service,
        background_indexer=background_indexer,
        claude_dir=claude_dir,
        session_registry=session_registry,
        fork_history_service=fork_history_service,
        preference_service=preference_service,
        tag_service=tag_service,
        duplicate_service=duplicate_service,
        clustering_service=clustering_service,
        diff_service=diff_service
    )
    server.run()


# Aliases for backwards compatibility with tests
format_search_results = format_search_results_with_selection
fork_detect_handler = create_fork_detect_handler(None)


if __name__ == "__main__":
    main()
