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
    session_registry: Optional[Any] = None
) -> str:
    """
    Format search results with interactive selection UI.

    Args:
        query: Search query
        results: List of search results
        claude_dir: Optional path to Claude directory (for ForkGenerator)
        session_registry: Optional SessionRegistry for database stats

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
        return f"""Fork Detection - No Results Found

Your query: {query}
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
    selection_data = selection_ui.display_selection(results, query)
    return selection_data['prompt']


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
            # Perform search with 3-second target
            logger.info(f"Processing fork-detect query: {query}")
            results = search_service.search(query, top_n=5)

            # Format and return results with selection UI (including fork commands)
            formatted_output = format_search_results_with_selection(
                query,
                results,
                claude_dir=claude_dir,
                session_registry=session_registry
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

        # Initialize services
        embedding_service = EmbeddingService()
        vector_db_service = VectorDBService(persist_directory=str(vector_db_path))
        scoring_service = ScoringService()
        session_registry = SessionRegistry(registry_path=str(registry_path))
        chunking_service = ChunkingService()
        session_parser = SessionParser()

        # Create search service
        search_service = SearchService(
            embedding_service=embedding_service,
            vector_db_service=vector_db_service,
            scoring_service=scoring_service,
            session_registry=session_registry,
            k_chunks=200,
            top_n_sessions=5,
            preview_length=200
        )

        # Get Claude directory to monitor
        claude_dir = Path.home() / ".claude"

        # Create background indexer
        background_indexer = BackgroundIndexer(
            claude_dir=claude_dir,
            vector_db=vector_db_service,
            session_registry=session_registry,
            embedding_service=embedding_service,
            chunking_service=chunking_service,
            session_parser=session_parser,
            debounce_seconds=5.0,
            checkpoint_interval=15
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


def create_server(
    search_service: Optional[SearchService] = None,
    background_indexer: Optional[BackgroundIndexer] = None,
    claude_dir: Optional[str] = None,
    session_registry: Optional[Any] = None
) -> MCPServer:
    """
    Create and configure the MCP server.

    Args:
        search_service: Optional SearchService instance
        background_indexer: Optional BackgroundIndexer instance
        claude_dir: Optional path to Claude directory (for ForkGenerator)
        session_registry: Optional SessionRegistry for database stats
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

    return server


def main() -> None:
    """Main entry point for the Smart Fork MCP server."""
    # Initialize services (may be None if initialization fails)
    search_service, background_indexer = initialize_services()

    if search_service is None:
        logger.warning("Services not initialized - server will run with limited functionality")

    # Get Claude directory path
    claude_dir = str(Path.home() / ".claude")

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

    # Create and run server
    server = create_server(
        search_service=search_service,
        background_indexer=background_indexer,
        claude_dir=claude_dir,
        session_registry=session_registry
    )
    server.run()


# Aliases for backwards compatibility with tests
format_search_results = format_search_results_with_selection
fork_detect_handler = create_fork_detect_handler(None)


if __name__ == "__main__":
    main()
