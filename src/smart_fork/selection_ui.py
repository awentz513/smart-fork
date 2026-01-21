"""
Interactive selection UI for choosing from search results.

This module provides functionality to display search results in a selectable format
and handle user selection with options for forking, refining search, or starting fresh.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .search_service import SessionSearchResult
from .fork_generator import ForkGenerator

logger = logging.getLogger(__name__)


@dataclass
class SelectionOption:
    """Represents a single selectable option."""
    id: str
    label: str
    description: str
    session_id: Optional[str] = None
    is_recommended: bool = False
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    preview: Optional[str] = None
    fork_terminal_cmd: Optional[str] = None
    fork_in_session_cmd: Optional[str] = None


class SelectionUI:
    """
    Interactive selection UI for search results.

    Displays exactly 5 options:
    - Top 3 search results (with highest marked as 'Recommended')
    - 'None - start fresh' option
    - 'Type something else' refinement option
    """

    def __init__(self, fork_generator: Optional[ForkGenerator] = None):
        """
        Initialize the SelectionUI.

        Args:
            fork_generator: Optional ForkGenerator instance for generating fork commands.
                          If not provided, fork commands will not be included in options.
        """
        self.fork_generator = fork_generator
        logger.info("Initialized SelectionUI")

    def format_date(self, date_str: str) -> str:
        """
        Format date string for display.

        Args:
            date_str: ISO format date string

        Returns:
            Formatted date string (e.g., "2026-01-20 15:30")
        """
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            return date_str

    def truncate_preview(self, preview: str, max_length: int = 150) -> str:
        """
        Truncate preview text to maximum length.

        Args:
            preview: Preview text
            max_length: Maximum length in characters

        Returns:
            Truncated preview with ellipsis if needed
        """
        if len(preview) <= max_length:
            return preview

        # Truncate at word boundary
        truncated = preview[:max_length].rsplit(' ', 1)[0]
        return truncated + "..."

    def create_options(
        self,
        search_results: List[SessionSearchResult],
        query: str
    ) -> List[SelectionOption]:
        """
        Create selection options from search results.

        Args:
            search_results: List of search results (may be empty)
            query: Original search query

        Returns:
            List of exactly 5 SelectionOption objects
        """
        options = []

        # Add top 3 results (or fewer if less than 3 results)
        num_results = min(len(search_results), 3)

        for idx in range(num_results):
            result = search_results[idx]
            is_recommended = (idx == 0)  # Mark first result as recommended

            # Extract metadata
            metadata_dict = None
            if result.metadata:
                metadata_dict = {
                    'project': result.metadata.project or 'Unknown',
                    'created_at': self.format_date(result.metadata.created_at),
                    'messages': result.metadata.message_count,
                    'chunks': result.metadata.chunk_count,
                    'tags': result.metadata.tags or []
                }

            # Create label with recommendation marker
            label_prefix = "⭐ [RECOMMENDED] " if is_recommended else f"   [{idx + 1}] "
            project = result.metadata.project if result.metadata else "Unknown"
            score_pct = int(result.score.final_score * 100)

            label = f"{label_prefix}Session: {result.session_id[:16]}... ({score_pct}%)"

            # Create description with details
            date_str = self.format_date(result.metadata.created_at) if result.metadata else "Unknown date"
            preview_text = self.truncate_preview(result.preview, max_length=150)

            description_lines = [
                f"Project: {project}",
                f"Date: {date_str}",
                f"Score: {score_pct}%",
                f"Preview: {preview_text}"
            ]

            if result.metadata and result.metadata.tags:
                description_lines.append(f"Tags: {', '.join(result.metadata.tags)}")

            description = "\n".join(description_lines)

            # Generate fork commands if fork_generator is available
            fork_terminal_cmd = None
            fork_in_session_cmd = None
            if self.fork_generator:
                try:
                    from .session_registry import SessionMetadata
                    # Convert metadata dict back to SessionMetadata for fork_generator
                    session_metadata = None
                    if result.metadata:
                        session_metadata = SessionMetadata(
                            session_id=result.session_id,
                            project=result.metadata.project,
                            created_at=result.metadata.created_at,
                            last_synced=result.metadata.created_at,
                            message_count=result.metadata.message_count,
                            chunk_count=result.metadata.chunk_count,
                            tags=result.metadata.tags or []
                        )
                    fork_cmd = self.fork_generator.generate_fork_command(
                        result.session_id,
                        metadata=session_metadata
                    )
                    fork_terminal_cmd = fork_cmd.terminal_command
                    fork_in_session_cmd = fork_cmd.in_session_command
                except Exception as e:
                    logger.warning(f"Failed to generate fork commands for {result.session_id}: {e}")

            option = SelectionOption(
                id=f"result_{idx}",
                label=label,
                description=description,
                session_id=result.session_id,
                is_recommended=is_recommended,
                score=result.score.final_score,
                metadata=metadata_dict,
                preview=result.preview,
                fork_terminal_cmd=fork_terminal_cmd,
                fork_in_session_cmd=fork_in_session_cmd
            )

            options.append(option)

        # Add 'None - start fresh' option
        none_option = SelectionOption(
            id="none",
            label="❌ None of these - start fresh",
            description="Don't fork from any session. Start a completely new conversation instead.",
            session_id=None,
            is_recommended=False
        )
        options.append(none_option)

        # Add 'Type something else' refinement option
        refine_option = SelectionOption(
            id="refine",
            label="🔍 Type something else",
            description=f"Refine your search with a different query.\nCurrent query: {query}",
            session_id=None,
            is_recommended=False
        )
        options.append(refine_option)

        # If we have fewer than 3 results, add explanation options
        while len(options) < 5:
            empty_slot = SelectionOption(
                id=f"empty_{len(options)}",
                label="   [No more results]",
                description="No additional matching sessions found.",
                session_id=None,
                is_recommended=False
            )
            options.append(empty_slot)

        return options

    def format_selection_prompt(
        self,
        options: List[SelectionOption],
        query: str
    ) -> str:
        """
        Format selection prompt for display.

        Args:
            options: List of selection options
            query: Original search query

        Returns:
            Formatted prompt string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Fork Detection - Select a Session")
        lines.append("=" * 80)
        lines.append(f"\nYour query: {query}\n")

        if not any(opt.session_id for opt in options):
            lines.append("No matching sessions found.")
            lines.append("\nOptions:")
        else:
            lines.append("Please select one of the following options:\n")

        for idx, option in enumerate(options, 1):
            lines.append(f"\n{idx}. {option.label}")

            # Format description with proper indentation
            desc_lines = option.description.split('\n')
            for desc_line in desc_lines:
                lines.append(f"   {desc_line}")

            if option.is_recommended:
                lines.append("   💡 This is the best match based on relevance scoring.")

            # Add fork commands if available
            if option.fork_terminal_cmd and option.fork_in_session_cmd:
                lines.append("")
                lines.append("   Fork Commands (copy & paste):")
                lines.append(f"   New terminal:  {option.fork_terminal_cmd}")
                lines.append(f"   In-session:    {option.fork_in_session_cmd}")

            lines.append("")

        lines.append("=" * 80)
        lines.append("\nKeyboard shortcuts:")
        lines.append("  • Enter: Select highlighted option")
        lines.append("  • ↑/↓: Navigate options")
        lines.append("  • Esc: Cancel")
        lines.append("\nTip: You can also chat about a session before forking by asking questions about it.")

        return "\n".join(lines)

    def format_chat_option(
        self,
        result: SessionSearchResult
    ) -> str:
        """
        Format the 'Chat about this' option for a specific result.

        Args:
            result: Search result to chat about

        Returns:
            Formatted chat prompt string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Session Details: {result.session_id}")
        lines.append("=" * 80)
        lines.append("")

        if result.metadata:
            lines.append(f"Project: {result.metadata.project or 'Unknown'}")
            lines.append(f"Created: {self.format_date(result.metadata.created_at)}")
            lines.append(f"Messages: {result.metadata.message_count}")
            lines.append(f"Chunks: {result.metadata.chunk_count}")
            if result.metadata.tags:
                lines.append(f"Tags: {', '.join(result.metadata.tags)}")
            lines.append("")

        lines.append(f"Relevance Score: {int(result.score.final_score * 100)}%")
        lines.append("")
        lines.append("Score Breakdown:")
        lines.append(f"  • Best Similarity: {result.score.best_similarity:.2%}")
        lines.append(f"  • Avg Similarity: {result.score.avg_similarity:.2%}")
        lines.append(f"  • Chunk Ratio: {result.score.chunk_ratio:.2%}")
        lines.append(f"  • Recency: {result.score.recency_score:.2%}")
        lines.append(f"  • Chain Quality: {result.score.chain_quality:.2%}")
        lines.append(f"  • Memory Boost: +{result.score.memory_boost:.2%}")
        lines.append("")
        lines.append("Preview:")
        lines.append("-" * 80)
        lines.append(result.preview)
        lines.append("-" * 80)
        lines.append("")
        lines.append("What would you like to know about this session?")

        return "\n".join(lines)

    def display_selection(
        self,
        search_results: List[SessionSearchResult],
        query: str
    ) -> Dict[str, Any]:
        """
        Display selection UI and return selection data.

        Args:
            search_results: List of search results
            query: Original search query

        Returns:
            Dictionary containing selection prompt and options
        """
        # Create options
        options = self.create_options(search_results, query)

        # Format prompt
        prompt = self.format_selection_prompt(options, query)

        # Return structured data
        return {
            'prompt': prompt,
            'options': [
                {
                    'id': opt.id,
                    'label': opt.label,
                    'description': opt.description,
                    'session_id': opt.session_id,
                    'is_recommended': opt.is_recommended,
                    'score': opt.score,
                    'metadata': opt.metadata,
                    'preview': opt.preview,
                    'fork_terminal_cmd': opt.fork_terminal_cmd,
                    'fork_in_session_cmd': opt.fork_in_session_cmd
                }
                for opt in options
            ],
            'query': query,
            'num_results': len([opt for opt in options if opt.session_id])
        }

    def handle_selection(
        self,
        selection_id: str,
        options: List[SelectionOption]
    ) -> Dict[str, Any]:
        """
        Handle user selection and return appropriate response.

        Args:
            selection_id: ID of selected option
            options: List of available options

        Returns:
            Dictionary with selection result and action
        """
        # Find selected option
        selected = None
        for option in options:
            if option.id == selection_id:
                selected = option
                break

        if selected is None:
            return {
                'status': 'error',
                'message': 'Invalid selection'
            }

        # Handle based on selection type
        if selected.session_id:
            # Session selected - return fork instructions
            return {
                'status': 'selected',
                'action': 'fork',
                'session_id': selected.session_id,
                'metadata': selected.metadata,
                'message': f"Selected session: {selected.session_id}"
            }
        elif selected.id == 'none':
            # Start fresh
            return {
                'status': 'selected',
                'action': 'start_fresh',
                'message': 'Starting fresh without forking'
            }
        elif selected.id == 'refine':
            # Refine search
            return {
                'status': 'selected',
                'action': 'refine',
                'message': 'Please provide a new search query'
            }
        else:
            # Empty slot or invalid
            return {
                'status': 'error',
                'message': 'Cannot select this option'
            }
