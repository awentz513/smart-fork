"""
Unit tests for SelectionUI module.
"""

import pytest
from datetime import datetime
from dataclasses import dataclass

from smart_fork.selection_ui import SelectionUI, SelectionOption
from smart_fork.search_service import SessionSearchResult
from smart_fork.scoring_service import SessionScore
from smart_fork.session_registry import SessionMetadata
from smart_fork.fork_generator import ForkGenerator


@dataclass
class MockSessionMetadata:
    """Mock session metadata for testing."""
    project: str
    created_at: str
    message_count: int
    chunk_count: int
    tags: list


def create_mock_result(
    session_id: str,
    score: float,
    project: str = "test-project",
    preview: str = "Test preview content"
) -> SessionSearchResult:
    """Create a mock search result for testing."""
    session_score = SessionScore(
        session_id=session_id,
        final_score=score,
        best_similarity=score * 0.4,
        avg_similarity=score * 0.2,
        chunk_ratio=score * 0.05,
        recency_score=score * 0.25,
        chain_quality=0.5,
        memory_boost=0.0,
        num_chunks_matched=5
    )

    metadata = MockSessionMetadata(
        project=project,
        created_at="2026-01-20T15:30:00Z",
        message_count=50,
        chunk_count=10,
        tags=["tag1", "tag2"]
    )

    return SessionSearchResult(
        session_id=session_id,
        score=session_score,
        metadata=metadata,
        preview=preview,
        matched_chunks=[]
    )


class TestSelectionUI:
    """Test SelectionUI initialization."""

    def test_init(self):
        """Test SelectionUI initialization."""
        ui = SelectionUI()
        assert ui is not None


class TestDateFormatting:
    """Test date formatting."""

    def test_format_date_iso(self):
        """Test formatting ISO date string."""
        ui = SelectionUI()
        date_str = "2026-01-20T15:30:00Z"
        formatted = ui.format_date(date_str)
        assert "2026-01-20" in formatted
        assert "15:30" in formatted

    def test_format_date_invalid(self):
        """Test formatting invalid date string."""
        ui = SelectionUI()
        date_str = "invalid-date"
        formatted = ui.format_date(date_str)
        assert formatted == "invalid-date"


class TestPreviewTruncation:
    """Test preview truncation."""

    def test_truncate_short_preview(self):
        """Test truncating a short preview (no truncation needed)."""
        ui = SelectionUI()
        preview = "Short preview"
        truncated = ui.truncate_preview(preview, max_length=150)
        assert truncated == "Short preview"

    def test_truncate_long_preview(self):
        """Test truncating a long preview."""
        ui = SelectionUI()
        preview = "A" * 200
        truncated = ui.truncate_preview(preview, max_length=150)
        assert len(truncated) <= 153  # 150 + "..."
        assert truncated.endswith("...")

    def test_truncate_at_word_boundary(self):
        """Test that truncation happens at word boundary."""
        ui = SelectionUI()
        preview = "word1 word2 word3 " + "A" * 200
        truncated = ui.truncate_preview(preview, max_length=20)
        assert truncated.endswith("...")
        # Should truncate at word boundary
        assert not truncated[:-3].endswith(" ")


class TestCreateOptions:
    """Test option creation."""

    def test_create_options_with_results(self):
        """Test creating options with search results."""
        ui = SelectionUI()
        results = [
            create_mock_result("session1", 0.9),
            create_mock_result("session2", 0.8),
            create_mock_result("session3", 0.7),
        ]
        query = "test query"

        options = ui.create_options(results, query)

        # Should have exactly 5 options
        assert len(options) == 5

        # First 3 should be results
        assert options[0].session_id == "session1"
        assert options[1].session_id == "session2"
        assert options[2].session_id == "session3"

        # First should be marked as recommended
        assert options[0].is_recommended is True
        assert options[1].is_recommended is False
        assert options[2].is_recommended is False

        # Last 2 should be none and refine
        assert options[3].id == "none"
        assert options[4].id == "refine"

    def test_create_options_with_no_results(self):
        """Test creating options with no search results."""
        ui = SelectionUI()
        results = []
        query = "test query"

        options = ui.create_options(results, query)

        # Should still have exactly 5 options
        assert len(options) == 5

        # Should have none and refine options
        none_found = any(opt.id == "none" for opt in options)
        refine_found = any(opt.id == "refine" for opt in options)
        assert none_found
        assert refine_found

    def test_create_options_with_one_result(self):
        """Test creating options with only one search result."""
        ui = SelectionUI()
        results = [
            create_mock_result("session1", 0.9),
        ]
        query = "test query"

        options = ui.create_options(results, query)

        # Should have exactly 5 options
        assert len(options) == 5

        # First should be the result
        assert options[0].session_id == "session1"
        assert options[0].is_recommended is True

        # Should have none and refine options
        none_found = any(opt.id == "none" for opt in options)
        refine_found = any(opt.id == "refine" for opt in options)
        assert none_found
        assert refine_found

    def test_create_options_labels(self):
        """Test that option labels are formatted correctly."""
        ui = SelectionUI()
        results = [
            create_mock_result("session1", 0.9),
            create_mock_result("session2", 0.8),
        ]
        query = "test query"

        options = ui.create_options(results, query)

        # First option should have recommended marker
        assert "RECOMMENDED" in options[0].label
        assert "⭐" in options[0].label

        # Second option should not
        assert "RECOMMENDED" not in options[1].label

        # None option should have cross marker
        none_option = [opt for opt in options if opt.id == "none"][0]
        assert "❌" in none_option.label

        # Refine option should have search marker
        refine_option = [opt for opt in options if opt.id == "refine"][0]
        assert "🔍" in refine_option.label


class TestFormatSelectionPrompt:
    """Test selection prompt formatting."""

    def test_format_selection_prompt_with_results(self):
        """Test formatting selection prompt with results."""
        ui = SelectionUI()
        results = [
            create_mock_result("session1", 0.9),
            create_mock_result("session2", 0.8),
        ]
        query = "test query"

        options = ui.create_options(results, query)
        prompt = ui.format_selection_prompt(options, query)

        # Check for key elements
        assert "Fork Detection - Select a Session" in prompt
        assert query in prompt
        assert "session1" in prompt
        assert "session2" in prompt
        assert "None of these - start fresh" in prompt
        assert "Type something else" in prompt
        assert "Keyboard shortcuts" in prompt

    def test_format_selection_prompt_no_results(self):
        """Test formatting selection prompt with no results."""
        ui = SelectionUI()
        results = []
        query = "test query"

        options = ui.create_options(results, query)
        prompt = ui.format_selection_prompt(options, query)

        # Check for key elements
        assert "Fork Detection - Select a Session" in prompt
        assert query in prompt
        assert "No matching sessions found" in prompt


class TestFormatChatOption:
    """Test chat option formatting."""

    def test_format_chat_option(self):
        """Test formatting chat option for a result."""
        ui = SelectionUI()
        result = create_mock_result("session1", 0.9)

        chat_prompt = ui.format_chat_option(result)

        # Check for key elements
        assert "Session Details" in chat_prompt
        assert "session1" in chat_prompt
        assert "Project: test-project" in chat_prompt
        assert "Score Breakdown" in chat_prompt
        assert "Preview:" in chat_prompt
        assert result.preview in chat_prompt


class TestDisplaySelection:
    """Test display selection."""

    def test_display_selection_with_results(self):
        """Test display selection with results."""
        ui = SelectionUI()
        results = [
            create_mock_result("session1", 0.9),
            create_mock_result("session2", 0.8),
        ]
        query = "test query"

        display_data = ui.display_selection(results, query)

        # Check structure
        assert 'prompt' in display_data
        assert 'options' in display_data
        assert 'query' in display_data
        assert 'num_results' in display_data

        # Check data
        assert display_data['query'] == query
        assert display_data['num_results'] == 2
        assert len(display_data['options']) == 5

        # Check options structure
        for option in display_data['options']:
            assert 'id' in option
            assert 'label' in option
            assert 'description' in option

    def test_display_selection_no_results(self):
        """Test display selection with no results."""
        ui = SelectionUI()
        results = []
        query = "test query"

        display_data = ui.display_selection(results, query)

        # Check structure
        assert 'prompt' in display_data
        assert 'options' in display_data
        assert 'num_results' in display_data
        assert display_data['num_results'] == 0


class TestHandleSelection:
    """Test selection handling."""

    def test_handle_selection_session(self):
        """Test handling selection of a session."""
        ui = SelectionUI()
        results = [
            create_mock_result("session1", 0.9),
        ]
        query = "test query"

        options = ui.create_options(results, query)
        result = ui.handle_selection("result_0", options)

        assert result['status'] == 'selected'
        assert result['action'] == 'fork'
        assert result['session_id'] == 'session1'

    def test_handle_selection_none(self):
        """Test handling selection of 'none'."""
        ui = SelectionUI()
        results = []
        query = "test query"

        options = ui.create_options(results, query)
        result = ui.handle_selection("none", options)

        assert result['status'] == 'selected'
        assert result['action'] == 'start_fresh'

    def test_handle_selection_refine(self):
        """Test handling selection of 'refine'."""
        ui = SelectionUI()
        results = []
        query = "test query"

        options = ui.create_options(results, query)
        result = ui.handle_selection("refine", options)

        assert result['status'] == 'selected'
        assert result['action'] == 'refine'

    def test_handle_selection_invalid(self):
        """Test handling invalid selection."""
        ui = SelectionUI()
        results = []
        query = "test query"

        options = ui.create_options(results, query)
        result = ui.handle_selection("invalid_id", options)

        assert result['status'] == 'error'

    def test_handle_selection_empty_slot(self):
        """Test handling selection of empty slot."""
        ui = SelectionUI()
        results = [
            create_mock_result("session1", 0.9),
        ]
        query = "test query"

        options = ui.create_options(results, query)
        # Try to select an empty slot
        empty_id = [opt.id for opt in options if opt.id.startswith("empty")][0]
        result = ui.handle_selection(empty_id, options)

        assert result['status'] == 'error'


class TestOptionDataclass:
    """Test SelectionOption dataclass."""

    def test_selection_option_creation(self):
        """Test creating a SelectionOption."""
        option = SelectionOption(
            id="test",
            label="Test Label",
            description="Test Description",
            session_id="session1",
            is_recommended=True,
            score=0.9,
            metadata={"key": "value"},
            preview="Test preview"
        )

        assert option.id == "test"
        assert option.label == "Test Label"
        assert option.description == "Test Description"
        assert option.session_id == "session1"
        assert option.is_recommended is True
        assert option.score == 0.9
        assert option.metadata == {"key": "value"}
        assert option.preview == "Test preview"

    def test_selection_option_defaults(self):
        """Test SelectionOption with default values."""
        option = SelectionOption(
            id="test",
            label="Test Label",
            description="Test Description"
        )

        assert option.session_id is None
        assert option.is_recommended is False
        assert option.score is None
        assert option.metadata is None
        assert option.preview is None
        assert option.fork_terminal_cmd is None
        assert option.fork_in_session_cmd is None


class TestSelectionUIWithForkGenerator:
    """Test SelectionUI with ForkGenerator integration."""

    def test_create_options_with_fork_generator(self):
        """Test creating options with fork commands when ForkGenerator is provided."""
        fork_generator = ForkGenerator(claude_sessions_dir="~/.claude")
        ui = SelectionUI(fork_generator=fork_generator)

        results = [
            create_mock_result("session1", 0.9),
            create_mock_result("session2", 0.8),
        ]
        query = "test query"

        options = ui.create_options(results, query)

        # Should have exactly 5 options
        assert len(options) == 5

        # First 2 should be results with fork commands
        assert options[0].session_id == "session1"
        assert options[0].fork_terminal_cmd is not None
        assert "claude --resume session1" in options[0].fork_terminal_cmd
        assert options[0].fork_in_session_cmd is not None
        assert "/fork session1" in options[0].fork_in_session_cmd

        assert options[1].session_id == "session2"
        assert options[1].fork_terminal_cmd is not None
        assert "claude --resume session2" in options[1].fork_terminal_cmd

        # None and refine options should not have fork commands
        none_option = [opt for opt in options if opt.id == "none"][0]
        assert none_option.fork_terminal_cmd is None

        refine_option = [opt for opt in options if opt.id == "refine"][0]
        assert refine_option.fork_terminal_cmd is None

    def test_create_options_without_fork_generator(self):
        """Test creating options without ForkGenerator (backwards compatibility)."""
        ui = SelectionUI()  # No fork_generator provided

        results = [
            create_mock_result("session1", 0.9),
        ]
        query = "test query"

        options = ui.create_options(results, query)

        # Should still work but without fork commands
        assert len(options) == 5
        assert options[0].session_id == "session1"
        assert options[0].fork_terminal_cmd is None
        assert options[0].fork_in_session_cmd is None

    def test_format_selection_prompt_with_fork_commands(self):
        """Test that fork commands are included in the formatted prompt."""
        fork_generator = ForkGenerator(claude_sessions_dir="~/.claude")
        ui = SelectionUI(fork_generator=fork_generator)

        results = [
            create_mock_result("session1", 0.9),
        ]
        query = "test query"

        options = ui.create_options(results, query)
        prompt = ui.format_selection_prompt(options, query)

        # Check that fork commands are in the prompt
        assert "Fork Commands (copy & paste):" in prompt
        assert "New terminal:" in prompt
        assert "In-session:" in prompt
        assert "claude --resume session1" in prompt
        assert "/fork session1" in prompt

    def test_display_selection_includes_fork_commands(self):
        """Test that display_selection includes fork commands in returned data."""
        fork_generator = ForkGenerator(claude_sessions_dir="~/.claude")
        ui = SelectionUI(fork_generator=fork_generator)

        results = [
            create_mock_result("session1", 0.9),
        ]
        query = "test query"

        display_data = ui.display_selection(results, query)

        # Check that fork commands are in the options data
        assert 'options' in display_data
        result_option = display_data['options'][0]
        assert 'fork_terminal_cmd' in result_option
        assert 'fork_in_session_cmd' in result_option
        assert result_option['fork_terminal_cmd'] is not None
        assert "claude --resume session1" in result_option['fork_terminal_cmd']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
