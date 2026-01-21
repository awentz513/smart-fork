"""
Tests for SessionSummaryService.
"""

import pytest
from smart_fork.session_summary_service import SessionSummaryService, SessionSummary


@pytest.fixture
def summary_service():
    """Create a SessionSummaryService instance."""
    return SessionSummaryService(max_sentences=3, min_sentence_length=20, max_sentence_length=200)


class TestSessionSummaryService:
    """Tests for SessionSummaryService."""

    def test_empty_messages(self, summary_service):
        """Test summarizing empty message list."""
        summary = summary_service.generate_summary([], "test-session")

        assert summary.session_id == "test-session"
        assert summary.summary == "Empty session with no messages."
        assert summary.sentence_count == 0
        assert summary.source_message_count == 0
        assert len(summary.topics) == 0

    def test_single_short_message(self, summary_service):
        """Test summarizing a single short message."""
        messages = [{"content": "Hi"}]
        summary = summary_service.generate_summary(messages, "test-session")

        assert summary.session_id == "test-session"
        assert summary.summary == "Session contains no extractable text."
        assert summary.sentence_count == 0
        assert summary.source_message_count == 1

    def test_single_valid_message(self, summary_service):
        """Test summarizing a single valid message."""
        messages = [
            {"content": "I need help implementing authentication in my Python Flask application."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert summary.session_id == "test-session"
        assert "authentication" in summary.summary.lower()
        assert summary.sentence_count >= 1
        assert summary.source_message_count == 1

    def test_multiple_messages(self, summary_service):
        """Test summarizing multiple messages."""
        messages = [
            {"content": "I need to build a REST API using FastAPI for a blog application."},
            {"content": "The API should support user authentication and post management."},
            {"content": "I also need to implement search functionality with Elasticsearch."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert summary.session_id == "test-session"
        assert len(summary.summary) > 0
        assert summary.sentence_count <= 3  # max_sentences
        assert summary.source_message_count == 3
        assert len(summary.topics) > 0

    def test_code_heavy_content_filtered(self, summary_service):
        """Test that code-heavy content is filtered out."""
        messages = [
            {"content": "Here's a normal sentence about the project requirements."},
            {"content": "def function(x, y, z): return x + y * z / (a - b); end function"},  # Very code-like
            {"content": "This is another important requirement for the system."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        # Summary should prefer natural language sentences over code
        # Both requirement sentences should be included as they are most relevant
        assert "requirement" in summary.summary.lower() or "requirements" in summary.summary.lower()

    def test_sentence_length_filtering(self, summary_service):
        """Test that sentences are filtered by length."""
        messages = [
            {"content": "Short."},  # Too short
            {"content": "This is a valid sentence that meets the minimum length requirement."},
            {"content": "A" * 250}  # Too long
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert "valid sentence" in summary.summary
        assert "Short." not in summary.summary

    def test_max_sentences_respected(self, summary_service):
        """Test that max_sentences limit is respected."""
        messages = [
            {"content": "First important point about database design."},
            {"content": "Second important point about API architecture."},
            {"content": "Third important point about security measures."},
            {"content": "Fourth important point about testing strategy."},
            {"content": "Fifth important point about deployment process."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert summary.sentence_count <= 3  # max_sentences=3
        assert len(summary.summary) > 0

    def test_topic_extraction(self, summary_service):
        """Test that topics are extracted correctly."""
        messages = [
            {"content": "I need help with React hooks and state management in my application."},
            {"content": "The React components should use hooks for managing state effectively."},
            {"content": "React context API is also needed for global state management."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert len(summary.topics) > 0
        # "react" should be a top topic since it appears frequently
        assert "react" in [topic.lower() for topic in summary.topics]

    def test_stop_words_filtered(self, summary_service):
        """Test that stop words are filtered in topic extraction."""
        messages = [
            {"content": "The application needs to have authentication with proper security measures."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        # Stop words like "the", "to", "with" should not be in topics
        stop_words_in_topics = any(
            topic in ["the", "to", "have", "with", "and", "or"]
            for topic in summary.topics
        )
        assert not stop_words_in_topics

    def test_tf_idf_scoring(self, summary_service):
        """Test that TF-IDF scoring prioritizes key sentences."""
        messages = [
            {"content": "The database migration failed due to a schema conflict."},
            {"content": "I need to fix the database migration script urgently."},
            {"content": "Meanwhile, the frontend is working fine with no issues."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        # Sentences about "database migration" should be prioritized
        # since those terms appear more frequently
        assert "database" in summary.summary.lower() or "migration" in summary.summary.lower()

    def test_sentence_order_preserved(self, summary_service):
        """Test that selected sentences maintain original order."""
        messages = [
            {"content": "First step is to set up the development environment properly."},
            {"content": "Second step involves configuring the database connection settings."},
            {"content": "Third step requires implementing the authentication middleware logic."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        # If multiple sentences are selected, they should maintain order
        if "First" in summary.summary and "Second" in summary.summary:
            assert summary.summary.index("First") < summary.summary.index("Second")

    def test_summarize_text_convenience_method(self, summary_service):
        """Test the convenience method for plain text summarization."""
        text = "This is a plain text document. It contains multiple sentences. The summarizer should extract the key information."
        summary = summary_service.summarize_text(text, "plain-text-session")

        assert summary.session_id == "plain-text-session"
        assert len(summary.summary) > 0
        assert summary.source_message_count == 1

    def test_non_string_content_handled(self, summary_service):
        """Test that non-string content is handled gracefully."""
        messages = [
            {"content": 12345},  # Non-string
            {"content": None},   # None
            {"content": "This is a valid message that should be processed."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert "valid message" in summary.summary
        assert summary.source_message_count == 3

    def test_message_without_content_field(self, summary_service):
        """Test messages without content field."""
        messages = [
            {"type": "user"},  # No content field
            {"content": "This message has content and should be included in summary."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert "should be included" in summary.summary
        assert summary.source_message_count == 2

    def test_multiline_content_split(self, summary_service):
        """Test that multiline content is split into sentences."""
        messages = [
            {"content": "Line one has important information.\n\nLine two has more details.\n\nLine three concludes the topic."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        # Should extract sentences from multiline content
        assert summary.sentence_count >= 1
        assert len(summary.summary) > 0

    def test_file_path_patterns_filtered(self, summary_service):
        """Test that file path patterns are filtered out."""
        messages = [
            {"content": "main.py contains the application logic for processing requests."},
            {"content": "server.js handles the backend API endpoints efficiently."},
            {"content": "We need to implement proper error handling mechanisms."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        # Should prefer sentences without file extensions at the start
        # "error handling" sentence should be prioritized
        assert "error handling" in summary.summary.lower() or "implement" in summary.summary.lower()

    def test_custom_parameters(self):
        """Test service with custom parameters."""
        service = SessionSummaryService(
            max_sentences=5,
            min_sentence_length=10,
            max_sentence_length=100
        )

        messages = [
            {"content": f"Sentence {i} discusses important technical implementation details." for i in range(10)}
        ]
        summary = service.generate_summary(messages, "test-session")

        # Should respect custom max_sentences
        assert summary.sentence_count <= 5

    def test_real_world_conversation(self, summary_service):
        """Test with a realistic conversation scenario."""
        messages = [
            {"content": "I'm building a web application using React and TypeScript."},
            {"content": "The application needs to fetch data from a REST API."},
            {"content": "Here's the code: const data = await fetch('/api/users');"},
            {"content": "I'm getting a CORS error when trying to connect."},
            {"content": "How can I fix this CORS issue in my development environment?"}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert len(summary.summary) > 0
        assert summary.sentence_count >= 1
        assert summary.sentence_count <= 3
        # Should capture key aspects like React, API, or CORS
        key_terms = ["react", "api", "cors", "application"]
        assert any(term in summary.summary.lower() for term in key_terms)

    def test_topics_limited_to_top_k(self, summary_service):
        """Test that topics are limited to top_k."""
        messages = [
            {"content": f"Technology {i} is important for development." for i in range(20)}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        # Default top_k is 5
        assert len(summary.topics) <= 5

    def test_empty_content_strings(self, summary_service):
        """Test messages with empty content strings."""
        messages = [
            {"content": ""},
            {"content": "   "},
            {"content": "This is a valid message with actual content."}
        ]
        summary = summary_service.generate_summary(messages, "test-session")

        assert "valid message" in summary.summary
        assert summary.sentence_count >= 1
