"""
Session summarization service using extractive summarization.

This module provides lightweight extractive summarization for Claude Code sessions
using TF-IDF scoring and sentence ranking. The summaries provide quick overviews
of session content without requiring LLM API calls.
"""

import re
import logging
from typing import List, Dict, Optional
from collections import Counter
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Represents a summary of a Claude Code session."""
    session_id: str
    summary: str
    sentence_count: int
    source_message_count: int
    topics: List[str]


class SessionSummaryService:
    """
    Service for generating extractive summaries of Claude Code sessions.

    Uses TF-IDF scoring to identify key sentences that best represent
    the session content. Summaries are lightweight and fast to generate.
    """

    def __init__(
        self,
        max_sentences: int = 3,
        min_sentence_length: int = 20,
        max_sentence_length: int = 200
    ):
        """
        Initialize the SessionSummaryService.

        Args:
            max_sentences: Maximum number of sentences in summary
            min_sentence_length: Minimum character length for a sentence
            max_sentence_length: Maximum character length for a sentence
        """
        self.max_sentences = max_sentences
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length

        # Common stop words for TF-IDF filtering
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'can', 'could', 'would',
            'should', 'i', 'you', 'we', 'they', 'my', 'your', 'their', 'our',
            'me', 'him', 'her', 'them', 'us', 'am', 'do', 'does', 'did',
            'been', 'being', 'have', 'had', 'having', 'or', 'but', 'if',
            'because', 'when', 'where', 'why', 'how', 'which', 'who', 'whom'
        }

    def generate_summary(
        self,
        messages: List[Dict],
        session_id: str
    ) -> SessionSummary:
        """
        Generate an extractive summary from session messages.

        Args:
            messages: List of message dictionaries from session
            session_id: ID of the session being summarized

        Returns:
            SessionSummary object with extracted key sentences
        """
        if not messages:
            return SessionSummary(
                session_id=session_id,
                summary="Empty session with no messages.",
                sentence_count=0,
                source_message_count=0,
                topics=[]
            )

        # Extract text content from messages
        sentences = self._extract_sentences(messages)

        if not sentences:
            return SessionSummary(
                session_id=session_id,
                summary="Session contains no extractable text.",
                sentence_count=0,
                source_message_count=len(messages),
                topics=[]
            )

        # Score sentences using TF-IDF
        scored_sentences = self._score_sentences(sentences)

        # Select top sentences (maintain original order)
        top_sentences = self._select_top_sentences(
            sentences,
            scored_sentences,
            self.max_sentences
        )

        # Extract topics (most common significant terms)
        topics = self._extract_topics(sentences, top_k=5)

        # Combine into summary text
        summary_text = " ".join(top_sentences)

        return SessionSummary(
            session_id=session_id,
            summary=summary_text,
            sentence_count=len(top_sentences),
            source_message_count=len(messages),
            topics=topics
        )

    def _extract_sentences(self, messages: List[Dict]) -> List[str]:
        """
        Extract sentences from messages.

        Args:
            messages: List of message dictionaries

        Returns:
            List of cleaned sentences
        """
        sentences = []

        for message in messages:
            content = message.get('content', '')
            if not isinstance(content, str):
                continue

            # Skip very short messages
            if len(content.strip()) < self.min_sentence_length:
                continue

            # Split into sentences (simple heuristic)
            # Match sentences ending with . ! ? followed by space or end
            raw_sentences = re.split(r'[.!?]\s+|\n{2,}', content)

            for sentence in raw_sentences:
                sentence = sentence.strip()

                # Filter by length
                if (len(sentence) < self.min_sentence_length or
                    len(sentence) > self.max_sentence_length):
                    continue

                # Skip code-heavy sentences (heuristic: contains many special chars)
                special_chars = sum(1 for c in sentence if c in '{}[]()<>=;:')
                if special_chars > len(sentence) * 0.2:
                    continue

                # Skip sentences that are mostly code patterns
                if re.match(r'^[\w\-\.]+\.(py|js|ts|java|cpp|rs|go)', sentence):
                    continue

                sentences.append(sentence)

        return sentences

    def _score_sentences(self, sentences: List[str]) -> Dict[int, float]:
        """
        Score sentences using TF-IDF.

        Args:
            sentences: List of sentences to score

        Returns:
            Dictionary mapping sentence index to score
        """
        if not sentences:
            return {}

        # Tokenize sentences into words
        tokenized = []
        for sentence in sentences:
            # Convert to lowercase and extract words
            words = re.findall(r'\b[a-z]{3,}\b', sentence.lower())
            # Filter stop words
            words = [w for w in words if w not in self.stop_words]
            tokenized.append(words)

        # Calculate term frequency for each sentence
        tf_scores = []
        for words in tokenized:
            word_count = len(words)
            if word_count == 0:
                tf_scores.append({})
                continue

            word_freq = Counter(words)
            # Normalize by sentence length
            tf = {word: count / word_count for word, count in word_freq.items()}
            tf_scores.append(tf)

        # Calculate inverse document frequency
        doc_count = len(tokenized)
        all_words = set()
        for words in tokenized:
            all_words.update(words)

        idf = {}
        for word in all_words:
            # Count how many sentences contain this word
            containing_docs = sum(1 for words in tokenized if word in words)
            # IDF = log(total_docs / docs_containing_word)
            idf[word] = math.log((doc_count + 1) / (containing_docs + 1)) + 1

        # Calculate TF-IDF scores for each sentence
        sentence_scores = {}
        for i, tf in enumerate(tf_scores):
            # Sum of TF-IDF for all words in sentence
            score = sum(tf_value * idf.get(word, 0) for word, tf_value in tf.items())
            sentence_scores[i] = score

        return sentence_scores

    def _select_top_sentences(
        self,
        sentences: List[str],
        scores: Dict[int, float],
        top_n: int
    ) -> List[str]:
        """
        Select top-scoring sentences while maintaining original order.

        Args:
            sentences: Original list of sentences
            scores: Dictionary of sentence scores
            top_n: Number of sentences to select

        Returns:
            List of top sentences in original order
        """
        if not scores:
            return []

        # Sort by score (descending)
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

        # Take top N
        top_indices = sorted_indices[:top_n]

        # Sort by original order
        top_indices.sort()

        # Extract sentences
        return [sentences[i] for i in top_indices if i < len(sentences)]

    def _extract_topics(self, sentences: List[str], top_k: int = 5) -> List[str]:
        """
        Extract key topics from sentences using term frequency.

        Args:
            sentences: List of sentences
            top_k: Number of top topics to extract

        Returns:
            List of topic terms
        """
        # Tokenize all sentences
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b[a-z]{3,}\b', sentence.lower())
            words = [w for w in words if w not in self.stop_words]
            all_words.extend(words)

        # Count frequencies
        word_freq = Counter(all_words)

        # Get top K most common
        top_terms = [word for word, count in word_freq.most_common(top_k)]

        return top_terms

    def summarize_text(self, text: str, session_id: str = "unknown") -> SessionSummary:
        """
        Generate summary from plain text.

        Convenience method for summarizing text that's not in message format.

        Args:
            text: Plain text to summarize
            session_id: Optional session ID

        Returns:
            SessionSummary object
        """
        # Convert to message format
        messages = [{"content": text}]
        return self.generate_summary(messages, session_id)
