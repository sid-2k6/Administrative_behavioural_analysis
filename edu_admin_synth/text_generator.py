"""
text_generator.py
=================

Modality 1 - textual educational communication.

A lightweight *compositional* language engine assembles unique documents from
large vocabulary / phrase banks whose selection probabilities are driven by
the administrator's latent state.  This produces authentic-looking, varied
educational communication (emails, minutes, circulars, notices, reports,
faculty communication) without repeated templates.

Writing style is a direct function of the hidden profile / latent traits:

* high communication + high sentiment  -> supportive, collaborative, inclusive
* low communication                    -> short, directive
* high stress / workload               -> rushed, urgent, terse

For every document we also derive interpretable features (sentiment score,
positive/negative word ratios, urgency, topic, readability, word count) so the
text modality can feed both NLP models and classical feature pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import (
    COMMUNICATION_TYPES,
    TEXT_TOPICS,
    URGENCY_LEVELS,
    WORD_COUNT_GAMMA,
    WORD_COUNT_MAX,
    WORD_COUNT_MIN,
    Config,
)
from .temporal import MonthlyState

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
POSITIVE_WORDS = [
    "excellent", "outstanding", "collaborative", "supportive", "constructive",
    "encouraging", "proactive", "dedicated", "successful", "productive",
    "inclusive", "innovative", "commendable", "effective", "positive",
    "appreciated", "grateful", "committed", "thriving", "empowering",
]
NEGATIVE_WORDS = [
    "delayed", "inadequate", "concerning", "unacceptable", "insufficient",
    "overdue", "problematic", "disappointing", "critical", "urgent",
    "backlog", "shortfall", "deficient", "strained", "unresolved",
    "escalating", "understaffed", "overburdened", "noncompliant", "lacking",
]
NEUTRAL_CONNECTORS = [
    "Furthermore", "In addition", "Consequently", "Accordingly", "Meanwhile",
    "As a result", "With regard to this", "In this context", "Moving forward",
    "On a related note", "To this end", "Subsequently",
]
ACTION_VERBS = [
    "review", "finalize", "coordinate", "submit", "schedule", "evaluate",
    "implement", "monitor", "prepare", "approve", "circulate", "consolidate",
]
EDU_NOUNS = [
    "the curriculum committee", "faculty members", "the examination cell",
    "the quality assurance team", "department heads", "the academic council",
    "student representatives", "the administrative office", "the review panel",
    "the coordination unit",
]

GREETINGS_WARM = [
    "Dear colleagues,", "Dear team,", "Respected faculty members,",
    "Dear all,", "Greetings to the entire team,",
]
GREETINGS_FORMAL = [
    "To all concerned,", "For the attention of all staff,",
    "This is to inform all members,", "Notice to all departments,",
]
CLOSINGS_WARM = [
    "Thank you for your continued dedication and support.",
    "I truly appreciate the effort everyone continues to put in.",
    "Please feel free to reach out with any questions or ideas.",
    "Together we will continue to move forward positively.",
]
CLOSINGS_TERSE = [
    "Compliance is expected without exception.",
    "Immediate action is required.",
    "Please treat this as a priority.",
    "No further reminders will be issued.",
]

_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


@dataclass
class TextStyle:
    """Resolved stylistic parameters derived from a latent state."""

    positivity: float      # 0..1 tendency to use positive words
    formality: float       # 0..1 (higher = more polished, longer sentences)
    directiveness: float   # 0..1 (higher = terse/commanding)
    urgency: float         # 0..1


@dataclass
class TextDocument:
    """A single generated communication document with derived features."""

    document_id: str
    admin_id: str
    month: str
    month_step: int
    communication_type: str
    subject: str
    document_text: str
    word_count: int
    sentiment_score: float
    positive_word_ratio: float
    negative_word_ratio: float
    urgency_level: str
    topic: str
    readability_score: float
    writing_style: str


class TextGenerator:
    """Generates the textual modality for every administrator-month."""

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    # ------------------------------------------------------------------
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Approximate the syllable count of a word via vowel groups."""
        groups = _VOWEL_GROUP.findall(word)
        return max(1, len(groups))

    # ------------------------------------------------------------------
    def _resolve_style(self, state: MonthlyState) -> TextStyle:
        """Translate latent traits into concrete writing-style parameters."""
        lat = state.latent
        positivity = float(np.clip(0.5 * lat["sentiment"] + 0.5 * lat["communication"], 0, 1))
        formality = float(np.clip(0.7 * lat["communication"] + 0.3 * lat["leadership"], 0, 1))
        directiveness = float(np.clip(0.6 * (1 - lat["communication"]) + 0.4 * lat["stress"], 0, 1))
        urgency = float(np.clip(0.55 * lat["stress"] + 0.45 * lat["workload"], 0, 1))
        return TextStyle(positivity, formality, directiveness, urgency)

    # ------------------------------------------------------------------
    def _style_label(self, style: TextStyle) -> str:
        """Produce a short human-readable style descriptor for metadata."""
        tags: List[str] = []
        tags.append("supportive" if style.positivity > 0.6 else
                    "neutral" if style.positivity > 0.4 else "negative")
        tags.append("professional" if style.formality > 0.6 else "informal")
        if style.directiveness > 0.6:
            tags.append("directive")
        if style.urgency > 0.6:
            tags.append("urgent")
        if style.formality > 0.7 and style.urgency > 0.55:
            tags.append("rushed")
        return "-".join(tags)

    # ------------------------------------------------------------------
    def _pick_word(self, style: TextStyle) -> str:
        """Choose a sentiment-laden word consistent with the writing style."""
        if self._rng.random() < style.positivity:
            return str(self._rng.choice(POSITIVE_WORDS))
        return str(self._rng.choice(NEGATIVE_WORDS))

    # ------------------------------------------------------------------
    def _make_sentence(self, topic: str, style: TextStyle) -> str:
        """Compose a single varied sentence about the topic."""
        verb = str(self._rng.choice(ACTION_VERBS))
        noun = str(self._rng.choice(EDU_NOUNS))
        adj = self._pick_word(style)
        connector = str(self._rng.choice(NEUTRAL_CONNECTORS))

        templates = [
            f"{connector}, we must {verb} matters relating to {topic} with {noun}.",
            f"The progress on {topic} has been {adj}, and {noun} should {verb} it further.",
            f"I would like {noun} to {verb} the {adj} developments concerning {topic}.",
            f"It is {adj} that {noun} continue to {verb} the {topic} initiative.",
            f"{connector}, the {topic} plan requires that we {verb} inputs from {noun}.",
        ]
        # Terse styles favour short, direct sentences.
        if style.directiveness > 0.6 and self._rng.random() < style.directiveness:
            templates = [
                f"{noun.capitalize()} must {verb} {topic} now.",
                f"Please {verb} {topic} immediately.",
                f"{topic.capitalize()} is {adj}. {noun.capitalize()} must act.",
            ]
        return str(self._rng.choice(templates))

    # ------------------------------------------------------------------
    def _compose_document(
        self, state: MonthlyState, style: TextStyle, comm_type: str, topic: str
    ) -> str:
        """Assemble a full document body targeting a sampled word count."""
        target = int(
            np.clip(
                self._rng.gamma(*WORD_COUNT_GAMMA),
                WORD_COUNT_MIN,
                WORD_COUNT_MAX,
            )
        )

        parts: List[str] = []
        greeting_pool = GREETINGS_WARM if style.positivity > 0.5 else GREETINGS_FORMAL
        parts.append(str(self._rng.choice(greeting_pool)))

        opener = (
            f"This {comm_type.lower()} concerns {topic} for the current academic period"
        )
        if "Accreditation Visit" in state.events:
            opener += ", particularly in view of the upcoming accreditation review"
        elif "Exam Period" in state.events:
            opener += ", especially given the ongoing examination schedule"
        parts.append(opener + ".")

        # Add body sentences until we approach the target word length.
        def word_len(text: str) -> int:
            return len(_WORD_RE.findall(text))

        while word_len(" ".join(parts)) < target:
            parts.append(self._make_sentence(topic, style))
            # Occasionally add a bulleted action item for realism.
            if self._rng.random() < 0.15:
                parts.append(
                    f"Action item: {self._rng.choice(ACTION_VERBS)} the "
                    f"{topic} documentation before the next meeting."
                )

        closing_pool = CLOSINGS_WARM if style.directiveness < 0.5 else CLOSINGS_TERSE
        parts.append(str(self._rng.choice(closing_pool)))

        text = " ".join(parts)

        # Trim gently to the target if we overshot substantially.
        words = _WORD_RE.findall(text)
        if len(words) > target + 40:
            # Rebuild from sentences to avoid mid-sentence cuts.
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
            rebuilt: List[str] = []
            for sent in sentences:
                rebuilt.append(sent + ".")
                if len(_WORD_RE.findall(" ".join(rebuilt))) >= target:
                    break
            text = " ".join(rebuilt)
        return text

    # ------------------------------------------------------------------
    def _compute_features(
        self, text: str, style: TextStyle
    ) -> tuple[int, float, float, float, float]:
        """Return (word_count, sentiment, pos_ratio, neg_ratio, readability)."""
        words = _WORD_RE.findall(text.lower())
        word_count = len(words)
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        pos_ratio = pos / word_count if word_count else 0.0
        neg_ratio = neg / word_count if word_count else 0.0

        # Sentiment blends the latent-driven positivity with the actually
        # measured lexical balance, then adds small measurement noise.
        measured = (pos - neg) / max(1, pos + neg) if (pos + neg) else 0.0
        latent_based = 2 * style.positivity - 1  # map 0..1 -> -1..1
        sentiment = 0.6 * latent_based + 0.4 * measured + self._rng.normal(0, 0.05)
        sentiment = float(np.clip(sentiment, -1.0, 1.0))

        # Flesch Reading Ease.
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        n_sent = max(1, len(sentences))
        syllables = sum(self._count_syllables(w) for w in words) or 1
        readability = (
            206.835
            - 1.015 * (word_count / n_sent)
            - 84.6 * (syllables / max(1, word_count))
        )
        readability = float(np.clip(readability, 0.0, 100.0))
        return word_count, sentiment, pos_ratio, neg_ratio, readability

    # ------------------------------------------------------------------
    def _urgency_level(self, style: TextStyle) -> str:
        """Map the continuous urgency score onto an ordinal category."""
        u = style.urgency + self._rng.normal(0, 0.05)
        if u < 0.35:
            return URGENCY_LEVELS[0]  # Low
        if u < 0.6:
            return URGENCY_LEVELS[1]  # Medium
        if u < 0.82:
            return URGENCY_LEVELS[2]  # High
        return URGENCY_LEVELS[3]      # Critical

    # ------------------------------------------------------------------
    def _n_documents(self, state: MonthlyState) -> int:
        """Number of documents produced this month (workload-driven Poisson)."""
        lam = 1.4 * state.intensity * (0.7 + 0.6 * state.latent["workload"])
        n = int(self._rng.poisson(lam))
        return int(np.clip(n, 1, 5))

    # ------------------------------------------------------------------
    def generate_for_state(self, state: MonthlyState) -> List[TextDocument]:
        """Generate all documents authored by one administrator in one month."""
        style = self._resolve_style(state)
        style_label = self._style_label(style)
        docs: List[TextDocument] = []

        for k in range(self._n_documents(state)):
            comm_type = str(self._rng.choice(COMMUNICATION_TYPES))
            topic = str(self._rng.choice(TEXT_TOPICS))
            text = self._compose_document(state, style, comm_type, topic)
            wc, sentiment, pos_r, neg_r, readability = self._compute_features(text, style)
            subject = f"{comm_type}: {topic.title()} - {state.month_label}"

            docs.append(
                TextDocument(
                    document_id=f"D{state.admin_id}_{state.month_step:02d}_{k+1}",
                    admin_id=state.admin_id,
                    month=state.month_label,
                    month_step=state.month_step,
                    communication_type=comm_type,
                    subject=subject,
                    document_text=text,
                    word_count=wc,
                    sentiment_score=round(sentiment, 4),
                    positive_word_ratio=round(pos_r, 4),
                    negative_word_ratio=round(neg_r, 4),
                    urgency_level=self._urgency_level(style),
                    topic=topic,
                    readability_score=round(readability, 2),
                    writing_style=style_label,
                )
            )
        return docs

    # ------------------------------------------------------------------
    def generate(self, states: List[MonthlyState]) -> List[TextDocument]:
        """Generate the full text modality across all administrator-months."""
        documents: List[TextDocument] = []
        for state in states:
            documents.extend(self.generate_for_state(state))
        return documents
