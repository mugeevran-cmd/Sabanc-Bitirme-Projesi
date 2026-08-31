"""Narrative layer for the agent committee.

Architectural decision
----------------------
The agents' *findings and the final directive are computed deterministically in
Python*; the language model only writes the prose that explains them.

This split matters for a financial system.  If an LLM produced the Buy/Sell/Hold
call directly, the same market state could yield different directives on
different runs, the decision could not be unit-tested, and a hallucinated number
would flow straight into a trading recommendation.  With the split, every
directive is reproducible and auditable, and the model is used for what it is
genuinely good at -- turning a structured finding into readable analysis.

If ``ANTHROPIC_API_KEY`` is set the prose is written by Claude.  Otherwise a
template renderer produces the same report structure offline, so the full
pipeline runs with no external dependency.
"""
from __future__ import annotations

import logging

from finagent_pulse import config

log = logging.getLogger(__name__)


class NarrativeWriter:
    """Writes agent prose, via Claude when available and templates otherwise."""

    def __init__(self) -> None:
        self.api_key = config.anthropic_key()
        self._client = None
        # What the last attempted call actually did. ``mode`` reports this
        # rather than reporting whether a key was configured, because those two
        # answers diverge exactly when it matters: credit runs out mid-demo, the
        # calls start failing, the templates come back, and a report that still
        # says "llm" is claiming an author it does not have.
        self.last_error: str | None = None
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                log.info("NarrativeWriter: using %s", config.ANTHROPIC_MODEL)
            except Exception as exc:                       # pragma: no cover
                log.warning("Anthropic client unavailable (%s); using templates", exc)
                self._client = None
        else:
            log.info("NarrativeWriter: no API key -- deterministic template mode")

    @property
    def mode(self) -> str:
        """Who wrote the most recent prose: ``llm``, or a template and why."""
        if self._client is None:
            return "template"
        if self.last_error is not None:
            return "template (llm unavailable)"
        return "llm"

    @property
    def degraded(self) -> bool:
        """True when a key is configured but the calls are not getting through."""
        return self._client is not None and self.last_error is not None

    def write(self, system: str, prompt: str, fallback: str) -> str:
        """Return LLM prose, or ``fallback`` when no model is configured.

        Any API failure falls back rather than raising: a narrative outage must
        never take down the analytical pipeline. It does, however, get recorded
        -- silently degrading and still reporting ``llm`` would misattribute the
        text to a model that never saw it.
        """
        if self._client is None:
            return fallback
        try:
            resp = self._client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            self.last_error = None
            return text
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("LLM call failed (%s); falling back to template", exc)
            return fallback


_WRITER: NarrativeWriter | None = None


def get_writer() -> NarrativeWriter:
    global _WRITER
    if _WRITER is None:
        _WRITER = NarrativeWriter()
    return _WRITER
