"""What the report claims about who wrote it.

`mode` used to answer "is a key configured?" while presenting itself as "who
wrote this prose?". Those agree until the moment they matter most: credit runs
out mid-demo, every call starts failing, the deterministic templates come back,
and the footer still says `Narrative mode: llm` — the report crediting an author
that never saw it.

No network here. The Anthropic client is replaced with stubs that succeed or
raise on demand.
"""
from __future__ import annotations

import pytest

from finagent_pulse.agents.llm import NarrativeWriter

FALLBACK = "the deterministic template"


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Text(text)]


class _Client:
    """Stands in for anthropic.Anthropic. Fails whenever `error` is set."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.messages = self

    def create(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return _Response("prose from the model")


def writer(client: _Client | None) -> NarrativeWriter:
    """A writer wired to a stub, bypassing __init__'s key lookup."""
    w = NarrativeWriter.__new__(NarrativeWriter)
    w.api_key = "sk-ant-test" if client else None
    w._client = client
    w.last_error = None
    return w


# --------------------------------------------------------------------------
# No key at all
# --------------------------------------------------------------------------
def test_without_a_key_the_mode_is_template():
    w = writer(None)
    assert w.write("sys", "prompt", FALLBACK) == FALLBACK
    assert w.mode == "template"
    assert w.degraded is False


# --------------------------------------------------------------------------
# Key works
# --------------------------------------------------------------------------
def test_a_working_key_reports_llm():
    w = writer(_Client())
    assert w.write("sys", "prompt", FALLBACK) == "prose from the model"
    assert w.mode == "llm"
    assert w.degraded is False


# --------------------------------------------------------------------------
# Key configured but calls failing -- the case this test file exists for
# --------------------------------------------------------------------------
def test_a_failed_call_still_falls_back_rather_than_raising():
    """The analytical pipeline must survive a narrative outage."""
    w = writer(_Client(RuntimeError("credit balance is too low")))
    assert w.write("sys", "prompt", FALLBACK) == FALLBACK


def test_a_failed_call_stops_the_report_claiming_the_model_wrote_it():
    w = writer(_Client(RuntimeError("credit balance is too low")))
    w.write("sys", "prompt", FALLBACK)
    assert w.mode == "template (llm unavailable)"
    assert w.degraded is True


def test_the_reason_is_kept_so_the_dashboard_can_show_it():
    w = writer(_Client(RuntimeError("credit balance is too low")))
    w.write("sys", "prompt", FALLBACK)
    assert "credit balance is too low" in w.last_error
    assert "RuntimeError" in w.last_error


def test_recovery_clears_the_flag():
    """A transient failure must not brand the writer degraded forever."""
    client = _Client(RuntimeError("transient"))
    w = writer(client)
    w.write("sys", "prompt", FALLBACK)
    assert w.degraded is True

    client.error = None
    assert w.write("sys", "prompt", FALLBACK) == "prose from the model"
    assert w.mode == "llm"
    assert w.last_error is None


# --------------------------------------------------------------------------
# The footer the jury reads
# --------------------------------------------------------------------------
@pytest.mark.parametrize("client, expected", [
    (None, "template"),
    (_Client(), "llm"),
])
def test_the_executive_report_footer_names_the_actual_author(
        monkeypatch, client, expected):
    from finagent_pulse.agents import committee

    w = writer(client)
    w.write("sys", "prompt", FALLBACK)
    monkeypatch.setattr(committee, "get_writer", lambda: w)

    state = {
        "risk": {"as_of": "2024-03-04", "directive": "HOLD", "position_pct": 0.0,
                 "conviction": 0.5},
        "quant": {"forecast_7d_pct": 0.26, "volatility_regime": "normal"},
        "sentiment": {"sentiment_now": -0.161, "stance": "neutral"},
        "reports": {},
    }
    assert f"Narrative mode: `{expected}`" in committee.executive_report(state)


def test_the_footer_says_so_when_the_model_was_unreachable(monkeypatch):
    from finagent_pulse.agents import committee

    w = writer(_Client(RuntimeError("credit balance is too low")))
    w.write("sys", "prompt", FALLBACK)
    monkeypatch.setattr(committee, "get_writer", lambda: w)

    state = {
        "risk": {"as_of": "2024-03-04", "directive": "HOLD", "position_pct": 0.0,
                 "conviction": 0.5},
        "quant": {"forecast_7d_pct": 0.26, "volatility_regime": "normal"},
        "sentiment": {"sentiment_now": -0.161, "stance": "neutral"},
        "reports": {},
    }
    report = committee.executive_report(state)
    assert "Narrative mode: `template (llm unavailable)`" in report
    assert "`llm`" not in report
