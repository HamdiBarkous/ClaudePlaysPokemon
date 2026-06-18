"""Tests for rendering viewer chat into an observation (format_chat).

Runs under pytest, or standalone:
    .venv/bin/python tests/test_chat_observation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pokemon_agent.agent.tools._observation import MAX_CHAT_LINES, format_chat
from pokemon_agent.chat import ChatMessage


def _msgs(n, start=1):
    return [ChatMessage(f"u{i}", f"m{i}") for i in range(start, start + n)]


def test_empty_chat_renders_nothing():
    assert format_chat([]) == ""


def test_renders_author_and_text_newest_last():
    out = format_chat([ChatMessage("alice", "go left"), ChatMessage("bob", "no, up")])
    assert out == "alice: go left\nbob: no, up"


def test_caps_to_limit_and_keeps_most_recent():
    out = format_chat(_msgs(20), limit=5)
    lines = out.splitlines()
    # first line is the omitted-count note, then the 5 newest messages
    assert lines[0] == "(+15 earlier message(s) omitted)"
    assert lines[1:] == ["u16: m16", "u17: m17", "u18: m18", "u19: m19", "u20: m20"]


def test_at_limit_has_no_omitted_note():
    out = format_chat(_msgs(MAX_CHAT_LINES))
    assert "omitted" not in out
    assert len(out.splitlines()) == MAX_CHAT_LINES


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
