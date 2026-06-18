"""Tests for the Twitch chat reader (no network).

Covers the risky part — parsing raw IRC lines — plus the poll/buffer
semantics, exercised by pushing messages directly instead of over a socket.

Runs under pytest, or standalone:
    .venv/bin/python tests/test_twitch_chat.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pokemon_agent.chat.twitch import ChatMessage, TwitchChat, parse_irc_message

PREFIX = ":alice!alice@alice.tmi.twitch.tv"


# --------------------------------------------------------------------------- #
# parse_irc_message
# --------------------------------------------------------------------------- #

def test_parses_tagged_privmsg_using_display_name():
    line = (
        "@badges=;color=#FF0000;display-name=Alice;mod=0 "
        f"{PREFIX} PRIVMSG #somechannel :hello world"
    )
    msg = parse_irc_message(line)
    assert msg == ChatMessage(author="Alice", text="hello world")


def test_parses_untagged_privmsg_using_nick():
    msg = parse_irc_message(f"{PREFIX} PRIVMSG #somechannel :hey there")
    assert msg == ChatMessage(author="alice", text="hey there")


def test_empty_display_name_falls_back_to_nick():
    line = f"@display-name= {PREFIX} PRIVMSG #somechannel :yo"
    msg = parse_irc_message(line)
    assert msg == ChatMessage(author="alice", text="yo")


def test_message_keeps_later_colons():
    # The first " :" is the trailing delimiter; emoticons etc. must survive.
    line = f"{PREFIX} PRIVMSG #somechannel :go left :) then up"
    msg = parse_irc_message(line)
    assert msg is not None
    assert msg.text == "go left :) then up"


def test_non_privmsg_lines_are_ignored():
    assert parse_irc_message("PING :tmi.twitch.tv") is None
    assert parse_irc_message(f"{PREFIX} JOIN #somechannel") is None
    assert parse_irc_message(":tmi.twitch.tv 001 justinfan123 :Welcome") is None


def test_privmsg_without_trailing_text_is_ignored():
    assert parse_irc_message(f"{PREFIX} PRIVMSG #somechannel") is None


# --------------------------------------------------------------------------- #
# poll / buffer
# --------------------------------------------------------------------------- #

def test_poll_drains_in_order_and_empties():
    chat = TwitchChat("SomeChannel")
    chat._append(ChatMessage("a", "first"))
    chat._append(ChatMessage("b", "second"))
    assert chat.poll() == [ChatMessage("a", "first"), ChatMessage("b", "second")]
    assert chat.poll() == []  # drained


def test_buffer_evicts_oldest_when_full():
    chat = TwitchChat("c", buffer_size=2)
    chat._append(ChatMessage("u", "1"))
    chat._append(ChatMessage("u", "2"))
    chat._append(ChatMessage("u", "3"))
    assert chat.poll() == [ChatMessage("u", "2"), ChatMessage("u", "3")]


def test_channel_name_is_normalized():
    assert TwitchChat("#FooBar").channel == "foobar"
    assert TwitchChat("FooBar").channel == "foobar"


def test_anonymous_nick_is_a_justinfan_login():
    assert TwitchChat("c")._nick.startswith("justinfan")


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

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
