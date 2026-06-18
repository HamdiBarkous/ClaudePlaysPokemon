"""Tests for the decision-point early-exit in Emulator.press_buttons.

Verifies the fix for blind button batches mashing through screens the model
hasn't seen yet (the "A x8 types AAAAAAA at the name menu" failure):

  - the new RAM/tilemap signals is_in_battle / is_menu_open read correctly
  - a batch stops when an A/START press opens a menu, a battle starts, or a warp
    happens — and skips the rest
  - directional cursor moves within a menu still batch (not over-blocked)
  - a full batch runs to completion when nothing notable happens
  - a single press never early-exits (navigate_to relies on this)

Runs under pytest, or standalone:  .venv/bin/python tests/test_press_buttons_early_exit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pokemon_agent.emulator.emulator import VALID_BUTTONS, Emulator
from pokemon_agent.emulator.memory_reader import PokemonRedReader

MAP_ID_ADDR = 0xD35E
BATTLE_ADDR = 0xD057
TILEMAP_START = 0xC3A0
TILEMAP_END = 0xC508
MENU_CURSOR_TILE = 0xED


# --------------------------------------------------------------------------- #
# Fakes: a scripted memory that changes state on each settle, and a minimal
# PyBoy stand-in. The Emulator is built without __init__ (no ROM, no threads);
# the per-press helpers (_settle_impl etc.) are stubbed so the test fully
# controls when state transitions happen.
# --------------------------------------------------------------------------- #

def phase(map_id=1, in_battle=False, menu_open=False):
    return {"map_id": map_id, "in_battle": in_battle, "menu_open": menu_open}


class ScriptedMemory:
    """Memory whose readings follow a list of phases.

    phases[0] is the state before the first press; phases[k] the state after
    the k-th press settles. ``advance()`` (called from the stubbed settle) steps
    to the next phase.
    """

    def __init__(self, phases):
        self.phases = phases
        self.idx = 0

    def advance(self, *_):
        if self.idx < len(self.phases) - 1:
            self.idx += 1

    @property
    def _cur(self):
        return self.phases[self.idx]

    def __getitem__(self, key):
        if isinstance(key, slice):  # tilemap region read by is_menu_open
            blob = bytearray(key.stop - key.start)
            if self._cur["menu_open"]:
                blob[0] = MENU_CURSOR_TILE
            return bytes(blob)
        if key == MAP_ID_ADDR:
            return self._cur["map_id"]
        if key == BATTLE_ADDR:
            return 1 if self._cur["in_battle"] else 0
        return 0


class FakePyBoy:
    def __init__(self, memory):
        self.memory = memory
        self.screen = type("S", (), {"ndarray": np.zeros((8, 8, 4), dtype=np.uint8)})()
        self.pressed = []

    def button_press(self, b):
        self.pressed.append(b)

    def button_release(self, b):
        pass


def make_emulator(phases):
    """An Emulator wired to a scripted memory, with per-press helpers stubbed."""
    mem = ScriptedMemory(phases)
    pyboy = FakePyBoy(mem)

    emu = object.__new__(Emulator)
    emu.pyboy = pyboy
    emu._run_on_pyboy = lambda fn, *a: fn(*a)  # run inline, no thread hop
    emu._tick_impl = lambda *a, **k: None
    emu._settle_impl = mem.advance               # each settle steps a phase
    emu._wait_map_ready_impl = lambda *a, **k: None
    emu._update_world_map_impl = lambda *a, **k: None
    return emu, pyboy


# --------------------------------------------------------------------------- #
# RAM / tilemap signal tests
# --------------------------------------------------------------------------- #

def test_is_in_battle_reads_battle_byte():
    mem = ScriptedMemory([phase(in_battle=False)])
    assert PokemonRedReader(mem).is_in_battle() is False
    mem = ScriptedMemory([phase(in_battle=True)])
    assert PokemonRedReader(mem).is_in_battle() is True


def test_is_menu_open_detects_cursor_tile():
    # A plain bytearray-backed memory: menu cursor present vs absent.
    base = bytearray(0x10000)
    reader = PokemonRedReader(base)
    assert reader.is_menu_open() is False           # all zero -> no cursor
    base[TILEMAP_START + 5] = 0xEE                   # ▼ continue arrow, not a menu
    assert reader.is_menu_open() is False
    base[TILEMAP_START + 5] = MENU_CURSOR_TILE       # ► selection cursor
    assert reader.is_menu_open() is True


# --------------------------------------------------------------------------- #
# Early-exit behavior tests
# --------------------------------------------------------------------------- #

def test_stops_when_A_opens_a_menu():
    # A x4, but the first A brings up a menu -> stop after one press.
    emu, pyboy = make_emulator([
        phase(menu_open=False),  # before press 1
        phase(menu_open=True),   # after press 1: menu is up
        phase(menu_open=True),
        phase(menu_open=True),
        phase(menu_open=True),
    ])
    result, keyframes = emu.press_buttons(["a", "a", "a", "a"])
    assert pyboy.pressed == ["a"]
    assert len(keyframes) == 1
    assert "Stopped early" in result and "a menu opened" in result


def test_stops_when_menu_opens_mid_batch():
    # First A only advances dialog; the second A opens a menu.
    emu, pyboy = make_emulator([
        phase(menu_open=False),
        phase(menu_open=False),  # after press 1: still dialog
        phase(menu_open=True),   # after press 2: menu up
        phase(menu_open=True),
    ])
    result, keyframes = emu.press_buttons(["a", "a", "a"])
    assert pyboy.pressed == ["a", "a"]
    assert len(keyframes) == 2
    assert "a menu opened" in result


def test_directional_moves_in_a_menu_do_not_stop():
    # Menu already open; deliberate down,down,a navigation must run to the end.
    emu, pyboy = make_emulator([
        phase(menu_open=True),
        phase(menu_open=True),
        phase(menu_open=True),
        phase(menu_open=True),
    ])
    result, keyframes = emu.press_buttons(["down", "down", "a"])
    assert pyboy.pressed == ["down", "down", "a"]
    assert len(keyframes) == 3
    assert "Stopped early" not in result


def test_stops_when_a_battle_starts():
    # Walking into grass; an encounter triggers on the second step.
    emu, pyboy = make_emulator([
        phase(in_battle=False),
        phase(in_battle=False),
        phase(in_battle=True),   # after press 2: battle
        phase(in_battle=True),
    ])
    result, keyframes = emu.press_buttons(["up", "up", "up"])
    assert pyboy.pressed == ["up", "up"]
    assert len(keyframes) == 2
    assert "a battle started" in result


def test_stops_when_a_warp_changes_the_map():
    emu, pyboy = make_emulator([
        phase(map_id=1),
        phase(map_id=1),
        phase(map_id=2),         # after press 2: new map
        phase(map_id=2),
    ])
    result, keyframes = emu.press_buttons(["down", "down", "down"])
    assert pyboy.pressed == ["down", "down"]
    assert len(keyframes) == 2
    assert "new area" in result


def test_full_batch_runs_when_nothing_notable_happens():
    emu, pyboy = make_emulator([phase() for _ in range(5)])
    result, keyframes = emu.press_buttons(["up", "up", "up", "up"])
    assert pyboy.pressed == ["up", "up", "up", "up"]
    assert len(keyframes) == 4
    assert "Stopped early" not in result
    assert result == "Pressed: up, up, up, up"


def test_single_press_never_early_exits():
    # navigate_to drives one button per call and must not be interrupted, even
    # though a menu is open after the press.
    emu, pyboy = make_emulator([
        phase(menu_open=False),
        phase(menu_open=True),
    ])
    result, keyframes = emu.press_buttons(["a"])
    assert pyboy.pressed == ["a"]
    assert len(keyframes) == 1
    assert "Stopped early" not in result


def test_valid_buttons_constant_is_complete():
    assert set(VALID_BUTTONS) == {
        "a", "b", "start", "select", "up", "down", "left", "right"
    }


# --------------------------------------------------------------------------- #
# Standalone runner (no pytest needed)
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
