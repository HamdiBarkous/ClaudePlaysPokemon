"""Pokemon Agent entry point.

Start a fresh game (saves never get overwritten — this just skips loading):

    uv run main.py --rom "Pokemon Red.gb" --steps 50 --display --new-game

Resume from the most recent save in states/ (the default):

    uv run main.py --rom "Pokemon Red.gb" --steps 50 --display

Every run saves a new states/game-NNN.state on exit (including Ctrl-C), so
history accumulates and any save can be branched from later with
--state states/game-007.state. Game sound is off unless you pass --sound;
silence the TTS voice too with TTS_ENGINE=none. Drop --display for headless.
"""

import argparse
import faulthandler
import logging
import os
import re
import signal

# `kill -USR1 <pid>` dumps all thread stacks — for diagnosing a stuck run
faulthandler.register(signal.SIGUSR1)

from langgraph.errors import GraphRecursionError

from pokemon_agent.agent import build_game_graph, build_initial_messages
from pokemon_agent.core import get_settings
from pokemon_agent.emulator import Emulator
from pokemon_agent.voice import create_speaker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_STATE_FILE_RE = re.compile(r"^game-(\d+)\.state$")


def _resolve(path: str) -> str:
    """Resolve a path relative to this file (like the ROM)."""
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _latest_state(states_dir: str) -> str | None:
    """Path of the highest-numbered save in the states folder, if any."""
    best_n, best = -1, None
    for name in os.listdir(states_dir):
        m = _STATE_FILE_RE.match(name)
        if m and int(m.group(1)) > best_n:
            best_n, best = int(m.group(1)), os.path.join(states_dir, name)
    return best


def _next_state_path(states_dir: str) -> str:
    """Next numbered save path (game-NNN.state) in the states folder."""
    numbers = [
        int(m.group(1))
        for name in os.listdir(states_dir)
        if (m := _STATE_FILE_RE.match(name))
    ]
    return os.path.join(states_dir, f"game-{max(numbers, default=0) + 1:03d}.state")


def main():
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Pokemon Agent")
    parser.add_argument(
        "--rom",
        type=str,
        default=settings.rom_path,
        help="Path to the Pokemon ROM file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=settings.max_steps,
        help="Number of agent steps to run",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Run with display (not headless)",
    )
    parser.add_argument(
        "--sound",
        action="store_true",
        help="Enable sound (only applicable with display)",
    )
    parser.add_argument(
        "--max-history-tokens",
        type=int,
        default=settings.max_history_tokens,
        help="Summarize the history once a turn's prompt reaches this many tokens",
    )
    parser.add_argument(
        "--states-dir",
        type=str,
        default="states",
        help="Folder where numbered saves (game-NNN.state) are written and resumed from",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Resume from this specific save file instead of the most recent one",
    )
    parser.add_argument(
        "--new-game",
        action="store_true",
        help="Start a fresh game instead of resuming a saved state",
    )

    args = parser.parse_args()

    # Get absolute path to ROM
    if not os.path.isabs(args.rom):
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rom)
    else:
        rom_path = args.rom

    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return

    states_dir = _resolve(args.states_dir)
    os.makedirs(states_dir, exist_ok=True)

    # Pick the state to resume: explicit --state, else most recent numbered
    # save, else the legacy single-file save from before numbered states
    load_path = None
    if args.state:
        load_path = _resolve(args.state)
        if not os.path.exists(load_path):
            logger.error(f"Save state not found: {load_path}")
            return
    elif not args.new_game:
        load_path = _latest_state(states_dir)
        legacy = _resolve("game.state")
        if load_path is None and os.path.exists(legacy):
            load_path = legacy

    emulator = Emulator(
        rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
    )
    emulator.initialize()
    if load_path:
        logger.info(f"Resuming saved game from {load_path}")
        emulator.load_state(load_path)

    graph = build_game_graph()
    speaker = create_speaker(settings)

    initial_state = {
        "messages": build_initial_messages(emulator),
        "emulator": emulator,
        "speaker": speaker,
        "max_steps": args.steps,
        "max_history_tokens": args.max_history_tokens,
        "step_count": 0,
    }

    try:
        logger.info(f"Starting agent for {args.steps} steps")
        final_state = graph.invoke(
            initial_state,
            # Each step is an agent + tools transition, plus occasional summarize
            config={"recursion_limit": args.steps * 3 + 10},
        )
        logger.info(f"Agent completed {final_state.get('step_count', 0)} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    except GraphRecursionError:
        logger.error(
            "Recursion limit reached before completing all steps — the model "
            "kept replying without tool calls. Stopping."
        )
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise
    finally:
        try:
            save_path = _next_state_path(states_dir)
            emulator.save_state(save_path)
            logger.info(f"Saved game state to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save game state: {e}")
        speaker.stop()
        emulator.stop()


if __name__ == "__main__":
    main()
