import argparse
import faulthandler
import logging
import os
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
        "--max-history",
        type=int,
        default=settings.max_history,
        help="Maximum number of messages in history before summarization",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="game.state",
        help="Path where the game state is resumed from and saved to",
    )
    parser.add_argument(
        "--new-game",
        action="store_true",
        help="Start a fresh game instead of resuming the saved state",
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

    # Resolve the save-state path next to this file (like the ROM)
    if not os.path.isabs(args.state):
        state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.state)
    else:
        state_path = args.state

    emulator = Emulator(
        rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
    )
    emulator.initialize()
    if not args.new_game and os.path.exists(state_path):
        logger.info(f"Resuming saved game from {state_path}")
        emulator.load_state(state_path)

    graph = build_game_graph()
    speaker = create_speaker(settings)

    initial_state = {
        "messages": build_initial_messages(emulator),
        "emulator": emulator,
        "speaker": speaker,
        "max_steps": args.steps,
        "max_history": args.max_history,
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
            emulator.save_state(state_path)
            logger.info(f"Saved game state to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save game state: {e}")
        speaker.stop()
        emulator.stop()


if __name__ == "__main__":
    main()
