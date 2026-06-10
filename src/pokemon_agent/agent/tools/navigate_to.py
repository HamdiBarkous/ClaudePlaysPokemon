"""Tool for automatic pathfinding navigation in the overworld."""

import logging
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from pokemon_agent.agent.tools._observation import observe_after_action

logger = logging.getLogger(__name__)


@tool
def navigate_to(
    say: Annotated[
        str,
        "What you say out loud to your audience right now, before acting. 1-2 short "
        "conversational sentences reacting to what's on screen and where you're "
        "heading. Spoken via text-to-speech: no technical jargon, no coordinates.",
    ],
    row: Annotated[int, "The row coordinate to navigate to (0-8)."],
    col: Annotated[int, "The column coordinate to navigate to (0-9)."],
    state: Annotated[dict, InjectedState],
) -> dict:
    """Automatically navigate to a position on the map grid, narrating as you play.

    The screen is divided into a 9x10 grid, with the top-left corner as (0, 0).
    This tool is only available in the overworld.

    Returns the navigation result along with a screenshot of the screen
    afterwards, game state information read from memory, and a collision map
    of the visible area.
    """
    emulator = state["emulator"]
    logger.info(f"[Say] {say}")
    logger.info(f"[Navigation] Navigating to: ({row}, {col})")

    status, path = emulator.find_path(row, col)
    if path:
        for direction in path:
            emulator.press_buttons([direction], True)
        result = f"Navigation successful: followed path with {len(path)} steps"
    else:
        result = f"Navigation failed: {status}"

    return {
        "result": f"Navigation result: {result}",
        **observe_after_action(emulator),
    }
