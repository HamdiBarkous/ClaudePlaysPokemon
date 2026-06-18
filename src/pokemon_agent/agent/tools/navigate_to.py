"""Tool for walking automatically to any explored coordinate on the map."""

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
    x: Annotated[int, "Target x coordinate, as labeled on the explored map."],
    y: Annotated[int, "Target y coordinate, as labeled on the explored map."],
    state: Annotated[dict, InjectedState],
) -> dict:
    """Walk automatically to (x, y) on the current map, narrating as you play.

    Uses the map coordinates shown in observations — the explored map's
    rulers, the doors/warps list, and the unexplored-spots list all use the
    same (x, y) system. The walk follows explored terrain, can go far beyond
    the visible screen, and routes around obstacles on its own. Targeting a
    door/warp walks through it; targeting a person/object walks up and faces
    it. Only works in the overworld, within the current map.

    Prefer this over press_buttons for any trip longer than a few steps.

    Returns the navigation result along with screenshots from the walk, game
    state information read from memory, and the explored map afterwards.
    """
    emulator = state["emulator"]
    logger.info(f"[Say] {say}")

    # Voice gate: see press_buttons — keeps speech in sync with the screen.
    speaker = state.get("speaker")
    if speaker is not None:
        speaker.submit(say).wait_started()

    logger.info(f"[Navigation] Navigating to ({x}, {y})")

    result, keyframes = emulator.navigate_to_global(x, y)
    logger.info(f"[Navigation] {result}")

    return {
        "result": f"Navigation result: {result}",
        **observe_after_action(emulator, keyframes, state.get("chat")),
    }
