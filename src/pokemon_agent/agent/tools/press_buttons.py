"""Tool for pressing Game Boy buttons on the emulator."""

import logging
from typing import Annotated, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from pokemon_agent.agent.tools._observation import observe_after_action

logger = logging.getLogger(__name__)

Button = Literal["a", "b", "start", "select", "up", "down", "left", "right"]


@tool
def press_buttons(
    say: Annotated[
        str,
        "What you say out loud to your audience right now, before acting. 1-2 short "
        "conversational sentences reacting to what's on screen and what you're about "
        "to do. Spoken via text-to-speech: no technical jargon, no button names.",
    ],
    buttons: Annotated[
        list[Button],
        "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', "
        "'select', 'up', 'down', 'left', 'right'",
    ],
    state: Annotated[dict, InjectedState],
    wait: Annotated[
        bool,
        "Whether to wait for a brief period after pressing each button. Defaults to true.",
    ] = True,
) -> dict:
    """Press a sequence of buttons on the Game Boy, narrating as you play.

    Returns the result of the button presses along with a screenshot of the
    screen afterwards, game state information read from memory, and a collision
    map of the visible area (when in the overworld).
    """
    emulator = state["emulator"]
    logger.info(f"[Say] {say}")

    # Voice gate: start synthesizing now, press buttons only once the previous
    # line finished and this one starts playing — speech stays in sync with
    # what's happening on screen.
    speaker = state.get("speaker")
    if speaker is not None:
        speaker.submit(say).wait_started()

    logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")

    emulator.press_buttons(buttons, wait)

    return {
        "result": f"Pressed buttons: {', '.join(buttons)}",
        **observe_after_action(emulator),
    }
