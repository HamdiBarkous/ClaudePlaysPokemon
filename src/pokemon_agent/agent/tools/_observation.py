"""Shared post-action observation: screenshot, memory state, and collision map."""

import logging

from pokemon_agent.emulator import get_screenshot_data_url

logger = logging.getLogger(__name__)


def observe_after_action(emulator) -> dict:
    """Capture the game observation after a tool action.

    Returns a dict with the memory state, collision map (overworld only), and a
    screenshot as a data URL. VisionToolNode splits the screenshot out into an
    image_url content block so the model can see it.
    """
    memory_info = emulator.get_state_from_memory()
    logger.info("[Memory State after action]")
    logger.info(memory_info)

    collision_map = emulator.get_collision_map()
    if collision_map:
        logger.info(f"[Collision Map after action]\n{collision_map}")

    observation = {
        "memory_info": memory_info,
        "screenshot": get_screenshot_data_url(emulator, upscale=2),
    }
    if collision_map:
        observation["collision_map"] = collision_map
    return observation
