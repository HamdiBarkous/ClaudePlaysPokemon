"""Shared post-action observation: keyframes, memory state, and explored map."""

import logging

from pokemon_agent.core import get_settings
from pokemon_agent.emulator import get_screenshot_data_url, image_to_data_url

logger = logging.getLogger(__name__)

MAX_CHAT_LINES = 15  # cap viewer chat shown per turn to bound token cost


def format_chat(messages, limit: int = MAX_CHAT_LINES) -> str:
    """Render recent viewer chat for an observation, newest last, capped.

    Returns "" when there is nothing to show, so quiet turns add no tokens.
    On a busy channel the oldest messages are dropped and noted, never silent.
    """
    if not messages:
        return ""
    shown = messages[-limit:]
    lines = [f"{m.author}: {m.text}" for m in shown]
    dropped = len(messages) - len(shown)
    if dropped > 0:
        lines.insert(0, f"(+{dropped} earlier message(s) omitted)")
    return "\n".join(lines)


def observe_after_action(emulator, keyframes=None, chat=None) -> dict:
    """Capture the game observation after a tool action.

    Returns a dict with the memory state, explored world map (overworld
    only), the action's screenshots, and any recent Twitch chat. With multiple
    keyframes (one per settled button press), the most recent max_keyframes are
    attached in press order so the model sees what happened inside the batch —
    VisionToolNode fans them out into ordered image_url content blocks.
    """
    memory_info = emulator.get_state_from_memory()
    logger.info("[Memory State after action]")
    logger.info(memory_info)

    world_map = emulator.get_world_map_text()
    if world_map:
        logger.info(f"[Explored map after action]\n{world_map}")

    observation = {"memory_info": memory_info}

    if keyframes and len(keyframes) > 1:
        limit = get_settings().max_keyframes
        kept = keyframes[-limit:]
        info = (
            "One screenshot per button press, in press order; the last one is "
            "the current state."
        )
        if len(kept) < len(keyframes):
            info += f" (showing the last {len(kept)} of {len(keyframes)} presses)"
        observation["screenshots_info"] = info
        observation["screenshots"] = [image_to_data_url(kf) for kf in kept]
    elif keyframes:
        observation["screenshot"] = image_to_data_url(keyframes[-1])
    else:
        observation["screenshot"] = get_screenshot_data_url(emulator)

    if world_map:
        observation["map"] = world_map

    if chat is not None:
        chat_text = format_chat(chat.poll())
        if chat_text:
            logger.info(f"[Chat]\n{chat_text}")
            observation["chat"] = chat_text

    return observation
