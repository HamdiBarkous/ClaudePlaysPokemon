"""ReAct agent graph for playing Pokemon Red.

The agent loop:
1. The model sees the conversation (screenshots, memory state, collision maps)
2. It reasons briefly and calls the press_buttons (or navigate_to) tool
3. VisionToolNode executes the tool and returns a multimodal observation
4. When a turn's prompt grows past max_history_tokens, the history is condensed into a summary
5. The loop ends after max_steps agent turns
"""

import logging
import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from pokemon_agent.agent.prompt_manager import PromptManager
from pokemon_agent.agent.state import GameAgentState
from pokemon_agent.agent.tools import navigate_to, press_buttons
from pokemon_agent.agent.vision_tool_node import VisionToolNode
from pokemon_agent.core import get_llm, get_settings
from pokemon_agent.emulator import get_screenshot_data_url

logger = logging.getLogger(__name__)


def build_initial_messages(emulator=None) -> list:
    """Build the seed conversation: system prompt + kickoff message.

    When an emulator is given, the kickoff includes the current screenshot and
    memory state so the model starts with eyes open — essential when resuming
    a saved game mid-playthrough.
    """
    if emulator is None:
        kickoff_content = "You may now begin playing."
    else:
        kickoff_content = [
            {
                "type": "text",
                "text": "You may now begin playing. Here is the current game "
                "screen and state:",
            },
            {
                "type": "image_url",
                "image_url": {"url": get_screenshot_data_url(emulator)},
            },
            {"type": "text", "text": emulator.get_state_from_memory()},
        ]
    return [
        SystemMessage(content=PromptManager.get_system_prompt("game_player")),
        HumanMessage(content=kickoff_content),
    ]


# =============================================================================
# Agent Nodes
# =============================================================================


def agent_node(state: GameAgentState, model_with_tools: BaseChatModel) -> dict:
    """Main agent reasoning node - calls the LLM to decide what to do."""
    start = time.monotonic()
    response = model_with_tools.invoke(state["messages"])
    logger.info(f"[Agent] LLM responded in {time.monotonic() - start:.1f}s")

    if response.text:
        logger.info(f"[Text] {response.text}")
    for tool_call in response.tool_calls or []:
        logger.info(f"[Tool] Using tool: {tool_call['name']}")

    # A step is an agent turn that takes an action
    step_count = state.get("step_count", 0)
    if response.tool_calls:
        step_count += 1

    return {"messages": [response], "step_count": step_count}


def nudge_node(state: GameAgentState) -> dict:
    """Tell the model its last reply did nothing because it called no tool.

    Keeps the conversation user/assistant-alternating and gives the model an
    explicit signal to act. The graph's recursion_limit is the only backstop
    against a model that never acts.
    """
    logger.warning("[Agent] Reply had no tool call, nudging...")
    return {
        "messages": [HumanMessage(content=PromptManager.get_human_prompt("nudge_no_tool"))],
    }


def summarize_node(state: GameAgentState, summarizer: BaseChatModel) -> dict:
    """Condense the conversation history into a summary to manage context size.

    Replaces the full history with the system prompt and a single handoff
    message carrying the summary plus a fresh screenshot.
    """
    logger.info("[Agent] Generating conversation summary...")

    messages = state["messages"]
    request = list(messages) + [
        HumanMessage(content=PromptManager.get_human_prompt("summarize_request"))
    ]
    response = summarizer.invoke(request)
    summary_text = response.text or ""

    logger.info("[Agent] Game Progress Summary:")
    logger.info(summary_text)

    handoff_text = PromptManager.get_human_prompt(
        "summary_handoff",
        num_messages=len(messages),
        summary=summary_text,
    )

    handoff = HumanMessage(
        content=[
            {"type": "text", "text": handoff_text},
            {"type": "text", "text": "\n\nCurrent game screenshot for reference:"},
            {
                "type": "image_url",
                "image_url": {"url": get_screenshot_data_url(state["emulator"])},
            },
            {
                "type": "text",
                "text": "You were just asked to summarize your playthrough so far, "
                "which is the summary you see above. You may now continue playing by "
                "selecting your next action.",
            },
        ]
    )

    logger.info("[Agent] Message history condensed into summary.")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(content=PromptManager.get_system_prompt("game_player")),
            handoff,
        ]
    }


# =============================================================================
# Routing Logic
# =============================================================================


def route_after_agent(state: GameAgentState) -> Literal["tools", "nudge"]:
    """Execute tool calls if the model made any, otherwise nudge it to act.

    Never ends the run — only reaching max_steps (checked after tools) does.
    """
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "tool_calls", None):
        return "tools"
    return "nudge"


# When the provider returns no token usage, fall back to a message-count trigger
FALLBACK_MAX_MESSAGES = 30


def _last_prompt_tokens(messages) -> int | None:
    """Input token count of the most recent LLM call, as reported by the provider.

    This is the true measured size of the conversation (images included) —
    more honest than counting messages or running a local tokenizer.
    """
    for message in reversed(messages):
        usage = getattr(message, "usage_metadata", None)
        if usage and usage.get("input_tokens"):
            return usage["input_tokens"]
    return None


def route_after_tools(state: GameAgentState) -> Literal["agent", "summarize", "__end__"]:
    """Decide whether to keep playing, summarize the history first, or stop."""
    settings = get_settings()

    if state.get("step_count", 0) >= state.get("max_steps", settings.max_steps):
        return END

    messages = state.get("messages", [])
    prompt_tokens = _last_prompt_tokens(messages)
    limit = state.get("max_history_tokens", settings.max_history_tokens)
    if prompt_tokens is not None:
        if prompt_tokens >= limit:
            logger.info(f"[Agent] History at {prompt_tokens} prompt tokens (limit {limit})")
            return "summarize"
    elif len(messages) >= FALLBACK_MAX_MESSAGES:
        return "summarize"
    return "agent"


# =============================================================================
# Graph Builder
# =============================================================================


def build_game_graph(
    model: BaseChatModel | None = None,
    summarizer: BaseChatModel | None = None,
):
    """Build the gameplay ReAct graph.

    Args:
        model: The gameplay LLM. Defaults to get_llm() with settings.game_model.
        summarizer: The summarization LLM (no tools). Defaults to
            settings.summarizer_model (falling back to the game model).

    Returns:
        Compiled StateGraph for gameplay.
    """
    settings = get_settings()
    model = model or get_llm()
    summarizer = summarizer or get_llm(settings.resolved_summarizer_model)

    tools = [press_buttons]
    if settings.use_navigator:
        tools.append(navigate_to)

    # tool_choice="any" forces the model to always call a tool, preventing
    # text-only turns from stalling the loop — but Anthropic models then emit
    # no reasoning text at all. With "auto" the model can narrate before
    # acting; the nudge node covers the occasional tool-less reply.
    tool_choice = "any" if settings.force_tool_use else "auto"
    model_with_tools = model.bind_tools(tools, tool_choice=tool_choice)

    workflow = StateGraph(GameAgentState)

    workflow.add_node("agent", lambda state: agent_node(state, model_with_tools))
    workflow.add_node("tools", VisionToolNode(tools))
    workflow.add_node("summarize", lambda state: summarize_node(state, summarizer))
    workflow.add_node("nudge", nudge_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "nudge": "nudge"},
    )
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "summarize": "summarize", END: END},
    )
    workflow.add_edge("summarize", "agent")
    workflow.add_edge("nudge", "agent")

    return workflow.compile()
