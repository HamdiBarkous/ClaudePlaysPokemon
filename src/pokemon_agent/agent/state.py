"""LangGraph state definitions."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GameAgentState(TypedDict, total=False):
    """State for the Pokemon gameplay ReAct agent."""

    # Conversation - use add_messages reducer to accumulate messages
    messages: Annotated[list[BaseMessage], add_messages]

    # Context (read-only, set at start) — live Emulator and TTS Speaker used by tools
    emulator: Any
    speaker: Any

    # Run parameters (set at start, may be overridden from the CLI)
    max_steps: int
    max_history_tokens: int

    # Tracking
    step_count: int
