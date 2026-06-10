"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Thinking level for LLM reasoning control.
# "none" disables thinking; others map to OpenRouter's reasoning effort.
ThinkingLevel = Literal["none", "minimal", "low", "medium", "high"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def find_dotenv() -> Path | None:
    """Find .env file by walking up from current directory."""
    current = Path.cwd()
    while current != current.parent:
        env_file = current / ".env"
        if env_file.exists():
            return env_file
        current = current.parent
    return None


class Settings(BaseSettings):
    """Application settings.

    Loads configuration from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key",
    )

    # Model Configuration (OpenRouter model IDs, e.g. "google/gemini-3.1-flash-lite")
    game_model: str = Field(
        default="google/gemini-3-flash-preview",
        description="Model for gameplay decisions",
    )
    summarizer_model: str = Field(
        default="",
        description="Model for history summarization (empty = use game_model)",
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature for gameplay",
    )
    max_tokens: int = Field(
        default=4000,
        description="Maximum output tokens per LLM call",
    )
    thinking: ThinkingLevel = Field(
        default="low",
        description="Reasoning effort level (none disables thinking)",
    )
    max_requests_per_second: float = Field(
        default=0.0,
        description="Global LLM API rate limit (requests per second, 0 to disable)",
    )

    # Gameplay Settings
    rom_path: str = Field(
        default="Pokemon Red.gb",
        description="Path to the Pokemon Red ROM file",
    )
    max_steps: int = Field(
        default=10,
        description="Number of agent steps to run",
    )
    max_history: int = Field(
        default=30,
        description="Maximum number of messages in history before summarization",
    )
    use_navigator: bool = Field(
        default=False,
        description="Expose the navigate_to pathfinding tool to the agent",
    )
    force_tool_use: bool = Field(
        default=False,
        description=(
            "Force a tool call every turn (tool_choice=any). Guarantees action but "
            "suppresses visible reasoning text; the default (tool_choice=auto) lets "
            "the model narrate, with tool-less replies handled by the nudge node"
        ),
    )

    @property
    def api_key(self) -> str:
        """Get the OpenRouter API key."""
        return self.openrouter_api_key

    @property
    def base_url(self) -> str:
        """Get the OpenRouter base URL."""
        return OPENROUTER_BASE_URL

    @property
    def resolved_summarizer_model(self) -> str:
        """Summarizer model, falling back to the game model."""
        return self.summarizer_model or self.game_model


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance loaded from environment.
    """
    env_file = find_dotenv()
    if env_file:
        return Settings(_env_file=env_file)  # type: ignore
    return Settings()
