# Pokemon Agent

An LLM agent that plays Pokemon Red through the PyBoy emulator, built with LangGraph on top of OpenRouter (OpenAI-compatible). Features:

- ReAct-style LangGraph agent that sees the game screen and presses buttons via tool calls
- Memory reading functionality to extract game state information (party, inventory, dialog, collision map)
- Automatic history summarization to manage context size
- Jinja2 prompt templates and pydantic-settings configuration

## Setup

1. Clone this repository
2. Install the required packages with [uv](https://docs.astral.sh/uv/):
   ```
   uv sync
   ```
3. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. Place your Pokemon Red ROM file in the root directory (you need to provide your own ROM)

## Usage

Run the main script:

```
uv run main.py --rom "Pokemon Red.gb"
```

Optional arguments:
- `--rom`: Path to the Pokemon ROM file (default: `Pokemon Red.gb` in the root directory)
- `--steps`: Number of agent steps to run (default: 10)
- `--display`: Run with display (not headless)
- `--sound`: Enable sound (only applicable with display)
- `--max-history`: Messages in history before summarization (default: 30)
- `--state`: Save-state path (default: `game.state`) — the game auto-saves here on exit and auto-resumes from it on launch
- `--new-game`: Start a fresh game instead of resuming the saved state

Example:
```
uv run main.py --rom "Pokemon Red.gb" --steps 20 --display --sound
```

## Configuration

Settings load from environment variables / `.env` (see `src/pokemon_agent/core/settings.py`), e.g.:

```
OPENROUTER_API_KEY=...
GAME_MODEL=google/gemini-3.1-flash-lite
SUMMARIZER_MODEL=          # empty = use GAME_MODEL
TEMPERATURE=1.0
THINKING=none              # none|minimal|low|medium|high (OpenRouter reasoning effort)
USE_NAVIGATOR=false        # expose the navigate_to pathfinding tool
```

## Implementation Details

### Structure

```
prompts/
├── system/game_player.j2        # gameplay system prompt
└── human/                       # summarization prompts
src/pokemon_agent/
├── core/
│   ├── settings.py              # pydantic-settings configuration
│   └── llm.py                   # OpenRouter LLM factory (retry, reasoning control)
├── agent/
│   ├── graph.py                 # LangGraph ReAct loop (agent → tools → summarize)
│   ├── state.py                 # graph state
│   ├── prompt_manager.py        # Jinja2 + frontmatter prompt loading
│   ├── vision_tool_node.py      # ToolNode that emits multimodal tool results
│   └── tools/                   # press_buttons, navigate_to
└── emulator/
    ├── emulator.py              # PyBoy wrapper (buttons, collision map, A* pathfinding)
    └── memory_reader.py         # game state extraction from emulator memory
```

### How It Works

1. The agent (LLM) decides on an action and calls the `press_buttons` tool
2. The tool executes the button presses on the emulator
3. The tool result carries a screenshot, memory-derived game state, and a collision map back to the model in a single multimodal tool message
4. When the conversation grows past `--max-history` messages, a summarize node condenses it and play continues
5. The loop ends after `--steps` agent turns
