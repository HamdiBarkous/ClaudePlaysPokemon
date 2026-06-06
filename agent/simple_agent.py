import base64
import copy
import io
import json
import logging
import os

from config import MAX_TOKENS, MODEL_NAME, OPENROUTER_BASE_URL, TEMPERATURE, USE_NAVIGATOR

from agent.emulator import Emulator
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


def screenshot_message(screenshot_b64, memory_info, intro):
    """Build an OpenAI-format user message carrying a screenshot and game state.

    OpenAI/OpenRouter tool messages (role "tool") cannot contain images, so the
    screenshot is delivered in a follow-up user message instead.
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": intro},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            },
            {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
        ],
    }


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the emulator tool to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "press_buttons",
            "description": "Press a sequence of buttons on the Game Boy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "buttons": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["a", "b", "start", "select", "up", "down", "left", "right"]
                        },
                        "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Whether to wait for a brief period after pressing each button. Defaults to true."
                    }
                },
                "required": ["buttons"],
            },
        },
    }
]

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append({
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "The row coordinate to navigate to (0-8)."
                    },
                    "col": {
                        "type": "integer",
                        "description": "The column coordinate to navigate to (0-9)."
                    }
                },
                "required": ["row", "col"],
            },
        },
    })


class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, load_state=None):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()  # Initialize the emulator
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://hamdibarkous.com",
                "X-Title": "Claude Plays Pokemon",
            },
        )
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        if load_state:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)

    def process_tool_call(self, tool_call):
        """Process a single tool call.

        Returns a dict describing the result so the caller can assemble the
        OpenAI-format `tool` message plus a follow-up user message with the
        resulting screenshot.
        """
        tool_name = tool_call.function.name
        try:
            tool_input = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            tool_input = {}
        logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")

            self.emulator.press_buttons(buttons, wait)
            result_text = f"Pressed buttons: {', '.join(buttons)}"
            intro = "\nHere is a screenshot of the screen after your button presses:"
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            logger.info(f"[Navigation] Navigating to: ({row}, {col})")

            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            result_text = f"Navigation result: {result}"
            intro = "\nHere is a screenshot of the screen after navigation:"
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "tool_call_id": tool_call.id,
                "result_text": f"Error: Unknown tool '{tool_name}'",
                "screenshot_b64": None,
                "memory_info": None,
                "intro": None,
            }

        # Get a fresh screenshot after executing the action
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

        # Get game state from memory after the action
        memory_info = self.emulator.get_state_from_memory()

        # Log the memory state after the tool call
        logger.info("[Memory State after action]")
        logger.info(memory_info)

        collision_map = self.emulator.get_collision_map()
        if collision_map:
            logger.info(f"[Collision Map after action]\n{collision_map}")

        return {
            "tool_call_id": tool_call.id,
            "result_text": result_text,
            "screenshot_b64": screenshot_b64,
            "memory_info": memory_info,
            "intro": intro,
        }

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.message_history

                # Get model response
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    temperature=TEMPERATURE,
                )

                logger.info(f"Response usage: {response.usage}")

                message = response.choices[0].message
                tool_calls = message.tool_calls or []

                # Display the model's reasoning
                if message.content:
                    logger.info(f"[Text] {message.content}")
                for tool_call in tool_calls:
                    logger.info(f"[Tool] Using tool: {tool_call.function.name}")

                # Process tool calls
                if tool_calls:
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                    self.message_history.append(assistant_message)

                    # Process tool calls: each produces a `tool` message (text only),
                    # then a follow-up user message carries the screenshot.
                    follow_up_messages = []
                    for tool_call in tool_calls:
                        result = self.process_tool_call(tool_call)
                        self.message_history.append({
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": result["result_text"],
                        })
                        if result["screenshot_b64"]:
                            follow_up_messages.append(
                                screenshot_message(
                                    result["screenshot_b64"],
                                    result["memory_info"],
                                    result["intro"],
                                )
                            )

                    # All `tool` messages must immediately follow the assistant
                    # message, so the screenshot user messages are appended last.
                    self.message_history.extend(follow_up_messages)

                    # Check if we need to summarize the history
                    if len(self.message_history) >= self.max_history:
                        self.summarize_history()

                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                raise e

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info("[Agent] Generating conversation summary...")

        # Get a new screenshot for the summary
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

        # Create messages for the summarization request - pass the entire conversation history
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + copy.deepcopy(self.message_history)
            + [{"role": "user", "content": SUMMARY_PROMPT}]
        )

        # Get summary from the model
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=messages,
            temperature=TEMPERATURE,
        )

        # Extract the summary text
        summary_text = response.choices[0].message.content or ""

        logger.info(f"[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")

        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]

        logger.info(f"[Agent] Message history condensed into summary.")

    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()
