"""Screenshot helpers for converting emulator frames to LLM-ready data URLs."""

import base64
import io


def get_screenshot_base64(screenshot, upscale: int = 1) -> str:
    """Convert PIL image to base64 string."""
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


def get_screenshot_data_url(emulator, upscale: int = 2) -> str:
    """Capture the current emulator frame as a data URL for multimodal messages."""
    screenshot = emulator.get_screenshot()
    return f"data:image/png;base64,{get_screenshot_base64(screenshot, upscale=upscale)}"
