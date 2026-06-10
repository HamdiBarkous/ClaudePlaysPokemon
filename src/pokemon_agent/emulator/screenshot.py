"""Screenshot helpers for converting emulator frames to LLM-ready data URLs."""

import base64
import io


def get_screenshot_base64(screenshot) -> str:
    """Convert PIL image to base64 string."""

    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


def image_to_data_url(image) -> str:
    """Convert a PIL image to a data URL for multimodal messages."""
    return f"data:image/png;base64,{get_screenshot_base64(image)}"


def get_screenshot_data_url(emulator) -> str:
    """Capture the current emulator frame as a data URL for multimodal messages."""
    return image_to_data_url(emulator.get_screenshot())
