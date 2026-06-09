"""Vision-aware ToolNode that properly handles image outputs for multimodal models.

This module provides a custom ToolNode that can process tool outputs containing
images and format them so vision-capable LLMs can actually "see" the images.

The key insight is that ToolMessage.content can be a list of content blocks,
including {"type": "image_url", "image_url": {"url": "data:image/..."}} blocks.
LangChain's TOOL_MESSAGE_BLOCK_TYPES includes "image_url", so this is supported.

However, the standard ToolNode doesn't automatically convert image data in tool
outputs to this format. VisionToolNode adds this capability by:
1. Detecting image data in tool outputs (base64 data URLs or specific keys)
2. Converting them to proper image_url content blocks
3. Combining text and image content in the ToolMessage
"""

from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode


class VisionToolNode(ToolNode):
    """A ToolNode that properly formats image outputs for vision models.

    When a tool returns a dict containing an image (detected by keys like
    'screenshot', 'image', etc.), this node will:
    1. Extract the image data (expected to be a base64 data URL)
    2. Format the non-image parts as JSON text
    3. Create a multimodal ToolMessage with both text and image content blocks

    This allows vision-capable LLMs to actually "see" images returned by tools,
    enabling true visual reasoning in ReAct-style agents.

    Example tool output that will be processed:
        {
            "result": "Pressed buttons: a",
            "screenshot": "data:image/png;base64,..."
        }

    Will become a ToolMessage with content:
        [
            {"type": "text", "text": '{"result": "Pressed buttons: a"}'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    """

    # Keys that indicate image data in tool outputs
    IMAGE_KEYS = {
        "image", "screenshot", "image_url", "preview_image",
    }

    def _format_output_with_images(self, output: Any) -> str | list[dict]:
        """Convert tool output to ToolMessage content, handling images specially.

        Args:
            output: The raw tool output (typically a dict).

        Returns:
            Either a string (for simple outputs) or a list of content blocks
            (for outputs containing images).
        """
        # If not a dict, use default behavior
        if not isinstance(output, dict):
            return self._default_format(output)

        # Check for image keys
        image_data = {}
        non_image_data = {}

        for key, value in output.items():
            if key in self.IMAGE_KEYS and isinstance(value, str) and value.startswith("data:image"):
                image_data[key] = value
            elif key in self.IMAGE_KEYS and isinstance(value, list):
                # Handle lists of images
                labels = []
                for j, img in enumerate(value):
                    if isinstance(img, str) and img.startswith("data:image"):
                        image_data[f"{key}_{j}"] = img
                        labels.append(f"[image {j + 1} below]")
                # Keep placeholder labels in text so the LLM knows ordering
                if labels:
                    non_image_data[key] = labels
            else:
                non_image_data[key] = value

        # If no images found, use default behavior
        if not image_data:
            return self._default_format(output)

        # Build multimodal content blocks
        content_blocks: list[dict] = []

        # Add text content (non-image data as JSON)
        if non_image_data:
            import json
            text_content = json.dumps(non_image_data, indent=2)
            content_blocks.append({"type": "text", "text": text_content})

        # Add image content blocks
        for key, image_url in image_data.items():
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        return content_blocks

    def _default_format(self, output: Any) -> str:
        """Default formatting for non-image outputs."""
        if isinstance(output, str):
            return output
        try:
            import json
            return json.dumps(output, indent=2)
        except (TypeError, ValueError):
            return str(output)

    def _run_one(self, *args, **kwargs) -> ToolMessage:
        """Override to post-process tool output for images."""
        result = super()._run_one(*args, **kwargs)
        return self._postprocess(result)

    async def _arun_one(self, *args, **kwargs) -> ToolMessage:
        """Override async version to post-process tool output for images."""
        result = await super()._arun_one(*args, **kwargs)
        return self._postprocess(result)

    def _postprocess(self, result: Any) -> Any:
        """Re-format a ToolMessage whose content is a JSON dict with images."""
        if isinstance(result, ToolMessage) and isinstance(result.content, str):
            try:
                import json
                parsed = json.loads(result.content)
                if isinstance(parsed, dict):
                    new_content = self._format_output_with_images(parsed)
                    if new_content != result.content:
                        result.content = new_content
            except (json.JSONDecodeError, ValueError):
                pass
        return result
