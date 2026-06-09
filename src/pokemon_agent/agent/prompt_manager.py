from pathlib import Path
from typing import Literal

import frontmatter
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError, meta

MessageType = Literal["system", "human"]


class PromptManager:
    _envs: dict[str, Environment] = {}

    @classmethod
    def _get_env(cls, message_type: MessageType) -> Environment:
        if message_type not in cls._envs:
            current_file = Path(__file__).parent
            templates_dir = current_file.parent.parent.parent / "prompts" / message_type
            cls._envs[message_type] = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                undefined=StrictUndefined,
            )
        return cls._envs[message_type]

    @staticmethod
    def get_prompt(template: str, message_type: MessageType = "system", **kwargs) -> str:
        """Get a rendered prompt template.

        Args:
            template: Template name (without .j2 extension)
            message_type: "system" or "human" - determines which folder to load from
            **kwargs: Variables to pass to the template

        Returns:
            Rendered prompt string
        """
        env = PromptManager._get_env(message_type)
        template_path = f"{template}.j2"
        with open(env.loader.get_source(env, template_path)[1]) as file:  # type: ignore
            post = frontmatter.load(file)
        jinja_template = env.from_string(post.content)
        try:
            return jinja_template.render(**kwargs)
        except TemplateError as e:
            raise ValueError(f"Error rendering template: {str(e)}") from e

    @staticmethod
    def get_system_prompt(template: str, **kwargs) -> str:
        """Get a rendered system prompt template."""
        return PromptManager.get_prompt(template, message_type="system", **kwargs)

    @staticmethod
    def get_human_prompt(template: str, **kwargs) -> str:
        """Get a rendered human prompt template."""
        return PromptManager.get_prompt(template, message_type="human", **kwargs)

    @staticmethod
    def get_template_info(template: str, message_type: MessageType = "system") -> dict:
        """Get metadata about a template."""
        env = PromptManager._get_env(message_type)
        template_path = f"{template}.j2"
        with open(env.loader.get_source(env, template_path)[1]) as file:  # type: ignore
            post = frontmatter.load(file)
        ast = env.parse(post.content)
        variables = meta.find_undeclared_variables(ast)
        return {
            "name": template,
            "message_type": message_type,
            "description": post.metadata.get("description", ""),
            "variables": variables,
            "frontmatter": post.metadata,
        }
