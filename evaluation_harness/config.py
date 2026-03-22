"""Configuration dataclasses for the evaluation harness."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """OpenAI-compatible LLM endpoint configuration."""

    model: str  # e.g. "gpt-4o", "deepseek-chat"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 16384


@dataclass
class TaskConfig:
    """Which task to evaluate and in what mode."""

    task_name: str  # e.g. "eht_black_hole"
    task_dir: Path = field(default_factory=Path)  # resolved at runtime
    mode: str = "end_to_end"  # "plan" | "function" | "end_to_end"
    target_function: str | None = None  # for function mode, e.g. "preprocessing.load_observation"


@dataclass
class RunConfig:
    """Top-level run configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    max_iterations: int = 20
    docker_image: str = "imaging101-sandbox"
    timeout_seconds: int = 600
    output_dir: Path = field(default_factory=lambda: Path("results"))
