"""Top-level benchmark runner: orchestrates agent, Docker, and scoring."""

import logging
import time
from pathlib import Path

from .agent import Agent, AgentResult
from .config import RunConfig
from .docker_runner import DockerRunner
from .llm_client import LLMClient
from .prompts import (
    end_to_end_impl_prompt,
    end_to_end_plan_prompt,
    function_prompt,
    plan_approach_prompt,
    plan_design_prompt,
)
from .scorer import EvalResult, Scorer

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs a single benchmark evaluation."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.client = LLMClient(config.llm)
        self.runner = DockerRunner(
            image=config.docker_image,
            task_dir=config.task.task_dir,
            timeout=config.timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Which files the agent is allowed to see per mode
    # (matches the Evaluation Modes table in CLAUDE.md)
    VISIBLE_PATHS = {
        "plan": ["README.md", "data"],
        "function": ["README.md", "plan", "evaluation"],
        "end_to_end": ["README.md", "data", "evaluation"],
    }

    # ------------------------------------------------------------------
    def run(self) -> EvalResult:
        """Execute the benchmark and return scored results."""
        self._validate()
        visible = self.VISIBLE_PATHS.get(self.config.task.mode)
        self.runner.start(visible_paths=visible)
        t0 = time.time()

        try:
            mode = self.config.task.mode
            if mode == "plan":
                agent_result = self._run_plan_mode()
            elif mode == "function":
                agent_result = self._run_function_mode()
            elif mode == "end_to_end":
                agent_result = self._run_end_to_end_mode()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            wall_time = time.time() - t0

            scorer = Scorer(self.runner, self.config)
            result = scorer.score(agent_result, self.client.total_usage, wall_time)
            scorer.save(result, self.config.output_dir)
            return result
        finally:
            self.runner.stop()

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------
    def _run_plan_mode(self) -> AgentResult:
        """Generate approach.md and design.md."""
        readme = self._read_host_file("README.md")
        meta_data = self._read_host_file("data/meta_data")

        agent = Agent(self.client, self.runner, self.config.max_iterations)

        # Phase 1: approach
        log.info("Plan mode — generating approach.md")
        prompt1 = plan_approach_prompt(readme, meta_data)
        result1 = agent.run(prompt1)

        # Phase 2: design (read the generated approach from container)
        approach = self.runner.read_file("plan/approach.md")
        log.info("Plan mode — generating design.md")
        prompt2 = plan_design_prompt(readme, approach)
        result2 = agent.run(prompt2)

        # Merge results
        result2.files_written = result1.files_written + result2.files_written
        result2.iterations = result1.iterations + result2.iterations
        return result2

    def _run_function_mode(self) -> AgentResult:
        """Implement a single target function."""
        target = self.config.task.target_function
        if not target:
            raise ValueError("function mode requires --target-function")

        readme = self._read_host_file("README.md")
        approach = self._read_host_file("plan/approach.md")
        design = self._read_host_file("plan/design.md")

        # Determine test file
        module = target.split(".")[0]
        test_path = f"evaluation/tests/test_{module}.py"
        test_content = self._read_host_file(test_path)

        prompt = function_prompt(readme, approach, design, target, test_content)
        agent = Agent(self.client, self.runner, self.config.max_iterations)
        return agent.run(prompt)

    def _run_end_to_end_mode(self) -> AgentResult:
        """Full pipeline: plan → implement → test."""
        readme = self._read_host_file("README.md")
        meta_data = self._read_host_file("data/meta_data")

        agent = Agent(self.client, self.runner, self.config.max_iterations)

        # Phase 1: planning
        log.info("End-to-end — Phase 1: planning")
        plan_prompt = end_to_end_plan_prompt(readme, meta_data)
        plan_result = agent.run(plan_prompt)

        # Phase 2: implementation
        approach = self.runner.read_file("plan/approach.md")
        design = self.runner.read_file("plan/design.md")
        log.info("End-to-end — Phase 2: implementation")
        impl_prompt = end_to_end_impl_prompt(approach, design)
        impl_result = agent.run(impl_prompt)

        # Merge
        impl_result.files_written = plan_result.files_written + impl_result.files_written
        impl_result.iterations = plan_result.iterations + impl_result.iterations
        return impl_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        task_dir = self.config.task.task_dir
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        if not (task_dir / "README.md").exists():
            raise FileNotFoundError(f"Missing README.md in {task_dir}")

    def _read_host_file(self, rel_path: str) -> str:
        """Read a file from the task directory on the host."""
        p = self.config.task.task_dir / rel_path
        if not p.exists():
            return f"[File not found: {rel_path}]"
        return p.read_text()
