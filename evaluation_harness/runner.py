"""Top-level benchmark runner: orchestrates agent, Docker/local, and scoring."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from .agent import Agent, AgentResult
from .config import RunConfig
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


def _docker_available() -> bool:
    """Check if Docker is available on this system."""
    return shutil.which("docker") is not None


class BenchmarkRunner:
    """Runs a single benchmark evaluation."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.client = LLMClient(config.llm)

        if _docker_available():
            from .docker_runner import DockerRunner
            self.runner = DockerRunner(
                image=config.docker_image,
                task_dir=config.task.task_dir,
                timeout=config.timeout_seconds,
            )
        else:
            log.warning("Docker not available — using local runner")
            from .local_runner import LocalRunner
            self.runner = LocalRunner(
                image=config.docker_image,
                task_dir=config.task.task_dir,
                timeout=config.timeout_seconds,
            )

    # ------------------------------------------------------------------
    # Which files the agent is allowed to see per mode
    # (matches the Evaluation Modes table in CLAUDE.md)
    VISIBLE_PATHS = {
        "plan": ["README.md", "data"],
        "function": ["README.md", "plan", "evaluation", "data"],
        "end_to_end": ["README.md", "data"],
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

            scorer = Scorer(self.runner, self.config, llm_client=self.client)
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

        agent = Agent(self.client, self.runner, self.config.max_iterations,
                      mode="plan", log_file=self.config.log_file)

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

        # --- FIX: Copy reference implementations of non-target modules ---
        # Some test files have cross-module dependencies (e.g. test_physics_model
        # imports from src.preprocessing). In function mode the agent only
        # implements the target module, so we seed the workspace with the
        # reference implementations for all *other* modules so that cross-module
        # imports succeed.  The target module itself is NOT copied — the agent
        # must create it from scratch.
        self._seed_dependency_modules(module)

        prompt = function_prompt(readme, approach, design, target, test_content)
        agent = Agent(self.client, self.runner, self.config.max_iterations,
                      mode="function", log_file=self.config.log_file)
        return agent.run(prompt)

    def _seed_dependency_modules(self, target_module: str) -> None:
        """Copy reference src/ modules (except the target) into the sandbox.

        This ensures that test files with cross-module imports (e.g.
        test_physics_model.py importing src.preprocessing) can run without
        error, isolating failures to the target module's implementation.
        """
        src_dir = self.config.task.task_dir / "src"
        if not src_dir.is_dir():
            log.warning("No src/ directory found in task — skipping dependency seeding")
            return

        # Ensure src/__init__.py exists in the sandbox
        self.runner.write_file("src/__init__.py", "")

        for py_file in sorted(src_dir.glob("*.py")):
            mod_name = py_file.stem  # e.g. "preprocessing"
            if mod_name == "__init__":
                continue
            if mod_name == target_module:
                # Skip the target module — the agent must implement it
                log.info("Skipping target module src/%s.py (agent must implement)", mod_name)
                continue
            content = py_file.read_text()
            self.runner.write_file(f"src/{py_file.name}", content)
            log.info("Seeded dependency: src/%s.py", mod_name)

    def _run_end_to_end_mode(self) -> AgentResult:
        """Full pipeline: plan → implement → run.

        In end-to-end mode the agent is NOT given any test cases or evaluation
        code.  It must freely design and implement the full reconstruction
        pipeline, then run ``main.py`` to produce ``output/reconstruction.npy``.
        Scoring is based solely on reconstruction quality (NRMSE / NCC) against
        the reference ground truth — no unit-test pass rate is computed.
        """
        readme = self._read_host_file("README.md")
        meta_data = self._read_host_file("data/meta_data")

        # Phase 1: planning — use "plan" mode so DONE is accepted after
        # writing plan files (no main.py gating needed in the planning phase)
        plan_agent = Agent(self.client, self.runner, self.config.max_iterations,
                           mode="plan", log_file=self.config.log_file)
        log.info("End-to-end — Phase 1: planning")
        plan_prompt = end_to_end_plan_prompt(readme, meta_data)
        plan_result = plan_agent.run(plan_prompt)

        # Phase 2: implementation — use "end_to_end" mode so DONE is gated
        # on main.py having been executed
        impl_agent = Agent(self.client, self.runner, self.config.max_iterations,
                           mode="end_to_end", log_file=self.config.log_file)
        approach = self.runner.read_file("plan/approach.md")
        design = self.runner.read_file("plan/design.md")
        log.info("End-to-end — Phase 2: implementation")
        impl_prompt = end_to_end_impl_prompt(approach, design)
        impl_result = impl_agent.run(impl_prompt)

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
