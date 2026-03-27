"""Scoring: run tests, collect metrics, save results."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentResult
from .config import RunConfig
from .docker_runner import DockerRunner
from .llm_client import LLMClient

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Structured evaluation output."""

    task_name: str = ""
    mode: str = ""
    model: str = ""
    timestamp: str = ""
    # Tests
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    test_pass_rate: float = 0.0
    test_details: list[dict] = field(default_factory=list)
    # Quality (end-to-end only)
    quality_metrics: dict | None = None
    # Plan evaluation (plan mode only)
    plan_scores: dict | None = None
    # Cost
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    iterations: int = 0
    # Agent
    stopped_reason: str = ""
    files_created: list[str] = field(default_factory=list)


class Scorer:
    """Runs tests inside the container and assembles an EvalResult."""

    def __init__(self, runner: DockerRunner, config: RunConfig,
                 llm_client: LLMClient | None = None) -> None:
        self.runner = runner
        self.config = config
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    def score(
        self,
        agent_result: AgentResult,
        llm_usage: dict[str, int],
        wall_time: float,
    ) -> EvalResult:
        result = EvalResult(
            task_name=self.config.task.task_name,
            mode=self.config.task.mode,
            model=self.config.llm.model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_tokens=llm_usage.get("prompt_tokens", 0),
            completion_tokens=llm_usage.get("completion_tokens", 0),
            total_tokens=llm_usage.get("prompt_tokens", 0) + llm_usage.get("completion_tokens", 0),
            wall_time_seconds=round(wall_time, 2),
            iterations=agent_result.iterations,
            stopped_reason=agent_result.stopped_reason,
            files_created=agent_result.files_written,
        )

        # Run tests (function mode only — end-to-end uses quality metrics)
        if self.config.task.mode == "function":
            total, passed, failed, details = self._run_tests()
            result.tests_total = total
            result.tests_passed = passed
            result.tests_failed = failed
            result.test_pass_rate = passed / total if total > 0 else 0.0
            result.test_details = details

        # Plan evaluation (plan mode only) — LLM-as-judge rubric scoring
        if self.config.task.mode == "plan" and self.llm_client:
            result.plan_scores = self._evaluate_plan()

        # Quality metrics for end-to-end (sole evaluation criterion)
        if self.config.task.mode == "end_to_end":
            result.quality_metrics = self._compute_quality_metrics()

        return result

    # ------------------------------------------------------------------
    def _run_tests(self) -> tuple[int, int, int, list[dict]]:
        """Run pytest and parse the summary."""
        if self.config.task.mode == "function" and self.config.task.target_function:
            module = self.config.task.target_function.split(".")[0]
            test_cmd = f"python -m pytest evaluation/tests/test_{module}.py -v --tb=short --no-header"
        else:
            test_cmd = "python -m pytest evaluation/tests/ -v --tb=short --no-header"

        output, _ = self.runner.exec(test_cmd)
        log.info("Test output:\n%s", output)

        # Parse per-test results (lines like "test_foo.py::TestBar::test_baz PASSED")
        details: list[dict] = []
        for m in re.finditer(r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)", output):
            details.append({"test": m.group(1), "status": m.group(2)})

        # Parse summary line: "N passed", "M failed"
        passed = 0
        failed = 0
        pm = re.search(r"(\d+)\s+passed", output)
        fm = re.search(r"(\d+)\s+failed", output)
        em = re.search(r"(\d+)\s+error", output)
        if pm:
            passed = int(pm.group(1))
        if fm:
            failed = int(fm.group(1))
        if em:
            failed += int(em.group(1))
        total = passed + failed

        # Fallback: count from per-test details if summary was truncated
        if total == 0 and details:
            passed = sum(1 for d in details if d["status"] == "PASSED")
            failed = sum(1 for d in details if d["status"] in ("FAILED", "ERROR"))
            total = passed + failed

        return total, passed, failed, details

    # ------------------------------------------------------------------
    def _compute_quality_metrics(self) -> dict | None:
        """Compare reconstruction output against ground truth.

        In end-to-end mode the evaluation/ directory is NOT copied into the
        sandbox, so we copy the ground truth file from the host task directory
        into the sandbox before running the comparison script.
        """
        import shutil as _shutil

        # Ensure the ground truth file is available in the sandbox
        gt_host = self.config.task.task_dir / "evaluation" / "reference_outputs" / "ground_truth.npy"
        if not gt_host.exists():
            log.warning("Ground truth file not found: %s", gt_host)
            return {"error": "ground_truth.npy not found in task directory"}

        # Copy ground truth into the sandbox workspace
        workspace = Path(self.runner.container) if hasattr(self.runner, 'container') else None
        if workspace and workspace.is_dir():
            gt_dst = workspace / "evaluation" / "reference_outputs" / "ground_truth.npy"
            gt_dst.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(gt_host, gt_dst)
        else:
            # Docker runner — use exec to copy
            pass

        snippet = """\
import numpy as np, json, sys, os
out_path = "output/reconstruction.npy"
gt_path = "evaluation/reference_outputs/ground_truth.npy"
if not os.path.exists(out_path):
    print(json.dumps({"error": "output/reconstruction.npy not found"}))
    sys.exit(0)
out = np.load(out_path)
gt = np.load(gt_path)
# Flux-normalize
out = out * (gt.sum() / (out.sum() + 1e-30))
nrmse = float(np.linalg.norm(out - gt) / (np.linalg.norm(gt) + 1e-30))
ncc = float(np.sum(out * gt) / (np.linalg.norm(out) * np.linalg.norm(gt) + 1e-30))
print(json.dumps({"nrmse": round(nrmse, 4), "ncc": round(ncc, 4)}))
"""
        output, rc = self.runner.exec(f"python -c '{snippet}'")
        try:
            return json.loads(output.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            log.warning("Could not parse quality metrics: %s", output)
            return None

    # ------------------------------------------------------------------
    def _evaluate_plan(self) -> dict | None:
        """Evaluate generated plan using ELO-inspired pairwise + rubric scoring.

        Compares the generated plan against the reference plan (golden standard)
        shipped with the task, following the inverse_planning_eval methodology:
        1. Pairwise comparison with position swapping (3 rounds)
        2. Rubric scoring (6 dimensions, weighted average)
        3. Combined score: 0.5 * pairwise_win_rate + 0.5 * rubric_normalized
        """
        from .plan_scorer import evaluate_plan
        from dataclasses import asdict as _asdict

        readme = (self.config.task.task_dir / "README.md").read_text()

        # Read the generated plan files from the container
        generated_approach = self.runner.read_file("plan/approach.md")
        generated_design = self.runner.read_file("plan/design.md")

        if not generated_approach or not generated_design:
            log.warning("Plan files not found in container")
            return {"error": "plan files not generated"}

        # Read reference (golden) plan from the task directory
        ref_approach_path = self.config.task.task_dir / "plan" / "approach.md"
        ref_design_path = self.config.task.task_dir / "plan" / "design.md"
        reference_approach = ref_approach_path.read_text() if ref_approach_path.exists() else ""
        reference_design = ref_design_path.read_text() if ref_design_path.exists() else ""

        log.info("Evaluating plan quality (pairwise + rubric)...")
        score = evaluate_plan(
            self.llm_client,
            readme,
            generated_approach,
            generated_design,
            reference_approach=reference_approach,
            reference_design=reference_design,
            n_pairwise_rounds=3,
        )
        return _asdict(score)

    # ------------------------------------------------------------------
    def save(self, result: EvalResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename (replace / with _)
        safe_model = result.model.replace("/", "_").replace("\\", "_")
        name = f"{result.task_name}_{result.mode}_{safe_model}_{ts}.json"
        path = output_dir / name
        path.write_text(json.dumps(asdict(result), indent=2))
        log.info("Results saved to %s", path)
        return path
