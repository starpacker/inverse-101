"""Scoring: run tests, collect metrics, save results."""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentResult
from .config import RunConfig
from .docker_runner import DockerRunner

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

    def __init__(self, runner: DockerRunner, config: RunConfig) -> None:
        self.runner = runner
        self.config = config

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

        # Run tests
        if self.config.task.mode != "plan":
            total, passed, failed, details = self._run_tests()
            result.tests_total = total
            result.tests_passed = passed
            result.tests_failed = failed
            result.test_pass_rate = passed / total if total > 0 else 0.0
            result.test_details = details

        # Quality metrics for end-to-end
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

        return total, passed, failed, details

    # ------------------------------------------------------------------
    def _compute_quality_metrics(self) -> dict | None:
        """Compare reconstruction output against ground truth."""
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
    def save(self, result: EvalResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{result.task_name}_{result.mode}_{result.model}_{ts}.json"
        path = output_dir / name
        path.write_text(json.dumps(asdict(result), indent=2))
        log.info("Results saved to %s", path)
        return path
