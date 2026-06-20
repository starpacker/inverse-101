"""Scoring: run tests, collect metrics, save results."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .agent import AgentResult
from .config import RunConfig
from .docker_runner import DockerRunner
from .llm_client import LLMClient

log = logging.getLogger(__name__)

REFERENCE_ARRAY_KEYS = (
    "ground_truth",
    "gt",
    "truth",
    "target",
    "reference",
    "reference_reconstruction",
    "reconstruction",
    "phantom",
    "image",
    "volume",
    "delta_n",
    "tissue_map",
    "bone_map",
    "y_pred",
    "spectrum",
)


def _reference_path_candidates(task_dir: Path) -> list[Path]:
    """Return reference array candidates in conservative priority order."""
    ref_dir = task_dir / "evaluation" / "reference_outputs"
    data_dir = task_dir / "data"
    ordered = [
        ref_dir / "ground_truth.npy",
        ref_dir / "ground_truth.npz",
        data_dir / "ground_truth.npy",
        data_dir / "ground_truth.npz",
        ref_dir / "reference.npy",
        ref_dir / "reference.npz",
        ref_dir / "reference_reconstruction.npy",
        ref_dir / "reference_reconstruction.npz",
        ref_dir / "reconstruction.npy",
        ref_dir / "reconstruction.npz",
        ref_dir / "reconstructions.npy",
        ref_dir / "reconstructions.npz",
        data_dir / "reference.npy",
        data_dir / "reference.npz",
        data_dir / "baseline_reference.npy",
        data_dir / "baseline_reference.npz",
    ]

    if ref_dir.is_dir():
        tokens = ("ground_truth", "reference", "reconstruction", "recon", "gt")
        dynamic = [
            path
            for path in sorted(ref_dir.iterdir())
            if path.suffix.lower() in (".npy", ".npz")
            and any(token in path.stem.lower() for token in tokens)
        ]
        ordered.extend(dynamic)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in ordered:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _prepare_metric_array(array: np.ndarray, *, label: str) -> np.ndarray:
    """Convert an array to the numeric form used for reconstruction metrics."""
    arr = np.asarray(array)
    if arr.dtype == object:
        raise ValueError(f"{label} is not a valid numeric array")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{label} is not numeric")
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError(f"{label} has wrong dimensions: 0 (expected at least 1)")
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float64)


def _key_rank(key: str) -> tuple[int, int, str]:
    lowered = key.lower()
    for index, preferred in enumerate(REFERENCE_ARRAY_KEYS):
        if lowered == preferred:
            return (0, index, lowered)
    for index, preferred in enumerate(REFERENCE_ARRAY_KEYS):
        if preferred in lowered:
            return (1, index, lowered)
    return (2, len(REFERENCE_ARRAY_KEYS), lowered)


def _load_array_file(path: Path, target_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Load a numeric metric array from .npy or .npz."""
    loaded = np.load(str(path), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            candidates: list[tuple[tuple[int, int, str], str, np.ndarray]] = []
            errors: list[str] = []
            for key in loaded.files:
                try:
                    arr = _prepare_metric_array(loaded[key], label=f"{path.name}:{key}")
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if target_shape is not None and arr.shape != target_shape:
                    continue
                candidates.append((_key_rank(key), key, arr))

            if not candidates and target_shape is not None:
                for key in loaded.files:
                    try:
                        arr = _prepare_metric_array(loaded[key], label=f"{path.name}:{key}")
                    except ValueError:
                        continue
                    candidates.append((_key_rank(key), key, arr))
                if len(candidates) == 1:
                    return candidates[0][2]
                shapes = {key: list(arr.shape) for _rank, key, arr in candidates}
                raise ValueError(
                    f"{path.name} has no array matching output shape {target_shape}; "
                    f"available shapes: {shapes}"
                )

            if not candidates:
                detail = "; ".join(errors[:3]) if errors else "no arrays found"
                raise ValueError(f"{path.name} does not contain a usable 2-D numeric array ({detail})")

            candidates.sort(key=lambda item: item[0])
            return candidates[0][2]
        finally:
            loaded.close()

    return _prepare_metric_array(loaded, label=path.name)


def load_reference_array(
    task_dir: Path,
    target_shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, Path]:
    """Find and load the reference array for end-to-end image metrics."""
    candidates = _reference_path_candidates(task_dir)
    existing = [path for path in candidates if path.exists()]
    errors: list[str] = []
    for path in existing:
        try:
            return _load_array_file(path, target_shape=target_shape), path
        except Exception as exc:
            errors.append(f"{path.relative_to(task_dir)}: {exc}")

    candidate_text = ", ".join(str(path.relative_to(task_dir)) for path in candidates[:10])
    if errors:
        raise ValueError(
            "reference array not found or not usable; "
            f"tried {candidate_text}; load errors: {' | '.join(errors[:4])}"
        )
    raise FileNotFoundError(f"reference array not found; tried {candidate_text}")


def compute_quality_metrics_from_arrays(out: np.ndarray, gt: np.ndarray) -> dict:
    """Compute end-to-end reconstruction metrics from prepared arrays."""
    if out.shape != gt.shape:
        return {
            "error": f"Shape mismatch: reconstruction {out.shape} vs expected {gt.shape}",
            "expected_shape": list(gt.shape),
        }

    out = out * (gt.sum() / (out.sum() + 1e-30))
    nrmse = float(np.linalg.norm(out - gt) / (np.linalg.norm(gt) + 1e-30))
    ncc = float(np.sum(out * gt) / (np.linalg.norm(out) * np.linalg.norm(gt) + 1e-30))
    mse = float(np.mean((out - gt) ** 2))
    max_val = float(gt.max())
    psnr = float(20 * np.log10(max_val / np.sqrt(mse))) if mse > 0 else float("inf")

    def _ssim_2d(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        mu_a, mu_b = a.mean(), b.mean()
        sig_a2, sig_b2 = a.var(), b.var()
        sig_ab = np.mean((a - mu_a) * (b - mu_b))
        num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
        den = (mu_a**2 + mu_b**2 + c1) * (sig_a2 + sig_b2 + c2)
        return float(num / den)

    ssim = _ssim_2d(out, gt, data_range=max_val)
    return {
        "nrmse": round(nrmse, 6),
        "ncc": round(ncc, 6),
        "mse": round(mse, 10),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 6),
    }


def _quality_metric_script(reference_path: str) -> str:
    """Build the sandbox-side metric script."""
    script = r'''
import json
import os
import sys

import numpy as np

out_path = "output/reconstruction.npy"
gt_path = "__REFERENCE_PATH__"
REFERENCE_ARRAY_KEYS = (
    "ground_truth", "truth", "target", "reference", "reference_reconstruction",
    "reconstruction", "phantom", "image", "volume", "delta_n", "tissue_map",
    "bone_map", "y_pred", "spectrum",
)


def _prepare_metric_array(array, label):
    arr = np.asarray(array)
    if arr.dtype == object:
        raise ValueError(f"{label} is not a valid numeric array")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{label} is not numeric")
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError(f"{label} has wrong dimensions: 0 (expected at least 1)")
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float64)


def _key_rank(key):
    lowered = key.lower()
    for index, preferred in enumerate(REFERENCE_ARRAY_KEYS):
        if lowered == preferred:
            return (0, index, lowered)
    for index, preferred in enumerate(REFERENCE_ARRAY_KEYS):
        if preferred in lowered:
            return (1, index, lowered)
    return (2, len(REFERENCE_ARRAY_KEYS), lowered)


def _load_array_file(path, target_shape=None):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            candidates = []
            errors = []
            for key in loaded.files:
                try:
                    arr = _prepare_metric_array(loaded[key], f"{os.path.basename(path)}:{key}")
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if target_shape is not None and arr.shape != target_shape:
                    continue
                candidates.append((_key_rank(key), key, arr))
            if not candidates:
                detail = "; ".join(errors[:3]) if errors else "no matching arrays found"
                raise ValueError(
                    f"{os.path.basename(path)} has no usable array for shape {target_shape}: {detail}"
                )
            candidates.sort(key=lambda item: item[0])
            return candidates[0][2]
        finally:
            loaded.close()
    return _prepare_metric_array(loaded, os.path.basename(path))


def _ssim_2d(a, b, data_range):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a2, sig_b2 = a.var(), b.var()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a2 + sig_b2 + c2)
    return float(num / den)


if not os.path.exists(out_path):
    print(json.dumps({"error": "output/reconstruction.npy not found"}))
    sys.exit(0)

try:
    out = _prepare_metric_array(np.load(out_path, allow_pickle=True), "reconstruction")
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    sys.exit(0)

try:
    gt = _load_array_file(gt_path, target_shape=out.shape)
except Exception as exc:
    print(json.dumps({"error": f"Failed to load reference array: {exc}"}))
    sys.exit(0)

if out.shape != gt.shape:
    print(json.dumps({
        "error": f"Shape mismatch: reconstruction {out.shape} vs expected {gt.shape}",
        "expected_shape": list(gt.shape),
    }))
    sys.exit(0)

out = out * (gt.sum() / (out.sum() + 1e-30))
nrmse = float(np.linalg.norm(out - gt) / (np.linalg.norm(gt) + 1e-30))
ncc = float(np.sum(out * gt) / (np.linalg.norm(out) * np.linalg.norm(gt) + 1e-30))
mse = float(np.mean((out - gt) ** 2))
max_val = float(gt.max())
psnr = float(20 * np.log10(max_val / np.sqrt(mse))) if mse > 0 else float("inf")
ssim = _ssim_2d(out, gt, data_range=max_val)
print(json.dumps({
    "nrmse": round(nrmse, 4),
    "ncc": round(ncc, 4),
    "mse": round(mse, 8),
    "psnr": round(psnr, 2),
    "ssim": round(ssim, 4),
}))
'''
    return script.replace("__REFERENCE_PATH__", reference_path)


@dataclass
class EvalResult:
    """Structured evaluation output."""

    task_name: str = ""
    mode: str = ""
    model: str = ""
    framework: str = "react"  # "react" | "multi_agent"
    level: str = "L1"  # end-to-end difficulty: "L1" | "L2" | "L3"
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
    llm_calls: int = 0  # Total individual LLM API calls (ReAct: =iterations, Multi-Agent: >>iterations)
    # Agent
    stopped_reason: str = ""
    files_created: list[str] = field(default_factory=list)
    # Visualization (end-to-end only)
    visualization_paths: dict[str, str] = field(default_factory=dict)


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
        llm_calls: int = 0,
    ) -> EvalResult:
        result = EvalResult(
            task_name=self.config.task.task_name,
            mode=self.config.task.mode,
            model=self.config.llm.model,
            framework=self.config.framework,
            level=self.config.task.level,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_tokens=llm_usage.get("prompt_tokens", 0),
            completion_tokens=llm_usage.get("completion_tokens", 0),
            total_tokens=llm_usage.get("prompt_tokens", 0) + llm_usage.get("completion_tokens", 0),
            wall_time_seconds=round(wall_time, 2),
            iterations=agent_result.iterations,
            llm_calls=llm_calls,
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
            # Visualization is handled by a dedicated downstream agent,
            # not the end-to-end evaluation pipeline.
            # result.visualization_paths = self._generate_visualizations(
            #     result.quality_metrics, result
            # )

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
        """Compare reconstruction output against the task reference array.

        In end-to-end mode the evaluation/ directory is NOT copied into the
        sandbox, so we copy the reference file from the host task directory
        into the sandbox before running the comparison script.
        """
        import shutil as _shutil

        ref_candidates = _reference_path_candidates(self.config.task.task_dir)
        ref_host = next((candidate for candidate in ref_candidates if candidate.exists()), None)
        if ref_host is None:
            tried = ", ".join(
                str(path.relative_to(self.config.task.task_dir))
                for path in ref_candidates[:10]
            )
            log.warning("Reference array not found in any location: %s", tried)
            return {"error": f"reference array not found; tried {tried}"}

        # Copy reference into the sandbox workspace
        ref_rel = f"evaluation/reference_outputs/_scoring_reference{ref_host.suffix.lower()}"
        workspace = Path(self.runner.container) if hasattr(self.runner, 'container') else None
        if workspace and workspace.is_dir():
            ref_dst = workspace / ref_rel
            ref_dst.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(ref_host, ref_dst)
        else:
            # Docker runner — use exec to copy
            source_rel = ref_host.relative_to(self.config.task.task_dir).as_posix()
            self.runner.exec(
                f"mkdir -p evaluation/reference_outputs && "
                f"cp /workspace_src/{source_rel} {ref_rel}"
            )

        snippet = _quality_metric_script(ref_rel)
        self.runner.write_file("_score_quality.py", snippet)
        output, rc = self.runner.exec("python _score_quality.py")
        try:
            return json.loads(output.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            log.warning("Could not parse quality metrics: %s", output)
            return None

    # ------------------------------------------------------------------
    def _generate_visualizations(
        self, quality_metrics: dict | None, result: EvalResult
    ) -> dict[str, str]:
        """Generate evaluation figures and persist reconstruction.npy.

        Copies the agent's ``output/reconstruction.npy`` from the sandbox
        into the results directory (so it survives sandbox cleanup), then
        calls the visualizer to produce comparison figures.

        Returns dict mapping figure name → absolute path string.
        """
        import shutil as _shutil

        from .visualizer import generate_eval_figures

        if quality_metrics is None or "error" in quality_metrics:
            return {}

        # --- Determine output directory for this run's artifacts ---
        safe_model = result.model.replace("/", "_").replace("\\", "_")
        run_id = f"{result.task_name}_{result.framework}_{safe_model}"
        fig_dir = self.config.output_dir / "figures" / run_id
        fig_dir.mkdir(parents=True, exist_ok=True)

        # --- Copy reconstruction.npy from sandbox to results ---
        workspace = Path(self.runner.container) if hasattr(self.runner, "container") else None
        recon_src = workspace / "output" / "reconstruction.npy" if workspace else None
        recon_dst = fig_dir / "reconstruction.npy"

        if recon_src and recon_src.exists():
            _shutil.copy2(recon_src, recon_dst)
            log.info("Saved reconstruction to %s", recon_dst)
        else:
            log.warning("reconstruction.npy not found in sandbox — cannot generate figures")
            return {}

        try:
            recon = _prepare_metric_array(
                np.load(str(recon_dst), allow_pickle=True),
                label="reconstruction",
            )
            gt, gt_path = load_reference_array(self.config.task.task_dir, target_shape=recon.shape)
            if recon.ndim != 2 or gt.ndim != 2:
                log.warning("visualization requires 2D arrays")
                return {}
        except Exception as e:
            log.warning("Failed to load arrays for visualization: %s", e)
            return {}

        # Also save ground truth copy for reference
        gt_dst = fig_dir / f"ground_truth{gt_path.suffix.lower()}"
        if not gt_dst.exists():
            _shutil.copy2(gt_path, gt_dst)

        # --- Generate figures ---
        run_label = f"{result.framework}_{safe_model}"
        paths = generate_eval_figures(
            reconstruction=recon,
            ground_truth=gt,
            metrics=quality_metrics,
            output_dir=fig_dir,
            run_label=run_label,
            task_name=result.task_name,
        )

        return paths

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

        readme = (self.config.task.task_dir / "README.md").read_text(encoding="utf-8")

        # Read the generated plan files from the container
        generated_approach = self.runner.read_file("plan/approach.md")
        generated_design = self.runner.read_file("plan/design.md")

        if not generated_approach or not generated_design:
            log.warning("Plan files not found in container")
            return {"error": "plan files not generated"}

        # Read reference (golden) plan from the task directory
        ref_approach_path = self.config.task.task_dir / "plan" / "approach.md"
        ref_design_path = self.config.task.task_dir / "plan" / "design.md"
        reference_approach = ref_approach_path.read_text(encoding="utf-8") if ref_approach_path.exists() else ""
        reference_design = ref_design_path.read_text(encoding="utf-8") if ref_design_path.exists() else ""

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
        """Save evaluation result.

        For function mode, saves into a structured directory:
            results/function_mode/{task}/{model_date}/{module}/result.json
            results/function_mode/{task}/{model_date}/{module}/src/{module}.py
        For plan mode, also saves a comparison markdown file listing
            model output vs ground-truth plan side by side.
        For other modes, saves flat JSON files in output_dir.
        """
        if result.mode == "function" and self.config.task.target_function:
            return self._save_function_mode(result, output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename (replace / with _)
        safe_model = result.model.replace("/", "_").replace("\\", "_")
        # Include level in filename for end-to-end mode
        level_suffix = f"_{result.level}" if result.mode == "end_to_end" else ""
        name = f"{result.task_name}_{result.mode}{level_suffix}_{result.framework}_{safe_model}_{ts}.json"
        path = output_dir / name
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        log.info("Results saved to %s", path)

        # For plan mode, also save a comparison document
        if result.mode == "plan":
            self._save_plan_comparison(output_dir, safe_model, ts)

        return path

    # ------------------------------------------------------------------
    def _save_plan_comparison(self, output_dir: Path, safe_model: str, ts: str) -> None:
        """Save a markdown comparison file: model output vs ground-truth plan."""
        task_name = self.config.task.task_name

        # Read model-generated plan from container
        generated_approach = self.runner.read_file("plan/approach.md") or "(not generated)"
        generated_design = self.runner.read_file("plan/design.md") or "(not generated)"

        # Read reference (ground-truth) plan from host task directory
        ref_approach_path = self.config.task.task_dir / "plan" / "approach.md"
        ref_design_path = self.config.task.task_dir / "plan" / "design.md"
        reference_approach = ref_approach_path.read_text(encoding="utf-8") if ref_approach_path.exists() else "(no reference)"
        reference_design = ref_design_path.read_text(encoding="utf-8") if ref_design_path.exists() else "(no reference)"

        # Build comparison markdown
        comparison = f"""# Plan Comparison: {task_name}

**Model**: {self.config.llm.model}
**Framework**: {self.config.framework}
**Timestamp**: {ts}

---

## approach.md

### 🤖 Model Output

{generated_approach}

---

### 🎯 Ground-Truth (Reference)

{reference_approach}

---

## design.md

### 🤖 Model Output

{generated_design}

---

### 🎯 Ground-Truth (Reference)

{reference_design}
"""
        comp_name = f"{task_name}_plan_{self.config.framework}_{safe_model}_{ts}_comparison.md"
        comp_path = output_dir / comp_name
        comp_path.write_text(comparison, encoding="utf-8")
        log.info("Plan comparison saved to %s", comp_path)

    def _save_function_mode(self, result: EvalResult, output_dir: Path) -> Path:
        """Save function-mode result into structured directory layout."""
        module = self.config.task.target_function.split(".")[0]
        safe_model = result.model.replace("/", "_").replace("\\", "_")
        date_str = datetime.now().strftime("%Y%m%d")

        # results/function_mode/{task}/{model_date}/{module}/
        mod_dir = output_dir / "function_mode" / result.task_name / f"{safe_model}_{date_str}" / module
        mod_dir.mkdir(parents=True, exist_ok=True)

        # Save result.json
        result_path = mod_dir / "result.json"
        result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        log.info("Function-mode result saved to %s", result_path)

        # Copy model-generated target module source from sandbox
        workspace = Path(self.runner.container) if hasattr(self.runner, 'container') and self.runner.container else None
        if workspace:
            src_file = workspace / "src" / f"{module}.py"
            if src_file.exists():
                dst_src = mod_dir / "src"
                dst_src.mkdir(exist_ok=True)
                import shutil
                shutil.copy2(src_file, dst_src / f"{module}.py")
                log.info("Archived model code: %s -> %s", src_file, dst_src / f"{module}.py")

        return result_path
