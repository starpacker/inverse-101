"""Scoring for third-party (copilot) agent evaluations.

This module computes quality metrics for reconstructions produced by
third-party agents working in a prepared sandbox. It uses the same
metric computation as the standard Scorer but operates directly on
files rather than through a sandbox runner.

The key difference from scorer.py:
- No sandbox runner needed — works directly on filesystem paths
- Reads ground truth from the task directory (not the sandbox)
- Collects file inventories from the workspace
- Returns an EvalResult compatible with the standard output format
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from evaluation_harness.scorer import (
    EvalResult,
    _prepare_metric_array,
    compute_quality_metrics_from_arrays,
    load_reference_array,
)

log = logging.getLogger(__name__)


def score_copilot_results(
    workspace_path: Path,
    task_dir: Path,
    task_name: str,
    level: str = "L1",
    agent_name: str = "copilot",
    framework: str = "copilot",
) -> EvalResult:
    """Score the results of a third-party agent evaluation.

    Parameters
    ----------
    workspace_path : Path
        Path to the sandbox workspace where the agent worked.
    task_dir : Path
        Path to the original task directory (for ground truth).
    task_name : str
        Name of the task.
    level : str
        Difficulty level used (L1/L2/L3).
    agent_name : str
        Name of the third-party agent.
    framework : str
        Framework identifier (always "copilot" for this module).

    Returns
    -------
    EvalResult
        Evaluation results compatible with the standard output format.
    """
    workspace_path = Path(workspace_path)
    task_dir = Path(task_dir)

    result = EvalResult(
        task_name=task_name,
        mode="end_to_end",
        model=agent_name,
        framework=framework,
        level=level,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Collect files created by the agent
    result.files_created = _collect_files(workspace_path)

    # Compute quality metrics
    result.quality_metrics = _compute_quality_metrics(workspace_path, task_dir)

    # Visualization is handled by a dedicated downstream agent,
    # not the end-to-end evaluation pipeline.
    # if result.quality_metrics and "error" not in result.quality_metrics:
    #     result.visualization_paths = _generate_visualizations(
    #         workspace_path, task_dir, result
    #     )

    # Set stopped reason
    recon_path = workspace_path / "output" / "reconstruction.npy"
    if recon_path.exists():
        result.stopped_reason = "completed"
    else:
        result.stopped_reason = "no_reconstruction"

    return result


def _collect_files(workspace_path: Path) -> list[str]:
    """Collect list of all files in the workspace."""
    files = []
    for root, dirs, filenames in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in filenames:
            if f.startswith("."):
                continue
            full = Path(root) / f
            rel = full.relative_to(workspace_path)
            files.append(str(rel))
    return sorted(files)


def _compute_quality_metrics(workspace_path: Path, task_dir: Path) -> dict:
    """Compare reconstruction against the task reference array.

    Uses the same metric computation as scorer.py for consistency.
    """
    recon_path = workspace_path / "output" / "reconstruction.npy"

    if not recon_path.exists():
        return {"error": "output/reconstruction.npy not found"}

    try:
        out = np.load(str(recon_path), allow_pickle=True)
    except Exception as e:
        return {"error": f"Failed to load reconstruction: {e}"}

    try:
        out = _prepare_metric_array(out, label="reconstruction")
        gt, _ = load_reference_array(task_dir, target_shape=out.shape)
    except Exception as e:
        return {"error": str(e)}

    return compute_quality_metrics_from_arrays(out, gt)


def _ssim_2d(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """Simplified SSIM (same as in scorer.py and self_eval.py)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a2, sig_b2 = a.var(), b.var()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(num / den)


def _generate_visualizations(
    workspace_path: Path,
    task_dir: Path,
    result: EvalResult,
) -> dict[str, str]:
    """Generate evaluation figures.

    Same as scorer.py but works directly on filesystem paths.
    """
    from evaluation_harness.visualizer import generate_eval_figures

    recon_path = workspace_path / "output" / "reconstruction.npy"
    if not recon_path.exists():
        return {}

    try:
        recon = _prepare_metric_array(np.load(str(recon_path), allow_pickle=True), label="reconstruction")
        gt, _ = load_reference_array(task_dir, target_shape=recon.shape)
        if recon.ndim != 2 or gt.ndim != 2:
            return {}
    except Exception as e:
        log.warning("Failed to load arrays for visualization: %s", e)
        return {}

    # Output directory
    safe_agent = result.model.replace("/", "_").replace("\\", "_")
    fig_dir = workspace_path / "output" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_label = f"copilot_{safe_agent}"
    paths = generate_eval_figures(
        reconstruction=recon,
        ground_truth=gt,
        metrics=result.quality_metrics,
        output_dir=fig_dir,
        run_label=run_label,
        task_name=result.task_name,
    )

    return paths
