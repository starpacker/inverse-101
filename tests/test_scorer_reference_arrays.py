from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluation_harness.config import LLMConfig, RunConfig, TaskConfig
from evaluation_harness.frameworks.claude_code.copilot_scorer import (
    _compute_quality_metrics as compute_copilot_quality_metrics,
)
from evaluation_harness.scorer import Scorer


class FakeRunner:
    def __init__(self, workspace: Path) -> None:
        self.container = str(workspace)
        self.workspace = workspace

    def exec(self, command: str) -> tuple[str, int]:
        if command == "python _score_quality.py":
            proc = subprocess.run(
                [sys.executable, "_score_quality.py"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return (proc.stdout + proc.stderr).strip(), proc.returncode

        prefix = "python -c '"
        if command.startswith(prefix) and command.endswith("'"):
            snippet = command[len(prefix):-1]
            proc = subprocess.run(
                [sys.executable, "-c", snippet],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return (proc.stdout + proc.stderr).strip(), proc.returncode
        raise AssertionError(f"unexpected command: {command}")

    def write_file(self, path: str, content: str) -> None:
        full_path = self.workspace / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")


def _make_scorer(tmp_path: Path) -> tuple[Scorer, Path, Path]:
    task_dir = tmp_path / "task"
    workspace = tmp_path / "workspace"
    (workspace / "output").mkdir(parents=True)
    task_dir.mkdir()
    config = RunConfig(
        llm=LLMConfig(model="test-model"),
        task=TaskConfig(task_name="task", task_dir=task_dir, mode="end_to_end"),
    )
    return Scorer(FakeRunner(workspace), config), task_dir, workspace


def test_quality_metrics_loads_data_ground_truth_npz_single_array(tmp_path: Path) -> None:
    scorer, task_dir, workspace = _make_scorer(tmp_path)
    gt = np.array([[1.0, 2.0, 3.0, 4.0]])
    (task_dir / "data").mkdir()
    np.savez(task_dir / "data" / "ground_truth.npz", phantom=gt)
    np.save(workspace / "output" / "reconstruction.npy", gt)

    metrics = scorer._compute_quality_metrics()

    assert metrics is not None
    assert metrics["nrmse"] == 0.0
    assert metrics["ncc"] == 1.0


def test_quality_metrics_selects_matching_array_from_reference_npz(tmp_path: Path) -> None:
    scorer, task_dir, workspace = _make_scorer(tmp_path)
    gt = np.array([[0.0, 1.0], [2.0, 3.0]])
    ref_dir = task_dir / "evaluation" / "reference_outputs"
    ref_dir.mkdir(parents=True)
    np.savez(
        ref_dir / "reference_reconstruction.npz",
        sinogram=np.ones((2, 3)),
        tissue_map=gt[np.newaxis, :, :],
        bone_map=np.ones((1, 3, 3)),
    )
    np.save(workspace / "output" / "reconstruction.npy", gt)

    metrics = scorer._compute_quality_metrics()

    assert metrics is not None
    assert metrics["nrmse"] == 0.0
    assert metrics["ncc"] == 1.0


def test_quality_metrics_reports_missing_reference_candidates(tmp_path: Path) -> None:
    scorer, _task_dir, workspace = _make_scorer(tmp_path)
    np.save(workspace / "output" / "reconstruction.npy", np.ones((2, 2)))

    metrics = scorer._compute_quality_metrics()

    assert metrics is not None
    assert "error" in metrics
    assert "reference array not found" in metrics["error"]
    assert "ground_truth.npz" in metrics["error"]


def test_copilot_quality_metrics_uses_npz_reference_loader(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    workspace = tmp_path / "workspace"
    ref_dir = task_dir / "evaluation" / "reference_outputs"
    (workspace / "output").mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    gt = np.array([[1.0, 0.0], [0.5, 2.0]])
    np.savez(ref_dir / "reconstruction.npz", preview=np.ones((3, 3)), reconstruction=gt)
    np.save(workspace / "output" / "reconstruction.npy", gt)

    metrics = compute_copilot_quality_metrics(workspace, task_dir)

    assert metrics["nrmse"] == 0.0
    assert metrics["ncc"] == 1.0
