import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "evaluation_harness", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_dry_run_accepts_mechanism_alias_for_end2end():
    result = run_cli(
        "run",
        "--task",
        "ct_fan_beam",
        "--mechanism",
        "end2end",
        "--model",
        "demo-model",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "mode: end_to_end" in result.stdout
    assert "task: ct_fan_beam" in result.stdout
    assert "model: demo-model" in result.stdout


def test_dry_run_normalizes_planning_alias():
    result = run_cli(
        "run",
        "--task",
        "ct_fan_beam",
        "--mode",
        "planning",
        "--model",
        "demo-model",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "mode: plan" in result.stdout
