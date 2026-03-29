"""CLI entry point: python -m evaluation_harness run ..."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import LLMConfig, RunConfig, TaskConfig
from .runner import BenchmarkRunner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation_harness",
        description="Run imaging-101 benchmark evaluations.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_p = sub.add_parser("run", help="Run a benchmark evaluation")
    run_p.add_argument("--task", required=True, help="Task name, e.g. eht_black_hole")
    run_p.add_argument(
        "--mode",
        required=True,
        choices=["plan", "function", "end_to_end"],
        help="Evaluation mode",
    )
    run_p.add_argument("--model", required=True, help="LLM model name, e.g. gpt-4o")
    run_p.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL",
    )
    run_p.add_argument(
        "--api-key",
        default=None,
        help="API key (defaults to OPENAI_API_KEY env var)",
    )
    run_p.add_argument(
        "--target-function",
        default=None,
        help="For function mode: module.function_name, e.g. preprocessing.load_observation",
    )
    run_p.add_argument("--max-iterations", type=int, default=20)
    run_p.add_argument("--docker-image", default="imaging101-sandbox")
    run_p.add_argument("--timeout", type=int, default=600)
    run_p.add_argument("--output", default="results", help="Output directory")
    run_p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    run_p.add_argument("--log-file", help="Path to save detailed interaction logs")
    run_p.add_argument(
        "--framework",
        default="react",
        choices=["react", "multi_agent"],
        help="Agent framework: 'react' (single-agent ReAct loop) or 'multi_agent' (Plan→Architect→Code→Judge pipeline)",
    )

    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve task directory
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "tasks" / args.task

    # Build config
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: --api-key or OPENAI_API_KEY env var required", file=sys.stderr)
        sys.exit(1)

    # Determine log file path
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        # Auto-generate default log path: logs/interactions/<task>_<mode>[_<target>]_<timestamp>.md
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_suffix = f"_{args.target_function}" if args.target_function else ""
        log_dir = Path("logs") / "interactions"
        log_file = log_dir / f"{args.task}_{args.mode}{target_suffix}_{ts}.md"

    config = RunConfig(
        llm=LLMConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
        ),
        task=TaskConfig(
            task_name=args.task,
            task_dir=task_dir,
            mode=args.mode,
            target_function=args.target_function,
        ),
        max_iterations=args.max_iterations,
        docker_image=args.docker_image,
        timeout_seconds=args.timeout,
        output_dir=Path(args.output),
        log_file=log_file,
        framework=args.framework,
    )

    # Run
    runner = BenchmarkRunner(config)
    result = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print(f"Task:      {result.task_name}")
    print(f"Mode:      {result.mode}")
    print(f"Model:     {result.model}")
    print(f"Framework: {args.framework}")
    print(f"Status:    {result.stopped_reason}")
    if result.tests_total > 0:
        print(f"Tests:   {result.tests_passed}/{result.tests_total} passed ({result.test_pass_rate:.0%})")
    if result.quality_metrics:
        qm = result.quality_metrics
        if "error" in qm:
            print(f"Quality: ERROR — {qm['error']}")
        else:
            print(f"Quality: NRMSE={qm.get('nrmse', 'N/A')}, NCC={qm.get('ncc', 'N/A')}, "
                  f"PSNR={qm.get('psnr', 'N/A')}, SSIM={qm.get('ssim', 'N/A')}")
    if result.visualization_paths:
        print(f"Figures: {len(result.visualization_paths)} generated")
        for name, path in result.visualization_paths.items():
            print(f"  {name}: {path}")
    print(f"Tokens:  {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})")
    print(f"LLM calls: {result.llm_calls}")
    print(f"Time:    {result.wall_time_seconds:.1f}s ({result.iterations} iterations)")
    print("=" * 60)


if __name__ == "__main__":
    main()
