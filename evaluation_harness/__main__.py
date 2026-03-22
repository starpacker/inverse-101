"""CLI entry point: python -m evaluation_harness run ..."""

import argparse
import json
import logging
import os
import sys
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
    )

    # Run
    runner = BenchmarkRunner(config)
    result = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print(f"Task:    {result.task_name}")
    print(f"Mode:    {result.mode}")
    print(f"Model:   {result.model}")
    print(f"Status:  {result.stopped_reason}")
    if result.tests_total > 0:
        print(f"Tests:   {result.tests_passed}/{result.tests_total} passed ({result.test_pass_rate:.0%})")
    if result.quality_metrics:
        print(f"Quality: {json.dumps(result.quality_metrics)}")
    print(f"Tokens:  {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})")
    print(f"Time:    {result.wall_time_seconds:.1f}s ({result.iterations} iterations)")
    print("=" * 60)


if __name__ == "__main__":
    main()
