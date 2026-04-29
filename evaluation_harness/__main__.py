"""CLI entry point: python -m evaluation_harness run | prepare | collect ..."""

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
    run_p.add_argument("--task", required=True, help="Task name, e.g. eht_black_hole_original")
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
        choices=["react", "multi_agent", "copilot", "deepcode"],
        help=(
            "Agent framework: 'react' (single-agent ReAct loop), "
            "'multi_agent' (Plan→Architect→Code→Judge pipeline), "
            "'copilot' (third-party agent — prepares sandbox + prompt), or "
            "'deepcode' (HKUDS DeepCode autonomous multi-agent)"
        ),
    )
    run_p.add_argument(
        "--level",
        default="L1",
        choices=["L1", "L2", "L3"],
        help=(
            "End-to-end difficulty level: "
            "'L1' = task description only (agent plans from scratch), "
            "'L2' = task description + approach (approach.md given), "
            "'L3' = task description + approach + design (both plan docs given)"
        ),
    )

    # --- prepare subcommand (copilot framework) ---
    prep_p = sub.add_parser(
        "prepare",
        help="Prepare a sandbox workspace for third-party agent evaluation",
    )
    prep_p.add_argument("--task", required=True, help="Task name, e.g. eht_black_hole_original")
    prep_p.add_argument(
        "--level",
        default="L1",
        choices=["L1", "L2", "L3"],
        help="End-to-end difficulty level",
    )
    prep_p.add_argument(
        "--workspace-dir",
        default=None,
        help="Custom workspace directory (default: auto-generated under ~/copilot_workspaces/)",
    )
    prep_p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    # --- collect subcommand (copilot framework) ---
    coll_p = sub.add_parser(
        "collect",
        help="Collect and score results from a completed third-party agent evaluation",
    )
    coll_p.add_argument("--task", required=True, help="Task name, e.g. eht_black_hole_original")
    coll_p.add_argument(
        "--workspace-dir",
        required=True,
        help="Path to the sandbox workspace to collect results from",
    )
    coll_p.add_argument(
        "--level",
        default="L1",
        choices=["L1", "L2", "L3"],
        help="End-to-end difficulty level used during preparation",
    )
    coll_p.add_argument(
        "--agent-name",
        default="unknown",
        help="Name of the third-party agent (e.g. 'copilot', 'claude_code', 'cursor')",
    )
    coll_p.add_argument("--output", default="results", help="Output directory for results JSON")
    coll_p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    # --- summarize subcommand (function mode) ---
    sum_p = sub.add_parser(
        "summarize",
        help="Generate summary.json for a function-mode evaluation run",
    )
    sum_p.add_argument("--dir", required=True, help="Path to function-mode run dir (e.g. results/function_mode/task/model_date)")
    sum_p.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "prepare":
        _handle_prepare(args)
    elif args.command == "collect":
        _handle_collect(args)
    elif args.command == "run":
        _handle_run(args)
    elif args.command == "summarize":
        _handle_summarize(args)
    else:
        parser.print_help()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_run(args) -> None:
    """Handle the 'run' subcommand (ReAct / multi_agent / copilot)."""
    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve task directory
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "tasks" / args.task

    # --- copilot framework: prepare + wait + collect ---
    if args.framework == "copilot":
        _handle_copilot_run(args, task_dir)
        return

    # --- deepcode framework: prepare + run DeepCode + collect ---
    if args.framework == "deepcode":
        _handle_deepcode_run(args, task_dir)
        return

    # Build config
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: --api-key or OPENAI_API_KEY env var required", file=sys.stderr)
        sys.exit(1)

    # Determine log file path
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_suffix = f"_{args.target_function}" if args.target_function else ""
        level_suffix = f"_{args.level}" if args.mode == "end_to_end" else ""
        log_dir = Path("logs") / "interactions"
        log_file = log_dir / f"{args.task}_{args.mode}{level_suffix}{target_suffix}_{ts}.md"

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
            level=args.level,
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
    _print_run_summary(result, args)


def _handle_copilot_run(args, task_dir: Path) -> None:
    """Handle 'run --framework copilot': prepare, wait for user, collect.

    This is a convenience wrapper that combines prepare + collect into a
    single interactive session.
    """
    from .frameworks.claude_code.copilot_runner import prepare_copilot_evaluation, collect_copilot_results
    from .frameworks.claude_code.copilot_scorer import score_copilot_results

    print("=" * 60)
    print("THIRD-PARTY AGENT EVALUATION (copilot framework)")
    print("=" * 60)

    # Prepare
    prep_result = prepare_copilot_evaluation(
        task_name=args.task,
        task_dir=task_dir,
        level=args.level,
    )

    print(f"\n✅ Sandbox prepared at:")
    print(f"   {prep_result['workspace_path']}")
    print(f"\n📋 Prompt saved to:")
    print(f"   {prep_result['prompt_file']}")
    print(f"\n📝 Level: {args.level}")
    print(f"\nInstructions:")
    print(f"  1. Open the workspace in your agent's IDE:")
    print(f"     cd {prep_result['workspace_path']}")
    print(f"  2. Paste the prompt from .prompt.md into the agent")
    print(f"  3. Let the agent work until it produces output/reconstruction.npy")
    print(f"  4. Press ENTER here when the agent is done")
    print("=" * 60)

    try:
        input("\n⏳ Press ENTER when the agent has finished... ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    # Collect
    workspace_path = Path(prep_result["workspace_path"])
    results = collect_copilot_results(workspace_path, task_dir, args.level)

    # Score
    eval_result = score_copilot_results(
        workspace_path=workspace_path,
        task_dir=task_dir,
        task_name=args.task,
        level=args.level,
        agent_name=getattr(args, 'model', 'copilot'),
        framework="copilot",
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = getattr(args, 'model', 'copilot').replace("/", "_")
    name = f"{args.task}_end_to_end_{args.level}_copilot_{safe_model}_{ts}.json"
    result_path = output_dir / name

    import dataclasses
    with open(result_path, "w") as f:
        json.dump(dataclasses.asdict(eval_result), f, indent=2, default=str)
    print(f"\n📊 Results saved to: {result_path}")

    _print_copilot_summary(eval_result)


def _handle_deepcode_run(args, task_dir: Path) -> None:
    """Handle 'run --framework deepcode': prepare + DeepCode CLI + collect."""
    from .frameworks.deepcode.runner import DeepCodeRunner

    print("=" * 60)
    print("DEEPCODE EVALUATION (HKUDS/DeepCode)")
    print("=" * 60)

    runner = DeepCodeRunner(
        task_name=args.task,
        task_dir=task_dir,
        level=args.level,
    )

    # Step 1: Prepare workspace
    prep_result = runner.prepare_workspace()
    workspace = Path(prep_result["workspace_path"])

    print(f"\n✅ Sandbox prepared at: {workspace}")
    print(f"📋 DeepCode prompt: {prep_result.get('deepcode_prompt_file', 'N/A')}")
    print(f"📝 Level: {args.level}")

    # Step 2: Try CLI mode, fall back to manual
    try:
        print("\n🚀 Launching DeepCode CLI...")
        cli_result = runner.run_cli(workspace, timeout=3600)
        print(f"   Status: {cli_result['status']}")
        print(f"   Duration: {cli_result['duration_seconds']:.0f}s")
        if cli_result["status"] == "error":
            print(f"   stderr: {cli_result['stderr'][:500]}")
    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        print("Falling back to manual mode...")
        print(f"\nInstructions:")
        print(f"  1. cd {workspace}")
        print(f"  2. Run: deepcode --cli")
        print(f"     (or paste deepcode_prompt.txt into DeepCode web UI)")
        print(f"  3. Press ENTER when done")
        try:
            input("\n⏳ Press ENTER when DeepCode has finished... ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            import sys
            sys.exit(1)

    # Step 3: Collect & score
    eval_result = runner.collect_and_score(workspace, Path(args.output))
    _print_copilot_summary(eval_result)


def _handle_prepare(args) -> None:
    """Handle the 'prepare' subcommand."""
    from .frameworks.claude_code.copilot_runner import prepare_copilot_evaluation

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "tasks" / args.task

    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else None

    prep_result = prepare_copilot_evaluation(
        task_name=args.task,
        task_dir=task_dir,
        level=args.level,
        workspace_dir=workspace_dir,
    )

    print("=" * 60)
    print("SANDBOX PREPARED FOR THIRD-PARTY AGENT EVALUATION")
    print("=" * 60)
    print(f"\n  Task:      {args.task}")
    print(f"  Level:     {args.level}")
    print(f"  Workspace: {prep_result['workspace_path']}")
    print(f"  Prompt:    {prep_result['prompt_file']}")
    print(f"\nFiles in workspace:")
    workspace = Path(prep_result['workspace_path'])
    for item in sorted(workspace.rglob("*")):
        if item.is_file() and "__pycache__" not in str(item):
            rel = item.relative_to(workspace)
            size = item.stat().st_size
            perm = "RO" if not os.access(item, os.W_OK) else "RW"
            print(f"  [{perm}] {rel}  ({size:,} bytes)")

    print(f"\nNext steps:")
    print(f"  1. Open the workspace: cd {prep_result['workspace_path']}")
    print(f"  2. Paste the prompt from .prompt.md into your agent")
    print(f"  3. Let the agent produce output/reconstruction.npy")
    print(f"  4. Run: python -m evaluation_harness collect \\")
    print(f"       --task {args.task} \\")
    print(f"       --workspace-dir {prep_result['workspace_path']} \\")
    print(f"       --level {args.level} \\")
    print(f"       --agent-name <agent_name>")
    print("=" * 60)


def _handle_collect(args) -> None:
    """Handle the 'collect' subcommand."""
    from .frameworks.claude_code.copilot_runner import collect_copilot_results
    from .frameworks.claude_code.copilot_scorer import score_copilot_results

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "tasks" / args.task
    workspace_path = Path(args.workspace_dir)

    if not workspace_path.exists():
        print(f"Error: workspace directory not found: {workspace_path}", file=sys.stderr)
        sys.exit(1)

    # Collect raw results
    raw_results = collect_copilot_results(workspace_path, task_dir, args.level)

    # Score
    eval_result = score_copilot_results(
        workspace_path=workspace_path,
        task_dir=task_dir,
        task_name=args.task,
        level=args.level,
        agent_name=args.agent_name,
        framework="copilot",
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_agent = args.agent_name.replace("/", "_").replace(" ", "_")
    name = f"{args.task}_end_to_end_{args.level}_copilot_{safe_agent}_{ts}.json"
    result_path = output_dir / name

    import dataclasses
    with open(result_path, "w") as f:
        json.dump(dataclasses.asdict(eval_result), f, indent=2, default=str)

    print(f"\n📊 Results saved to: {result_path}")
    _print_copilot_summary(eval_result)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_run_summary(result, args) -> None:
    """Print summary for react/multi_agent runs."""
    print("\n" + "=" * 60)
    print(f"Task:      {result.task_name}")
    print(f"Mode:      {result.mode}")
    if result.mode == "end_to_end":
        print(f"Level:     {args.level}")
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
    # Visualization reporting disabled — handled by dedicated downstream agent
    # if result.visualization_paths:
    #     print(f"Figures: {len(result.visualization_paths)} generated")
    #     for name, path in result.visualization_paths.items():
    #         print(f"  {name}: {path}")
    print(f"Tokens:  {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})")
    print(f"LLM calls: {result.llm_calls}")
    print(f"Time:    {result.wall_time_seconds:.1f}s ({result.iterations} iterations)")
    print("=" * 60)


def _print_copilot_summary(result) -> None:
    """Print summary for copilot evaluation results."""
    print("\n" + "=" * 60)
    print("THIRD-PARTY AGENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Task:      {result.task_name}")
    print(f"  Level:     {result.level}")
    print(f"  Agent:     {result.model}")
    print(f"  Framework: {result.framework}")

    if result.quality_metrics:
        qm = result.quality_metrics
        if "error" in qm:
            print(f"  Quality:   ERROR — {qm['error']}")
        else:
            print(f"  NRMSE:     {qm.get('nrmse', 'N/A')}")
            print(f"  NCC:       {qm.get('ncc', 'N/A')}")
            print(f"  PSNR:      {qm.get('psnr', 'N/A')} dB")
            print(f"  SSIM:      {qm.get('ssim', 'N/A')}")
    else:
        print("  Quality:   No reconstruction found")

    if result.files_created:
        print(f"  Files:     {len(result.files_created)} created")
    print("=" * 60)


def _handle_summarize(args) -> None:
    """Generate summary.json for a function-mode evaluation run directory."""
    run_dir = Path(args.dir)
    if not run_dir.is_dir():
        print(f"Error: directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    modules = {}
    total_tests = total_passed = total_failed = total_tokens = 0
    total_wall_time = 0.0
    task_name = model = framework = ""

    for mod_dir in sorted(run_dir.iterdir()):
        result_file = mod_dir / "result.json"
        if not mod_dir.is_dir() or not result_file.exists():
            continue
        with open(result_file) as f:
            r = json.load(f)
        mod_name = mod_dir.name
        if not task_name:
            task_name = r.get("task_name", "")
            model = r.get("model", "")
            framework = r.get("framework", "")

        modules[mod_name] = {
            "tests_total": r["tests_total"],
            "tests_passed": r["tests_passed"],
            "tests_failed": r["tests_failed"],
            "test_pass_rate": r["test_pass_rate"],
            "test_details": r.get("test_details", []),
            "iterations": r["iterations"],
            "wall_time_seconds": r["wall_time_seconds"],
            "total_tokens": r["total_tokens"],
            "stopped_reason": r["stopped_reason"],
        }
        total_tests += r["tests_total"]
        total_passed += r["tests_passed"]
        total_failed += r["tests_failed"]
        total_tokens += r["total_tokens"]
        total_wall_time += r["wall_time_seconds"]

    if not modules:
        print("Error: no module result.json files found", file=sys.stderr)
        sys.exit(1)

    summary = {
        "task_name": task_name,
        "mode": "function",
        "model": model,
        "framework": framework,
        "timestamp": run_dir.name.split("_")[-1] if "_" in run_dir.name else "",
        "aggregate": {
            "tests_total": total_tests,
            "tests_passed": total_passed,
            "tests_failed": total_failed,
            "test_pass_rate": round(total_passed / total_tests, 4) if total_tests > 0 else 0,
            "total_tokens": total_tokens,
            "total_wall_time_seconds": round(total_wall_time, 1),
        },
        "per_module": modules,
    }

    out = run_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary table
    agg = summary["aggregate"]
    print("\n" + "=" * 60)
    print(f"FUNCTION-MODE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Task:       {task_name}")
    print(f"  Model:      {model}")
    print(f"  Framework:  {framework}")
    print(f"  Overall:    {agg['tests_passed']}/{agg['tests_total']} passed ({agg['test_pass_rate']:.0%})")
    print(f"  Tokens:     {agg['total_tokens']:,}")
    print(f"  Wall time:  {agg['total_wall_time_seconds']:.0f}s")
    print("-" * 60)
    print(f"  {'Module':<20} {'Passed':<10} {'Rate':<10} {'Iters':<8} {'Status'}")
    print("-" * 60)
    for mod, info in modules.items():
        rate = f"{info['test_pass_rate']:.0%}"
        print(f"  {mod:<20} {info['tests_passed']}/{info['tests_total']:<7} {rate:<10} {info['iterations']:<8} {info['stopped_reason']}")
    print("=" * 60)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
