"""ReAct agent loop: Thought → Action → Observation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .docker_runner import DockerRunner
from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_FUNCTION

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Outcome of a single agent run."""

    messages: list[dict[str, str]] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    commands_run: list[dict] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "max_iterations"  # "done" | "max_iterations" | "error"


class Agent:
    """Minimal ReAct agent that drives an LLM to write & test code."""

    def __init__(
        self,
        client: LLMClient,
        runner: DockerRunner,
        max_iterations: int = 10,
        max_context_messages: int = 10,
        mode: str = "function",
        log_file: Path | None = None,
    ) -> None:
        self.client = client
        self.runner = runner
        self.max_iterations = max_iterations
        self.max_context_messages = max_context_messages
        self.mode = mode
        self.log_file = log_file

        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write(f"# Agent Interaction Log\n\n")
                    f.write(f"**Date**: {datetime.now().isoformat()}\n")
                    f.write(f"**Mode**: {mode}\n")
                    f.write("---\n\n")
            except Exception as e:
                log.warning(f"Failed to initialize log file {self.log_file}: {e}")

    def _log_step(
        self,
        iteration: int,
        inputs: list[dict[str, str]],
        response: str,
        actions: list[tuple[str, str]],
        observations: list[str],
    ) -> None:
        """Append a single interaction step to the log file."""
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"## Iteration {iteration}\n\n")

                # 1. Inputs (Full Context Window)
                f.write("### Context Window (Inputs)\n")
                for msg in inputs:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    
                    # Truncate very long inputs for readability (e.g. huge file reads)
                    display_content = content
                    if len(content) > 10000:
                        display_content = content[:5000] + "\n\n... [truncated 10000+ chars] ...\n\n" + content[-5000:]

                    f.write(f"#### {role}\n")
                    f.write(f"```text\n{display_content}\n```\n\n")

                # 2. Response
                f.write(f"### Model Response\n")
                f.write(f"```text\n{response}\n```\n\n")

                # 3. Execution
                f.write(f"### Execution\n")
                if not actions:
                    f.write("_No valid actions parsed._\n\n")
                
                for i, ((act_type, act_args), obs) in enumerate(zip(actions, observations)):
                    f.write(f"**Action {i+1}:** `{act_type}`\n")
                    # If args are multiline (like file content), block quote it
                    if "\n" in str(act_args):
                        f.write(f"```text\n{act_args}\n```\n")
                    else:
                        f.write(f"> {act_args}\n")
                    
                    # Observation
                    trunc_obs = obs
                    if len(obs) > 5000:
                        trunc_obs = obs[:2500] + "\n... [truncated] ...\n" + obs[-2500:]
                    f.write(f"**Observation:**\n```text\n{trunc_obs}\n```\n\n")
                
                f.write("---\n\n")

        except Exception as e:
            log.warning(f"Failed to write to log file: {e}")

    # ------------------------------------------------------------------
    def run(self, user_message: str) -> AgentResult:
        """Execute the ReAct loop until DONE or iteration limit."""
        result = AgentResult()
        # Use mode-appropriate system prompt:
        # - function mode mentions evaluation/tests
        # - end_to_end / plan modes use the generic prompt (no tests)
        sys_prompt = SYSTEM_PROMPT_FUNCTION if self.mode == "function" else SYSTEM_PROMPT
        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ]

        for i in range(1, self.max_iterations + 1):
            result.iterations = i
            log.info("── Iteration %d/%d ──", i, self.max_iterations)

            # Apply sliding window to keep conversation within context limits
            windowed = self._apply_sliding_window(messages)
            response_text, _ = self.client.chat(windowed)
            messages.append({"role": "assistant", "content": response_text})

            # Log first 500 chars and last 500 chars of response for debugging
            if len(response_text) > 1200:
                log.debug("Response (first 500): %s", response_text[:500])
                log.debug("Response (last 500): %s", response_text[-500:])
            else:
                log.debug("Response: %s", response_text)

            # Parse and execute actions from the response.
            actions = self._parse_all_actions(response_text)
            if not actions:
                log.info("Action: FORMAT_ERROR (no actions found)")
                observation = (
                    "[Format Error] Could not parse your action.\n"
                    "Please use exactly one of: WRITE_FILE, RUN, READ_FILE, or DONE."
                )
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})
                continue

            # Check if DONE appears after simulated Observation blocks.
            # If so, it's a hallucinated DONE — strip it and execute real actions.
            has_simulated_done = False
            if any(a[0] == "DONE" for a in actions):
                simulated_obs_count = len(
                    re.findall(r"^Observation:", response_text, re.MULTILINE))
                if simulated_obs_count > 0:
                    log.warning(
                        "Stripping simulated DONE (found %d fake Observation "
                        "blocks in model response)", simulated_obs_count)
                    actions = [(t, a) for t, a in actions if t != "DONE"]
                    has_simulated_done = True

            done = False
            observations = []
            for action_type, action_args in actions:
                log.info("Action: %s", action_type)
                if action_type == "DONE":
                    done = True
                    break
                observation = self._execute_action(action_type, action_args, result)
                observations.append(observation)

            # -- Log the step --
            self._log_step(i, windowed, response_text, actions, observations)

            if done:
                # Mode-specific DONE gating:
                # - function: require at least one pytest run
                # - end_to_end: require main.py to have been executed
                # - plan: no gating (just write plan files)
                if self.mode == "end_to_end":
                    # Check if output/reconstruction.npy exists in the sandbox
                    # rather than checking for a specific filename like "main.py".
                    # The agent may name its entry point differently (e.g. solution.py).
                    _, rc = self.runner.exec("test -f output/reconstruction.npy")
                    output_exists = (rc == 0)
                    if not output_exists and result.iterations > 1:
                        log.warning(
                            "DONE signaled but output/reconstruction.npy not found — "
                            "forcing continuation (iter %d)", i)
                        observation = (
                            "[System] output/reconstruction.npy was not found.\n"
                            "You must run your pipeline to produce "
                            "output/reconstruction.npy before signaling DONE.\n"
                            "Run your entry point (e.g. python main.py) and ensure "
                            "it saves the reconstructed image to output/reconstruction.npy.\n"
                            "Then signal DONE when the output is saved."
                        )
                        messages.append({"role": "user",
                                         "content": f"Observation:\n{observation}"})
                        continue
                    if output_exists:
                        # Sanity-check: validate output shape and basic stats
                        sanity_result = self._sanity_check_output()
                        if sanity_result:
                            log.warning("DONE accepted but sanity check has warnings: %s", sanity_result)
                            # Still accept DONE — warnings are informational only
                elif self.mode == "function":
                    pytest_run = any(
                        "pytest" in c.get("cmd", "") for c in result.commands_run
                    )
                    if not pytest_run and result.iterations > 1:
                        log.warning(
                            "DONE signaled without running pytest — "
                            "forcing continuation (iter %d)", i)
                        observation = (
                            "[System] You must run pytest on your implementation "
                            "before signaling DONE.\n"
                            "Run: python -m pytest evaluation/tests/ -v --tb=short\n"
                            "Then fix any failures and signal DONE when ready."
                        )
                        messages.append({"role": "user",
                                         "content": f"Observation:\n{observation}"})
                        continue
                # plan mode: no gating — accept DONE immediately
                result.stopped_reason = "done"
                break

            # Format observations for next prompt
            obs_text = ""
            for i, obs in enumerate(observations):
                obs_text += f"Observation {i+1}: {obs}\n"
            
            messages.append({"role": "user", "content": obs_text})
        
        return result

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_all_actions(text: str) -> list[tuple[str, dict]]:
        """Extract ALL (action_type, args_dict) from the LLM response.

        Models sometimes simulate the entire multi-turn loop in a single
        response. This method extracts every action and returns them in
        order so they can be executed sequentially.
        """
        actions = []
        # Find all "Action:" lines
        action_matches = list(re.finditer(r"^Action:\s*(\S+)", text, re.MULTILINE))
        if not action_matches:
            return []

        for idx, action_match in enumerate(action_matches):
            action_type = action_match.group(1).upper()

            # Determine the text region for this action's arguments
            start = action_match.end()
            if idx + 1 < len(action_matches):
                # Find the start of simulated "Observation:" or "Thought:" before next action
                end = action_matches[idx + 1].start()
                rest = text[start:end]
            else:
                rest = text[start:]

            if action_type == "DONE":
                actions.append(("DONE", {}))
                break  # DONE terminates the list

            if action_type == "WRITE_FILE":
                path_m = re.search(r"^Path:\s*(.+)", rest, re.MULTILINE)
                if not path_m:
                    actions.append(("FORMAT_ERROR", {"reason": "WRITE_FILE missing Path:"}))
                    continue
                path = path_m.group(1).strip()

                content_m = re.search(r"^Content:\s*\n?", rest, re.MULTILINE)
                if not content_m:
                    actions.append(("FORMAT_ERROR", {"reason": "WRITE_FILE missing Content:"}))
                    continue
                content_start = content_m.end()

                end_m = re.search(r"^END_CONTENT\s*$", rest[content_start:], re.MULTILINE)
                if end_m:
                    content = rest[content_start : content_start + end_m.start()]
                else:
                    # If no END_CONTENT, take until next Thought/Action/Observation
                    next_marker = re.search(
                        r"^(Thought:|Action:|Observation:)",
                        rest[content_start:], re.MULTILINE
                    )
                    if next_marker:
                        content = rest[content_start : content_start + next_marker.start()]
                    else:
                        content = rest[content_start:]

                # Strip optional code fences
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)

                actions.append(("WRITE_FILE", {"path": path, "content": content}))

            elif action_type == "RUN":
                cmd_m = re.search(r"^Command:\s*(.+)", rest, re.MULTILINE)
                if not cmd_m:
                    for line in rest.splitlines():
                        line = line.strip()
                        if line:
                            actions.append(("RUN", {"command": line}))
                            break
                    else:
                        actions.append(("FORMAT_ERROR", {"reason": "RUN missing Command:"}))
                else:
                    actions.append(("RUN", {"command": cmd_m.group(1).strip()}))

            elif action_type == "READ_FILE":
                path_m = re.search(r"^Path:\s*(.+)", rest, re.MULTILINE)
                if not path_m:
                    actions.append(("FORMAT_ERROR", {"reason": "READ_FILE missing Path:"}))
                else:
                    actions.append(("READ_FILE", {"path": path_m.group(1).strip()}))

            else:
                actions.append(("FORMAT_ERROR", {"reason": f"Unknown action: {action_type}"}))

        return actions

    @staticmethod
    def _parse_action(text: str) -> tuple[str, dict]:
        """Extract first (action_type, args_dict) from the LLM response.
        
        Kept for backward compatibility. Uses _parse_all_actions internally.
        """
        actions = Agent._parse_all_actions(text)
        if not actions:
            return "FORMAT_ERROR", {}
        return actions[0]

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute_action(
        self,
        action_type: str,
        args: dict,
        result: AgentResult,
    ) -> str:
        if action_type == "FORMAT_ERROR":
            reason = args.get("reason", "Could not parse your action.")
            
            # Specific feedback for the common "CHECK" hallucination
            if "Unknown action: CHECK" in reason:
                return (
                    f"[Format Error] {reason}\n"
                    "CHECK is not a valid action. To verify data or code, you must "
                    "write a Python script (WRITE_FILE) and then execute it (RUN).\n"
                    "Valid actions: WRITE_FILE, RUN, READ_FILE, DONE."
                )

            return (
                f"[Format Error] {reason}\n"
                "Please use exactly one of: WRITE_FILE, RUN, READ_FILE, or DONE."
            )

        if action_type == "WRITE_FILE":
            path = args["path"]
            self.runner.write_file(path, args["content"])
            if path not in result.files_written:
                result.files_written.append(path)
            return f"File written: {path}"

        if action_type == "RUN":
            cmd = args["command"]
            output, rc = self.runner.exec(cmd)
            result.commands_run.append({"cmd": cmd, "exit_code": rc, "output": output})
            status = "OK" if rc == 0 else f"EXIT CODE {rc}"
            # Truncate long command output (e.g. verbose pytest)
            output = self._truncate_text(output, max_chars=8000, keep_start=3500, keep_end=3500)
            return f"[{status}]\n{output}"

        if action_type == "READ_FILE":
            content = self.runner.read_file(args["path"])
            return self._truncate_text(content, max_chars=8000, keep_start=3500, keep_end=3500)

        return f"[Error] Unhandled action type: {action_type}"

    # ------------------------------------------------------------------
    # Context management helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int = 12000,
        keep_start: int = 5000,
        keep_end: int = 5000,
    ) -> str:
        """Truncate *text* if it exceeds *max_chars*, keeping head and tail."""
        if len(text) <= max_chars:
            return text
        omitted = len(text) - keep_start - keep_end
        return (
            f"{text[:keep_start]}\n\n"
            f"[...truncated {omitted} characters...]\n\n"
            f"{text[-keep_end:]}"
        )

    def _apply_sliding_window(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Return a windowed copy of *messages* that fits the context budget.

        Always keeps:
          - messages[0]  (system prompt)
          - messages[1]  (initial user prompt)
        Then keeps the most recent *max_context_messages - 2* messages.
        Older messages in between are replaced by a single summary placeholder.

        Finally, enforces a total character budget by truncating the largest
        messages (assistant WRITE_FILE responses) if the total exceeds the cap.
        """
        MAX_TOTAL_CHARS = 90_000  # ~22K tokens, safe for 128K context

        # max_context_messages includes system + initial user + recent tail
        if len(messages) <= self.max_context_messages:
            windowed = list(messages)
        else:
            head = messages[:2]  # system + initial user
            recent_count = self.max_context_messages - 2
            tail = messages[-recent_count:]
            dropped = len(messages) - 2 - recent_count
            dropped_iters = dropped // 2
            summary = {
                "role": "user",
                "content": (
                    f"[Earlier conversation with {dropped_iters} iterations "
                    f"({dropped} messages) omitted to fit context window]"
                ),
            }
            windowed = head + [summary] + tail
            log.debug(
                "Sliding window: kept %d head + 1 summary + %d recent "
                "(dropped %d messages)",
                len(head), len(tail), dropped,
            )

        # Enforce total character budget by truncating the largest messages
        total = sum(len(m["content"]) for m in windowed)
        while total > MAX_TOTAL_CHARS and len(windowed) > 3:
            # Find the largest non-system, non-initial-user message
            max_idx, max_len = -1, 0
            for idx in range(2, len(windowed)):
                clen = len(windowed[idx]["content"])
                if clen > max_len:
                    max_idx, max_len = idx, clen
            if max_idx < 0 or max_len <= 2000:
                break  # nothing left to trim
            msg = windowed[max_idx]
            trimmed = self._truncate_text(msg["content"], max_chars=3000,
                                          keep_start=1500, keep_end=1200)
            windowed[max_idx] = {"role": msg["role"], "content": trimmed}
            total = sum(len(m["content"]) for m in windowed)
            log.debug(
                "Budget trim: msg[%d] %d→%d chars, total now %d",
                max_idx, max_len, len(trimmed), total,
            )

        return windowed
