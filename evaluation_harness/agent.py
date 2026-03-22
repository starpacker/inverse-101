"""ReAct agent loop: Thought → Action → Observation."""

import logging
import re
from dataclasses import dataclass, field

from .docker_runner import DockerRunner
from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT

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
        max_iterations: int = 20,
    ) -> None:
        self.client = client
        self.runner = runner
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    def run(self, user_message: str) -> AgentResult:
        """Execute the ReAct loop until DONE or iteration limit."""
        result = AgentResult()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for i in range(1, self.max_iterations + 1):
            result.iterations = i
            log.info("── Iteration %d/%d ──", i, self.max_iterations)

            response_text, _ = self.client.chat(messages)
            messages.append({"role": "assistant", "content": response_text})

            action_type, action_args = self._parse_action(response_text)
            log.info("Action: %s", action_type)

            if action_type == "DONE":
                result.stopped_reason = "done"
                break

            observation = self._execute_action(action_type, action_args, result)
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        result.messages = messages
        return result

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_action(text: str) -> tuple[str, dict]:
        """Extract (action_type, args_dict) from the LLM response.

        Supported formats::

            Action: WRITE_FILE
            Path: src/foo.py
            Content:
            ...file content...
            END_CONTENT

            Action: RUN
            Command: pytest -v

            Action: READ_FILE
            Path: data/meta_data

            Action: DONE
        """
        # Find the *last* "Action:" line (models sometimes repeat)
        action_match = None
        for m in re.finditer(r"^Action:\s*(\S+)", text, re.MULTILINE):
            action_match = m
        if not action_match:
            return "FORMAT_ERROR", {}

        action_type = action_match.group(1).upper()
        rest = text[action_match.end():]

        if action_type == "DONE":
            return "DONE", {}

        if action_type == "WRITE_FILE":
            path_m = re.search(r"^Path:\s*(.+)", rest, re.MULTILINE)
            if not path_m:
                return "FORMAT_ERROR", {"reason": "WRITE_FILE missing Path:"}
            path = path_m.group(1).strip()

            # Content is everything after "Content:" until END_CONTENT or end
            content_m = re.search(r"^Content:\s*\n?", rest, re.MULTILINE)
            if not content_m:
                return "FORMAT_ERROR", {"reason": "WRITE_FILE missing Content:"}
            content_start = content_m.end()

            # Try to find END_CONTENT marker
            end_m = re.search(r"^END_CONTENT\s*$", rest[content_start:], re.MULTILINE)
            if end_m:
                content = rest[content_start : content_start + end_m.start()]
            else:
                content = rest[content_start:]

            # Strip optional code fences that models like to add
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)

            return "WRITE_FILE", {"path": path, "content": content}

        if action_type == "RUN":
            cmd_m = re.search(r"^Command:\s*(.+)", rest, re.MULTILINE)
            if not cmd_m:
                # Fallback: take the next non-empty line
                for line in rest.splitlines():
                    line = line.strip()
                    if line:
                        return "RUN", {"command": line}
                return "FORMAT_ERROR", {"reason": "RUN missing Command:"}
            return "RUN", {"command": cmd_m.group(1).strip()}

        if action_type == "READ_FILE":
            path_m = re.search(r"^Path:\s*(.+)", rest, re.MULTILINE)
            if not path_m:
                return "FORMAT_ERROR", {"reason": "READ_FILE missing Path:"}
            return "READ_FILE", {"path": path_m.group(1).strip()}

        return "FORMAT_ERROR", {"reason": f"Unknown action: {action_type}"}

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
            return f"[{status}]\n{output}"

        if action_type == "READ_FILE":
            content = self.runner.read_file(args["path"])
            return content

        return f"[Error] Unhandled action type: {action_type}"
