"""Coder agent: implements code modules based on plan + architecture.

Adapted from agentic_pipeline_dev for the imaging-101 benchmark.
Instead of a single solver.py, the Coder writes complete files
(src/preprocessing.py, src/physics_model.py, etc.) that the sandbox
runner executes.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .base import BaseAgent


class CoderAgent(BaseAgent):
    """Implements complete Python files based on plan and architecture."""

    def _build_system_prompt(self) -> str:
        return """\
You are a Senior Python Developer in Scientific Computing.
Your Goal: Implement a SPECIFIC Python file based on the plan and architecture provided.

### Critical Rules:
1. Output: Return ONLY the complete Python file content — NO explanations, NO markdown fences.
2. MANDATORY: Write the FULL implementation logic. NO `pass`, NO `TODO`, NO placeholders.
3. Data Flow:
   - Load data from data/ directory (numpy .npy or .npz files)
   - Save final reconstruction to output/reconstruction.npy as a 2-D numpy array
4. Type Safety: Explicitly handle dtype conversions (e.g., .astype(np.float64))
5. SELF-CONTAINED: Only import packages from requirements.txt and standard library.
   Do NOT import jax, torch, tensorflow, or any package not in requirements.txt.
   Do NOT import from local project files unless they are in src/.
6. API COMPATIBILITY: Use only well-documented, standard function parameters.
   Wrap risky API calls in try/except with a fallback.
7. SAFE DATA LOADING: Use np.load(..., allow_pickle=True) and check ndim
   before calling .item().
8. Sign and hyperparameter correctness: Follow the plan EXACTLY.
   If plan says "x_new = x_old - tau * grad", use subtraction, not addition.
9. PYTHON 3.9 COMPATIBILITY: Use `Optional[X]` from typing, NOT `X | None`.
   Use `Union[X, Y]` from typing, NOT `X | Y`. Use `list[X]` or `List[X]`.
10. DATA KEY NAMES: Use the EXACT key names shown in the data inventory.
    Do NOT guess key names — always check the data inventory for exact names.
11. IMPORT CONSISTENCY: Only import names (classes, functions) from src/ modules
    that are listed in the "ALL MODULE INTERFACES" section below. Do NOT invent
    imports from modules or symbols that do not exist in the architecture.
12. SCIPY OPTIMIZER CONSTRAINTS: scipy.optimize.minimize with method='L-BFGS-B'
    only supports box bounds via `bounds=`. Do NOT pass `constraints=` to L-BFGS-B
    (it will be silently ignored). Enforce equality constraints via projection
    or penalty terms in the objective instead.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        target = context.get("target_file", "unknown file")
        plan = context.get("plan", "No plan provided")
        skeleton = context.get("skeleton", "No skeleton provided")
        full_architecture = context.get("full_architecture", "")
        current_code = context.get("current_code", "")
        feedback = context.get("feedback", "")
        data_inventory = context.get("data_inventory", "")
        requirements = context.get("requirements", "")

        prompt = f"### TARGET FILE: {target}\n\n"
        prompt += f"### IMPLEMENTATION PLAN\n{plan}\n\n"
        prompt += f"### CODE ARCHITECTURE (Skeleton for this file)\n{skeleton}\n\n"

        if full_architecture:
            prompt += (
                f"### ALL MODULE INTERFACES (imports & signatures across ALL files)\n"
                f"{full_architecture}\n"
                "⚠️ ONLY import names that actually exist in the above interfaces. "
                "Do NOT invent imports from modules or classes not shown here.\n\n"
            )

        if requirements:
            prompt += (
                f"### AVAILABLE PACKAGES (STRICT — no others installed)\n{requirements}\n"
                "⚠️ ONLY use these packages. Do NOT import jax, torch, tensorflow.\n\n"
            )

        if data_inventory:
            prompt += f"### AVAILABLE DATA FILES\n{data_inventory}\n\n"

        if current_code:
            prompt += f"### CURRENT CODE (to fix/improve)\n```python\n{current_code}\n```\n\n"

        if feedback:
            fb_str = feedback
            if isinstance(feedback, dict):
                fb_str = feedback.get("analysis") or feedback.get("feedback") or str(feedback)
            prompt += f"### FEEDBACK TO ADDRESS\n{fb_str}\n\n"

        prompt += (
            "Output ONLY the complete Python file content for the target file.\n"
            "CRITICAL: Write FULL implementation. No placeholders. No pass statements.\n"
            "Ensure the code is syntactically correct Python."
        )
        return prompt

    @staticmethod
    def strip_markdown(code: str) -> str:
        """Remove markdown code fences from LLM output."""
        # Try to extract from ```python ... ``` blocks
        matches = re.findall(r"```(?:python)?\n(.*?)\n```", code, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        # Fallback: strip simple fences
        code = re.sub(r"^```\w*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
        return code.strip()

    @staticmethod
    def extract_python(text: str) -> str:
        """Extract Python code from LLM response, handling markdown wrapping."""
        code = text.strip()
        matches = re.findall(r"```python\s*\n(.*?)\n?\s*```", text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        matches = re.findall(r"```\s*\n(.*?)\n?\s*```", text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        # Heuristic: find first line starting with import/from/class/def
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ", "class ", "def ", "@", "#")):
                return "\n".join(lines[i:]).strip()
        return code
