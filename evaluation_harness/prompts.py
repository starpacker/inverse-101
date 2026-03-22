"""Prompt templates for each evaluation mode."""

# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a computational imaging expert implementing a reconstruction pipeline.
You work inside a Linux container at /workspace.
Available files: data/ (observations), README.md (problem description), \
evaluation/ (fixtures and tests).

You can take these actions:

1) WRITE_FILE — create or overwrite a file
   Action: WRITE_FILE
   Path: <file path relative to /workspace>
   Content:
   <file content — include the ENTIRE file, not just a snippet>
   END_CONTENT

2) RUN — execute a shell command
   Action: RUN
   Command: <shell command>

3) READ_FILE — read a file
   Action: READ_FILE
   Path: <file path>

4) DONE — signal that you have finished
   Action: DONE

Respond in this exact format every turn:
Thought: <your reasoning>
Action: <one of WRITE_FILE | RUN | READ_FILE | DONE>
<action arguments as shown above>

Important rules:
- Write ONE action per response. Wait for the Observation before continuing.
- Always write complete files, never partial patches.
- After writing a source file, run its unit tests to verify.
- If tests fail, read the error, fix the code, and re-run.
"""


# ---------------------------------------------------------------------------
# Plan-generation mode
# ---------------------------------------------------------------------------

def plan_approach_prompt(readme: str, meta_data: str) -> str:
    return f"""\
Read the problem description and data specification below, then write a \
solution approach document to plan/approach.md.

The document should include:
1. Problem statement (what we recover, from what measurements)
2. Mathematical formulation (forward model equation)
3. Solution strategy (step-by-step algorithmic approach)
4. Expected results (which methods, expected quality)

== Problem Description ==
{readme}

== Data Specification (meta_data) ==
{meta_data}

Begin by reading any additional data files if needed, then write plan/approach.md.
"""


def plan_design_prompt(readme: str, approach: str) -> str:
    return f"""\
Based on the problem description and the solution approach below, write a \
code design document to plan/design.md.

The document should include:
1. File structure (which src/*.py files)
2. Function signatures with full type annotations and docstrings
3. Class definitions where appropriate
4. Data flow diagram (text-based)

== Problem Description ==
{readme}

== Solution Approach ==
{approach}

Write plan/design.md with complete function signatures for every function \
that needs to be implemented.
"""


# ---------------------------------------------------------------------------
# Function-level mode
# ---------------------------------------------------------------------------

def function_prompt(
    readme: str,
    approach: str,
    design: str,
    target_function: str,
    test_file_content: str,
) -> str:
    return f"""\
Implement the function specified below. Context is provided so you understand \
the overall pipeline, but you only need to implement the target function.

== Problem Description ==
{readme}

== Solution Approach ==
{approach}

== Code Design (full signatures) ==
{design}

== Target Function ==
{target_function}

== Unit Tests (your implementation will be tested against these) ==
{test_file_content}

Steps:
1. Read any relevant data/fixture files to understand inputs and outputs.
2. Write the source file containing the target function.
3. Run the unit tests to verify.
4. Fix any failures, then signal DONE.
"""


# ---------------------------------------------------------------------------
# End-to-end mode
# ---------------------------------------------------------------------------

def end_to_end_plan_prompt(readme: str, meta_data: str) -> str:
    """Phase 1 of end-to-end: generate approach + design."""
    return f"""\
You will implement a full computational imaging pipeline from scratch.

First, create the solution plan:
1. Write plan/approach.md — solution strategy with math formulation
2. Write plan/design.md — code architecture with all function signatures

== Problem Description ==
{readme}

== Data Specification ==
{meta_data}

Start by reading any data files to understand the problem, then write both \
plan documents. Signal DONE when both files are written.
"""


def end_to_end_impl_prompt(approach: str, design: str) -> str:
    """Phase 2 of end-to-end: implement all source files."""
    return f"""\
Implement the full pipeline following the plan below.

== Solution Approach ==
{approach}

== Code Design ==
{design}

Implementation order:
1. Run: ls evaluation/tests/ — to discover which test files exist.
2. Write src/__init__.py (empty module marker).
3. For each source module listed in the Code Design, implement it and run
   the corresponding test file. For example, if the design lists
   src/preprocessing.py, run: python -m pytest evaluation/tests/test_preprocessing.py -v
4. Write main.py and run: python main.py
5. Final check: python -m pytest evaluation/tests/ -v

After writing each file, run its tests and fix any failures before proceeding \
to the next file. Signal DONE after all tests pass.
"""
