# Release Checklist

Status assessment for releasing imaging-101 as a public benchmark.

---

## 1. GitHub Repository (`HeSunPU/imaging-101`)

### Ready

- [x] **58 tasks** across 6 domains, all with `main.py`, `data/`, `src/`, `evaluation/`
- [x] **57/58 tasks** have complete structure (README, plan, notebooks, tests, reference_outputs)
- [x] All 58 tasks have unit tests (90 test files total)
- [x] Evaluation harness with 4 agent frameworks (ReAct, Multi-Agent, Copilot, DeepCode)
- [x] 3 evaluation modes (function, end_to_end, plan) with 3 difficulty levels (L1-L3)
- [x] Function-mode results for 7 models x 51 tasks
- [x] End-to-end results: Claude Code (50+ tasks), Multi-Agent (16 tasks), Copilot (21 tasks)
- [x] `docs/EVALUATION_GUIDE.md` — step-by-step evaluation tutorial
- [x] `docs/NEW_TASK_GUIDE.md` — how to add new tasks
- [x] Minimal dependencies (only `requests>=2.28`)

### Action Required

- [ ] **Remove API keys/tokens** from `config_llm.yaml` and `CLAUDE.md`
  - `config_llm.yaml`: contains `sk-Zj3a7RQDVCXr-Axg-0gtkg` (7 entries)
  - `CLAUDE.md`: contains GitHub PAT (must be removed)
  - Replace with placeholder `"YOUR_API_KEY"` / `"YOUR_TOKEN"`

- [ ] **Merge upstream** — local is 255 commits behind `origin/main`
  - Blocked by LFS pointer mismatches on Windows
  - **Fix**: Run merge from Linux/WSL: `git lfs pull && git merge origin/main`
  - Or: `GIT_LFS_SKIP_SMUDGE=1 git clone ... && git merge`

- [ ] **Remove large/temp files** from repo root:
  - `tasks.zip` (1.1 GB) — should be .gitignored
  - `__pycache__/` directories
  - `_build_appc_figures.py`, `_gen_appC_overleaf.py`, etc. (paper build scripts) — move to `scripts/paper/` or remove
  - `_appc_inventory.json` (76 KB)
  - `.claude/`, `.pytest_cache/`, `.vscode/`

- [ ] **Clean `eht_black_hole/`** — legacy duplicate of `eht_black_hole_original`; consider removing

- [ ] **Update README.md** to match current state:
  - Confirm task count (57 or 58?)
  - Add links to `docs/EVALUATION_GUIDE.md` and `docs/NEW_TASK_GUIDE.md`
  - Add leaderboard table or link to website

- [ ] **Add LICENSE file** (currently none)

- [ ] **Add .gitignore entries** for:
  ```
  tasks.zip
  __pycache__/
  *.pyc
  .claude/
  .vscode/
  _overleaf_upload/
  _zip_extract/
  ```

---

## 2. Evaluation Results

### Ready

- [x] Function-mode audit CSV: 57 tasks x 7 models (406 rows)
- [x] Claude Code e2e audit CSV: 58 tasks
- [x] Per-model success summaries with detailed pass/fail breakdowns
- [x] 191 comparison figures in `figs/appC/`

### Action Required

- [ ] **Consolidate result directories** — currently scattered across:
  - `results/function_mode/` (41 tasks, early runs)
  - `results/function_mode_3/` (15 tasks, subset)
  - `results/function_mode_total/` (51 tasks x 7 models — primary data)
  - `results/function_mode_server/` (52 tasks)
  
  **Recommendation**: Keep only `results/function_mode_total/` as the canonical function-mode results. Archive or remove the others.

- [ ] **Consolidate e2e results**:
  - `results/end_to_end/copilot/` (21 tasks)
  - `results/end_to_end/multi_agent/` (16 tasks)
  - `results/end_to_end/react/` (2 tasks)
  - `results/end_to_end/claude_code/` (50+ tasks)
  
  **Recommendation**: Keep all frameworks as separate directories (this is organized by framework, which makes sense).

- [ ] **Decide what to publish** — results are 2.1 GB total. Options:
  1. Include audit CSVs + summaries only (< 1 MB) in the repo; host full results on HuggingFace
  2. Include everything in a separate `results` branch
  3. Separate results repo

---

## 3. Website (`starpacker/agent-imaging-website`)

### Ready

- [x] Next.js + TypeScript + Tailwind CSS
- [x] Task browser with 57 task notebooks
- [x] Per-task comparison images
- [x] CI/CD deploy pipeline (`.github/workflows/deploy.yml`)

### Action Required

- [ ] **Resolve repo location** — currently says "[MOVED] to starpacker/inverse_benchmark/website/" but still receiving pushes. Pick one location.

- [ ] **Add leaderboard page** — reference designs:
  - SWE-bench (https://www.swebench.com/) — closest to your use case
  - HELM (https://crfm.stanford.edu/helm/) — multi-dimensional comparison
  - Aider (https://aider.chat/docs/leaderboards/) — clean pass-rate tables

- [ ] **Add methodology page** — explain evaluation modes, metrics, difficulty levels

- [ ] **Add "Submit Results" page** — let others submit their model results (reference: HF Open LLM Leaderboard)

- [ ] **Add domain breakdown visualization** — radar chart or grouped bar chart showing per-domain performance (reference: LiveBench)

- [ ] **Update task_db.json** with latest 58 tasks and evaluation results

---

## 4. HuggingFace Dataset (`AI4Imaging/imaging-101`)

### Action Required

- [ ] **Decide visibility**: currently private. Must be public for open benchmark.
- [ ] **Upload latest fixtures** — many tasks have been updated since last upload
- [ ] **Add dataset card** (README.md on HuggingFace) explaining contents
- [ ] **Upload evaluation results** (audit CSVs, summaries) for reproducibility

---

## 5. Priority Order

### Must-do before release (blocking)

1. Remove API keys/tokens from committed files
2. Add LICENSE file
3. Merge upstream (fix LFS issue)
4. Clean repo root (remove tasks.zip, __pycache__, paper scripts)
5. Make HuggingFace dataset public (if fixtures are needed for evaluation)

### Should-do (high impact)

6. Consolidate results into clean structure
7. Update README with links to evaluation guide + new task guide
8. Resolve website repo location
9. Add leaderboard to website
10. Update .gitignore

### Nice-to-have

11. Add radar chart / domain breakdown to website
12. Add "Submit Results" workflow
13. Add methodology page to website
14. Publish results to a separate location (HuggingFace or results branch)
