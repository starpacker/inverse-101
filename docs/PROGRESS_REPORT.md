# Progress Report — Multi-Agent Framework Improvements

> **Task**: `eht_black_hole_original` (end-to-end mode)
> **Model**: `gemini-2.5-pro`
> **Red Line**: All changes must be general-purpose. No task-specific tuning.

---

## Version History

### v7 — Baseline (2026-03-29)

**Best Result**: NRMSE=0.7355, NCC=0.694, PSNR=20.50, SSIM=0.5617 (1 iteration, 11 calls)

**Results Distribution** (all v7 multi_agent runs):
| Run | NRMSE | NCC | PSNR | SSIM | Calls |
|-----|-------|-----|------|------|-------|
| Best | 0.7355 | 0.694 | 20.50 | 0.5617 | 11 |
| 2nd | 0.7383 | 0.7013 | 20.46 | 0.545 | 99 |
| 3rd | 0.9397 | 0.344 | 18.37 | 0.1451 | 9 |
| 4th | 0.9705 | 0.2409 | 18.09 | 0.0548 | 63 |
| 5th | 1.0 | 0.0 | 17.83 | 0.0052 | 26 |
| 6th | 1.0379 | 0.1002 | 17.50 | 0.0163 | 10 |
| 7th | 1.0667 | 0.1137 | 17.27 | 0.0442 | 29 |

**Known Issues**:
1. Runtime off-by-one array indexing errors
2. Optimization may not be sufficient (single round, few iterations)
3. No self-verification of forward model consistency
4. Missing/inappropriate regularization
5. Pipeline stops at first "success" (exit code 0) even if quality is poor

---

### v8 — Prompt Improvements: Numerical Rigor + Multi-Round Optimization + Enhanced Critic (2026-03-29)

**Changes** (ALL general-purpose, zero task-specific content):

1. **Coder Agent — Numerical Computing Safety Rules** (`coder_agent.py`)
   - Added rules 13-18 to system prompt:
     - Array indexing safety (clip indices, verify bounds)
     - Numerical stability (epsilon denominators, safe log/exp)
     - Gradient verification (commented-out finite difference check)
     - Shape assertions at pipeline stages
     - Output validation (2D, finite, non-constant)
     - Optimizer convergence monitoring (maxiter≥200, disp=True, check success)
   - **General-purpose?** ✅ These rules apply to ANY scientific computing code.

2. **Planner Agent — Multi-Round Optimization Strategy** (`planner_agent.py`)
   - Added guidelines 9-11 to system prompt:
     - Coarse-to-fine optimization (3 rounds with decreasing regularization)
     - Mandatory regularization (TV, L2, or entropy with tunable λ)
     - Convergence monitoring (print cost every 50 iterations)
   - **General-purpose?** ✅ Standard inverse problem methodology.

3. **Critic Agent — Enhanced Checklist** (`planner_agent.py`)
   - Expanded from 4 to 9 checklist items:
     - Added: Regularization present, sufficient iterations (≥300), numerical stability,
       hyperparameters complete, output format check
   - **General-purpose?** ✅ All items are standard engineering review criteria.

**Results**:
| Run | NRMSE | NCC | PSNR | SSIM | Iters | Calls | Notes |
|-----|-------|-----|------|------|-------|-------|-------|
| 1 | 0.9425 | 0.3351 | 18.35 | — | 3 | 32 | Critic rejected 3x, Judge→Architect 2x |

**Analysis**: Worse than v7 best. Key issues:
1. Critic too strict — rejected plan 3 times for minor issues (epsilon guards), wasting Planner calls
2. Judge routed to Architect for KeyError (should route to Coder) — caused full file regeneration
3. Optimizer converged in only 0-2 iterations (gradient likely wrong but not detected)

---

### v9 — Judge Routing Fix + Critic Leniency + Convergence Detection (2026-03-29)

**Changes** (ALL general-purpose):

1. **Judge Agent — Stronger Runtime Error Routing** (`judge_agent.py`)
   - Added rules 5-8: explicit list of runtime errors that MUST go to Coder (never Architect)
   - Rule 8: "NEVER assign to Architect for KeyError, TypeError, ValueError, IndexError"
   - **General-purpose?** ✅ Error taxonomy applies to all Python programs.

2. **Multi-Agent Pipeline — Runtime Error Override** (`multi_agent.py`)
   - Added hardcoded safety net after Judge decision: if Judge routes to Architect for
     a runtime error (KeyError, TypeError, etc.), override to Coder
   - **General-purpose?** ✅ Runtime errors are always code-body bugs, not architecture bugs.

3. **Critic Agent — Leniency Rebalance** (`planner_agent.py`)
   - Split checklist into CRITICAL (4 items, reject if missing) and DESIRABLE (4 items, note but don't reject)
   - Added explicit instruction: "Be LENIENT. Only REJECT if fundamentally flawed."
   - **General-purpose?** ✅ Standard review heuristic — fatal vs nice-to-have.

4. **Convergence Detection — L-BFGS-B Fortran Format** (`multi_agent.py`)
   - Enhanced `_check_optimizer_convergence` to parse L-BFGS-B Fortran summary table
   - Detects: fast convergence (≤5 iterations), variables stuck at bounds (>50%)
   - Routes directly to Coder with specific gradient-fix instructions (skips Judge)
   - **General-purpose?** ✅ Detects common scipy optimizer failures.

5. **Coder Agent — Mandatory Gradient Verification** (`coder_agent.py`)
   - Changed rule 15 from "commented-out" to "RUNTIME CHECK"
   - Coder now must include approx_fprime verification at start of optimization
   - Falls back to numerical gradient if analytical gradient is wrong
   - **General-purpose?** ✅ Standard numerical optimization best practice.

**Results**:
| Run | NRMSE | NCC | PSNR | SSIM | Iters | Calls | Notes |
|-----|-------|-----|------|------|-------|-------|-------|
| — | — | — | — | — | — | — | Running... |

---

## Target Metrics

| Metric | v7 Best | Mid-Term | Ultimate | Reference Best |
|--------|---------|----------|----------|----------------|
| NRMSE ↓ | 0.7355 | ≤ 0.60 | ≤ 0.30 | 0.2648 |
| NCC ↑ | 0.694 | ≥ 0.80 | ≥ 0.95 | 0.9669 |
| PSNR ↑ | 20.50 | ≥ 22 | ≥ 28 | — |
| SSIM ↑ | 0.5617 | ≥ 0.70 | ≥ 0.90 | — |
