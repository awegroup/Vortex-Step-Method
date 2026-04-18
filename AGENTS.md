# AGENTS.md

## Repository purpose

This repository implements aerodynamic analysis tools based on the Vortex Step Method (VSM), with emphasis on physically consistent, numerically robust, and testable scientific software.

The code is used for research and engineering work. Correctness, traceability, and reproducibility are more important than clever abstractions or large stylistic rewrites.

---

## General principles

1. Preserve physical meaning.
   - Do not silently change sign conventions, reference frames, force directions, circulation definitions, or coefficient definitions.
   - If a change affects the physical model, state clearly what changed and why.

2. Prefer small, verifiable changes.
   - Make the minimum change needed to solve the task.
   - Avoid broad refactors unless explicitly requested or clearly necessary.

3. Prioritise numerical robustness.
   - Be careful with near-singular geometries, very small denominators, ill-conditioned solves, and convergence loops.
   - Prefer explicit safeguards over hidden failure.

4. Keep behaviour stable unless change is intentional.
   - If outputs are expected to change, explain:
     - which outputs change,
     - why they change,
     - whether the change is physical, numerical, or both.

5. Keep the code understandable.
   - Use explicit names for aerodynamic and geometric quantities.
   - Prefer simple, inspectable logic over overly abstract patterns.

---

## What matters most in this repository

When working in this repository, optimise for:

- physical correctness,
- numerical stability,
- reproducibility,
- testability,
- clarity for future research use.

Optimise less for:

- excessive abstraction,
- clever one-liners,
- premature micro-optimisation,
- stylistic churn without technical value.

---

## Expectations for code changes

### Solver and aerodynamic logic

When editing solver logic, geometry processing, circulation iteration, induced velocity evaluation, or force reconstruction:

- identify the governing equations or assumptions being modified,
- preserve units and dimensional consistency,
- check whether the change affects:
  - circulation convergence,
  - lift/drag calculation,
  - induced velocity evaluation,
  - geometric transforms,
  - wake or filament definitions,
  - control point definitions,
  - local reference frames.

If the change alters algorithmic behaviour, document the expected effect in the PR/commit message or task summary.

### Refactoring

Refactors are welcome only if they improve one or more of:

- correctness,
- clarity,
- testability,
- modularity without changing behaviour.

Avoid refactors that mix structural cleanup with physics changes unless explicitly requested.

### Bug fixing

When fixing a bug:

- identify the root cause,
- describe whether it is:
  - physical-model bug,
  - numerical bug,
  - implementation bug,
  - API or interface bug,
  - test bug.
- add a regression test when possible.

---

## Testing expectations

For any non-trivial change, add or update tests.

Especially add tests when changing:

- circulation solver logic,
- vortex-induced velocity routines,
- geometry rotation or transformation logic,
- aerodynamic coefficient calculation,
- force and moment reconstruction,
- panel/segment discretisation behaviour,
- convergence criteria,
- handling of degenerate geometries.

Preferred test types:

1. Regression tests
   - Verify known outputs do not change unexpectedly.

2. Invariant tests
   - Example: symmetry, zero-lift conditions, expected force direction, consistent dimensions.

3. Analytical or benchmark comparisons
   - Use simple cases with known or trusted behaviour where possible.

4. Failure-mode tests
   - Near-singular geometry,
   - zero or very small inflow components,
   - repeated points,
   - extreme discretisations.

If a task changes results intentionally, update tests to reflect the new expected behaviour and explain why.

---

## Numerical safety guidelines

Be cautious with:

- division by very small norms,
- normalisation of nearly zero vectors,
- cross products of nearly aligned vectors,
- angle computations near domain limits,
- matrix solves on poorly conditioned systems,
- convergence checks based only on absolute tolerance,
- hidden broadcasting or shape mismatches.

Preferred practices:

- guard small denominators explicitly,
- fail loudly when inputs are invalid,
- use tolerances consistently,
- keep iteration logic inspectable,
- avoid silently clipping unless justified and documented.

---

## Scientific software style

### Code style

- Prefer explicit over implicit operations.
- Use descriptive variable names for aerodynamic quantities.
- Keep functions focused.
- Separate:
  - geometry generation,
  - aerodynamic evaluation,
  - solver iteration,
  - post-processing,
  - plotting or I/O.

### Comments and docstrings

Add comments where the code reflects a physical or numerical assumption that may not be obvious.

Good comments explain:
- why something is done,
- what assumption is being used,
- what convention is being followed.

Avoid comments that merely restate the code.

### Units and conventions

Be explicit about:
- coordinate systems,
- angle conventions,
- radians vs degrees,
- dimensional vs non-dimensional quantities,
- sign conventions for lift, drag, moments, circulation, and normal directions.

If a function depends on a convention, state it in the docstring.

---

## Performance guidance

Performance matters, but not at the cost of correctness.

When improving performance:

- preserve output behaviour unless explicitly changing the method,
- benchmark before and after if the change is significant,
- prefer simple vectorisation over opaque optimisation,
- avoid making the code harder to validate.

If a slower but more robust method is introduced, note that trade-off explicitly.

---

## What to avoid

Avoid:

- changing physics and refactoring in the same step without saying so,
- introducing hidden defaults that alter behaviour,
- renaming many variables without clear benefit,
- adding dependencies without strong justification,
- suppressing numerical warnings without understanding them,
- replacing understandable scientific code with generic architecture patterns.

Do not assume:
- reference-frame conventions,
- sign conventions,
- coefficient definitions,
- wake definitions,
- or geometry ordering.

Check them explicitly from the local code.

---

## Preferred workflow for substantial tasks

For substantial changes, follow this order:

1. Understand the local model assumptions and conventions.
2. Identify the smallest safe change.
3. Implement the change clearly.
4. Add or update tests.
5. Summarise:
   - what changed,
   - why,
   - expected impact on outputs,
   - validation performed.

---

## Guidance for common task types

### If asked to debug
Focus on:
- wrong signs,
- frame inconsistencies,
- geometry ordering,
- unstable iteration,
- singular induced-velocity cases,
- mismatched array shapes,
- bad assumptions at low speed or degenerate geometry.

### If asked to refactor
Preserve outputs.
Do not alter equations or conventions unless explicitly requested.

### If asked to extend the model
State clearly:
- what new assumption is added,
- what existing assumption is relaxed,
- what tests or benchmarks support the extension.

### If asked to write documentation
Document:
- equations,
- conventions,
- assumptions,
- limitations,
- expected valid operating range.

---

## Final rule

When in doubt, prefer:
- explicitness over cleverness,
- validation over speed,
- local fixes over broad rewrites,
- physical consistency over stylistic purity.