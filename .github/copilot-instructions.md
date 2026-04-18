## Build and test
- Run `pytest` after non-trivial code changes.
- If formatting tools are configured, run them before finalising changes.

## Code style
- Prefer explicit variable names for aerodynamic and geometric quantities.
- Preserve units, reference frames, and sign conventions.
- Keep solver logic readable and testable.

## Workflow
- Do not silently change numerical behaviour.
- When outputs change, explain what changed and why.
- Add or update regression tests for solver changes.