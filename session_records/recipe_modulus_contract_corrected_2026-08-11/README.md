# Explicit per-slot modulus correction

This record supersedes the earlier interpretation that a two-valued toggle
demonstrated a missing pair primitive. That interpretation was wrong:
`INC(i, m=2); INC(j, m=2)` already expresses the toggle.

The actual defect was a hidden global modulus of eight. For slot domains
`(2, 2, 8, 8, 8, 8)`, a legacy modulus-8 increment matches the correct
family increment at per-slot rates `[0.5, 0.5, 1.0, 1.0, 1.0, 1.0]`.
The corrected `RecipeInstruction` ABI carries `(op, i, j, m)` for arithmetic
operations, and `RecipeBasis` accepts explicit per-slot value domains. The
runtime rejects arithmetic instructions whose modulus does not match the
target slot domain.

The correction is covered by deterministic tests for exact two-valued toggle
composition, modulus mismatch rejection, modulus-bearing candidate
generation, mixed-domain random-program execution, and learned feature
encoding. The 50-update smoke run was only an execution check; it is not a
capability promotion. The required learned result is a two-seed mixed-domain
audit with a diagnostic legacy-global-modulus comparison.

Implementation and evidence:

- `src/neural_computer/recipe_basis.py`
- `experiments/recipe_expressibility/audit.py`
- `tests/test_recipe_basis.py`
- `tests/test_recipe_expressibility_audit.py`
- `docs/RECIPE_EXPRESSIBILITY.md`
