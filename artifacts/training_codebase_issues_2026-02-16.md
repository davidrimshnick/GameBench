# Training codebase issue audit (2026-02-16)

## Scope
Reviewed training-related modules and ran targeted test/import checks.

## Findings

### 1) Training stack hard-fails when PyTorch is not installed
- `davechess/engine/network.py` defines `DaveChessNetwork` only inside `if HAS_TORCH:`.
- `davechess/engine/training.py` imports `DaveChessNetwork` at module import time.
- In an environment without torch, importing `davechess.engine.training` raises:
  - `ImportError: cannot import name 'DaveChessNetwork' from 'davechess.engine.network'`.

**Why this matters**
- The package currently degrades partially without torch, but training imports crash immediately rather than giving a clear runtime guidance message.

### 2) Unit tests fail in non-torch environments due to missing skip guard
- `tests/test_mcts.py::TestMultiprocessMCTS::test_multiprocess_selfplay_output_format` imports `DaveChessNetwork` unconditionally.
- In environments without torch, this test fails with ImportError instead of skipping.

**Why this matters**
- CI/dev test runs become noisy and non-actionable when optional dependencies are absent.

### 3) Default training network input-plane count is inconsistent with encoder
- `state_to_planes()` in `davechess/engine/network.py` returns **14** planes.
- `Trainer` in `davechess/engine/training.py` defaults to `input_planes=12` when config omits the field.

**Why this matters**
- Misconfigured or minimal configs can instantiate a model with incompatible input shape, causing runtime dimension errors.

### 4) `dev` extras do not include dependencies required by default tests
- `pyproject.toml` has `dev = ["pytest"]`.
- Test collection imports FastAPI modules (`tests/test_api_endpoints.py`, `tests/test_sdk.py`).
- Running `pytest -q` without `api` extras fails during collection with `ModuleNotFoundError: No module named 'fastapi'`.

**Why this matters**
- A default developer setup (`pip install .[dev]`) cannot run the repository's full test suite.

## Evidence: commands run

1. Full suite collection:
```bash
pytest -q
```
Output included:
- `ModuleNotFoundError: No module named 'fastapi'` from API-related tests.

2. Targeted non-API tests:
```bash
pytest -q tests/test_network.py tests/test_mcts.py tests/test_validation.py -x
```
Output included:
- Failure in `test_multiprocess_selfplay_output_format` with import error for `DaveChessNetwork`.

3. Direct training import check:
```bash
python - <<'PY'
try:
 import davechess.engine.training as t
 print('ok')
except Exception as e:
 print(type(e).__name__, e)
PY
```
Output:
- `ImportError cannot import name 'DaveChessNetwork' from 'davechess.engine.network' ...`

## Suggested next fixes (priority)
1. **High**: Add robust no-torch fallback behavior.
   - Define a stub `DaveChessNetwork` in no-torch mode that raises a clear message on initialization, or gate imports in `training.py` and fail with actionable guidance.
2. **High**: Align default `input_planes` with encoder output (14).
3. **Medium**: Add `pytest.importorskip("torch")` / conditional skip in torch-dependent tests.
4. **Medium**: Update dependency docs/extras so the default test workflow installs API test deps, or mark API tests to skip when FastAPI is unavailable.
