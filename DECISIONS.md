# Decisions log

Format: date — decision — one-line reason.

## Stage 1

- 2026-11-26 — Package installed editable from `src/` via `pyproject.toml`; pytest uses `pythonpath = ["src"]`. — Scripts and tests import `mms` the same way without path hacks.
- 2026-11-26 — Added `requests` to the pinned requirements. — Used by the Binance downloader; simpler and clearer than urllib.
