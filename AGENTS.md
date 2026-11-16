# MyAiVoice — Agent Instructions

## Language
Use English exclusively for all agent communication and generated artifacts. This includes progress updates, diagnostic reports, sub-agent communication, questions, code comments, documentation, and completion reports. Do not use Chinese or another language unless explicitly requested.

## Conventions
- All work happens in this workspace (`/Users/m10/Desktop/MyAiVoice`).
- The Python virtual environment is at `.venv/`; use `./.venv/bin/python` for tests and scripts.
- Tests live in `tests/` and run with `./.venv/bin/python -m pytest -v`.
- The application is a Streamlit app; the entry point is `app/main.py`.
- Database migrations must be transactional: `init_db()` wraps the v2 migration in a `BEGIN IMMEDIATE` transaction that commits only after all statements, including the schema-version row, have succeeded.
