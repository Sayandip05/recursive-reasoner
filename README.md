# Recursive Reasoner

A minimal scaffold for experimenting with multi-step, tool-using reasoning loops.

## Overview
- Goal: prototype and evaluate recursive reasoning pipelines (planning, tool calls, self-critique).
- Status: project scaffold — code files are stubs; fill in core logic in `src/`.

## Features (intended)
- Reusable reasoning loop with pluggable steps (prompting, parsing, acting, reflecting).
- Config-driven runs to compare strategies.
- Lightweight evaluation harness for scoring traces.

## Project Layout
- `src/config.py` — configuration schema/defaults.
- `src/prompts.py` — prompt templates and system messages.
- `src/reasoning.py` — main reasoning loop and step orchestration.
- `src/utils.py` — shared helpers (logging, I/O).
- `src/evaluate.py` — run tasks, capture traces, compute metrics.

## Getting Started
1) Clone the repo.
2) Create a virtualenv: `python -m venv .venv && .\.venv\Scripts\activate`.
3) Install deps: populate `requirements.txt` then `pip install -r requirements.txt`.
4) Add your OpenAI/LLM keys to environment variables if needed.

## Usage (future)
- Run experiments: `python -m src.evaluate --config configs/basic.yaml`
- Tweak prompts in `src/prompts.py`; adjust loop depth/termination in `src/reasoning.py`.

## Development Notes
- Keep functions pure where possible for easier testing.
- Log every reasoning step to make evaluation reproducible.
- Add docstrings and type hints once logic is in place.

## Testing (to add)
- Unit tests for prompt assembly and parsing.
- Golden trace tests to guard against regression in reasoning outputs.

## Roadmap
- Implement base reasoning loop.
- Add tool-call plugins (search, calculator, code runner).
- Build small benchmark suite with pass/fail tasks.
- Wire up telemetry/trace viewer.

## License
MIT (or add your preferred license).
