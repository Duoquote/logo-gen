# AGENTS.md

Notes for AI coding agents working in this repository. People can read it too. If you want to use logo-gen (make images through its API) rather than change it, read [llms.txt](llms.txt) instead.

logo-gen is a local web app for logo marks and app icons. The backend is Python (FastAPI) in `src/logo_gen/`, the frontend is React with TanStack Router and Query and shadcn/ui in `frontend/`, and they talk over a JSON API under `/api`. It works with OpenRouter cloud models and with local diffusion models.

## Setup and run

```sh
./launch.sh          # or .\launch.cmd on Windows: sets up uv, .venv and the frontend, then serves on :7860
./launch.sh --dev    # backend with reload, plus Vite with hot reload on :5173
```

The launcher logic is `scripts/launch.py` (stdlib only, run through `uv run --no-project`). It skips steps whose inputs didn't change, so a second run takes a few seconds. Local model support (PyTorch, diffusers) isn't installed until someone installs a local model or passes `--with-local`.

## Checks

| Command | What it checks |
|---|---|
| `uv run pytest` | Backend tests in `tests/`: chat turns that keep running when the browser leaves, branches, feedback events and what the designer learns from them, rendered prompts per model, caps and the rate limiter, rankings and the value formula, background removal math, and that `llms-full.txt` is current. They use a throwaway data folder (`tests/conftest.py`) and a fake model, so they cost nothing. |
| `cd frontend && bunx tsc -b && bunx eslint src && bun run build` | Types, lint and the production build. |
| `uvx ruff check --line-length 140 --select E,F,W src tests scripts` | Python lint. |
| `uv run python scripts/llms.py` | Regenerates `llms-full.txt` from the API. Run it after changing an endpoint, a request model or an endpoint docstring; `--check` only compares. |
| `uv run python scripts/update_rankings.py` | Refreshes the bundled leaderboard snapshot and fails if an alias in `rankings.py` points at a name that's gone. |

There's no CI yet, so run these yourself before you call something done.

## Verify in a real browser

Type checks don't catch a broken page. Start the app, drive it in Chrome (Playwright with the system Chrome works; nothing for it is committed), and look at screenshots:

- the flow you changed, plus Generate, Designer and Gallery still loading without console errors
- at 375px wide and in dark mode if you touched layout
- the no-GPU path if you touched anything device related: `LOGO_GEN_DEVICE=cpu ./launch.sh`
- for designer changes: send a message, leave the page while it answers, come back, and check the reply is still there and still streaming

Don't use or kill servers on ports you didn't start. Other people or agents may be running their own on the same machine.

## Hard rules

Don't change these without the maintainer asking for it:

- The server listens on 127.0.0.1 by default. The API has no authentication and can spend the user's OpenRouter credits, so don't change the default or add open CORS.
- Never print, log, commit or send anywhere an API key or token (`.env`, `OPENROUTER_KEY`, `HF_TOKEN`). The API only reports whether a key is set and its last four characters. `.env`, `data/`, `output/`, `.models/` and `.tools/` stay in `.gitignore`.
- Testing with OpenRouter costs real money. Keep test spend small (cents, not dollars), prefer the cheapest model that shows the behavior, and say what you spent. Spend is in `GET /api/spend`. Calibration and speed tests must never call paid APIs.
- Don't download anything over about 2 GB while testing (model weights, the CUDA PyTorch build) unless your user agreed to it. The tiny test pipelines are enough for most local-model code.
- Keep the no-GPU path working: cloud-only installs must not need PyTorch (local deps are lazy imports), and everything except local generation has to work on a CPU.
- Chat state lives on the server. A designer reply runs as a server task and is saved as it streams; the browser only follows it. Don't move that state back into the page.
- Spend is recorded from OpenRouter's response `usage`, including BYOK upstream cost. New paid calls need a `library.add_cost` row.

## Where things live

- `src/logo_gen/server.py`: every HTTP route. `jobs.py`: background jobs with progress and cancel. `generator.py`: turns prompts into images across cloud and local models, with caps and the adaptive rate limiter.
- `chats.py` and `turns.py`: designer chats as a message tree, and replies that run on the server and stream over SSE with resume.
- `prompt_engine.py`: the designer and prompt-writing system prompts, history building and reply parsing. The designer writes design briefs, not image prompts.
- `prompting/`: `brief.py` (the brief fields and JSON schema), `profiles.py` (how each image model family reads prompts, as data), `render.py` (brief to per-model prompt and request options) and `preferences.py` (what the designer learns from `feedback_events` and typed feedback). Rendered prompts have snapshot tests in `tests/snapshots/`; refresh them with `UPDATE_SNAPSHOTS=1 uv run pytest tests/test_prompting.py` after an intended change. `research/prompting-2026.md` has the sources behind the profiles.
- `library.py`: SQLite (`data/library.sqlite3`) for images, costs, chat tables and feedback events, with migrations in `_migrate`. `config.py`: settings and secrets.
- `clients/`: OpenRouter, local OpenAI-compatible servers and the Claude CLI. `local_models/`: the registry, downloads (`manager.py`, `worker.py`) and diffusers pipelines (`diffusion.py`). `rankings.py`: leaderboard and popularity data for model pickers.
- `postprocess.py` and `upscaler.py`: background removal and upscaling.
- `frontend/src/routes/`: file-based routes (`routeTree.gen.ts` is generated and committed). `frontend/src/components/chat/`: the designer UI. `frontend/src/lib/api.ts` has the API types, `queries.ts` the TanStack Query hooks.

## Code style

- Python: type hints, 4 spaces, lines up to 140. Comments say why, not what.
- TypeScript: the existing Prettier-like style (2 spaces, no semicolons, double quotes). Use the shadcn components in `frontend/src/components/ui/` before writing your own.
- User-facing text and docs are plain and short, in the first person for the maintainer's opinions and "we" for the project: no marketing words, no emoji, no em dashes, no exclamation marks. Claims about speed or cost come with numbers you measured.

## Pull requests

- One change per PR, in logical commits. A bug fix starts with a test that fails before the fix where that's practical.
- The checks above pass, and you checked the change in a browser. Put what you tested and what it cost in the PR description.
- Update the README and `llms.txt` when behavior changes, and regenerate `llms-full.txt`.
- Say in the PR description that an AI agent wrote or helped with the change, and which one. End commits with a `Co-Authored-By` line for the agent.
- Don't open an issue or a PR, merge, or push to the default branch on your own initiative. Ask the person you're working for first. Never force-push.
