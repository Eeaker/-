# Backend Project Structure

## Core Entry
- `app.py`: FastAPI startup and route registration.
- `runtime_paths.py`: unified runtime directories (`uploads`, `game_analysis`, `trajectory`, etc.).

## Engines
- `reid_engine.py`: player retrieval feature extraction/search.
- `video_engine.py`: player-focused highlight generation.
- `trajectory_engine.py`: shot trajectory analysis and prediction.
- `game_analysis_engine.py`: unified game-analysis orchestration.
- `third_party_basketball_adapter.py`: integration adapter for external basketball tracking repo.

## API Routes
- `routes/auth_routes.py`: auth/login APIs.
- `routes/dashboard_routes.py`: dashboard/stats APIs.
- `routes/reid_routes.py`: image search APIs.
- `routes/video_routes.py`: video highlight APIs.
- `routes/trajectory_routes.py`: trajectory task APIs.
- `routes/game_analysis_routes.py`: unified game-analysis APIs.

## Runtime Data (generated)
- `uploads/`: generated artifacts only (task videos, exports, stubs).
- `ballshow.db`: runtime SQLite database.

## Maintenance
- `tools/project_maintenance.py`: one-command cleanup utility.
  - `python tools/project_maintenance.py --all`
  - Clears runtime task tables, generated uploads, temp/test artifacts.

