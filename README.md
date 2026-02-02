# C9 Draft Assistant — Hackathon Judge README

## Quick Start
1. Create virtual environment: `python -m venv .venv`
2. Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)
3. Install: `pip install -r requirements.txt`
4. Run: `streamlit run app.py` (backend optional for demo)

## Project overview
C9 Draft Assistant is a real-time drafting assistant for *League of Legends* pick/ban phases. It helps a coach or analyst make faster, more consistent draft decisions by ranking the best next picks or bans for the current draft state and explaining the key reasons behind each recommendation.

The demo is designed to be easy to evaluate: you can run a Streamlit UI, step through a full draft, and see the system react instantly to every pick/ban. The backend exposes a small FastAPI surface area for draft state and recommendations.

## What problem this solves
Drafting is a high-pressure, time-constrained decision problem with a huge combinatorial space. Even experienced staff can miss a strong denial ban, overlook a high-synergy pairing, or fail to adapt quickly when the opponent pivots.

This project solves that by providing:
- **Fast ranked suggestions** for the next legal pick/ban in the current state
- **Context-aware reasoning** (policy patterns, comfort/denial, synergy, counters, and patch strength)
- **A repeatable workflow** so recommendations are consistent across drafts and staff

## High-level system architecture
- **Streamlit UI (`app.py`)**: interactive draft flow, auto-play, and explanation display.
- **FastAPI service (`api/`)**: endpoints for draft state updates and fetching recommendations.
- **Active inference path (`draft/`, `model/`, `inference/`)**: live demo inference using trained artifacts, scores all legal actions, returns ranked recommendations with explanations.
- **ML training path (`ml/`, `core/`)**: model training code, feature engineering, and ML-specific draft state representations.
- **Services layer (`services/`, `core/`)**: draft rules, recommendation engine, and shared configuration.
- **Data layer (`data/`, `scripts/`)**: ingestion and preprocessing utilities (built around the GRID Esports API).

**Note**: The codebase maintains separation between active inference (used by the live demo) and ML training code. Both paths share configuration from `core/config.py` but use independent draft state implementations for architectural clarity.

## How the model works
At each step, the system considers every **legal** pick or ban and assigns each one a score. The final ranking is a weighted blend of several intuitive signals:
- **Pro draft policy**: what professional drafts tend to pick/ban in similar situations (primary driver).
- **Meta/patch strength**: champions that are currently strong and frequently present in drafts.
- **Synergy**: champions that work well with the team's existing picks.
- **Counter**: options that punish the opponent's picks.
- **Comfort**: player/team tendencies (e.g., champions a player has historically played a lot, or champions teams often ban against a particular team).

The UI surfaces the top recommendations and provides a short explanation highlighting the most important contributors for that specific state.

## How to run the project from zero

### Prerequisites
- **Python 3.13** (recommended) or Python 3.11+
- Git (to clone the repository)

### Step-by-step setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jlandrygit/Cloud9DraftAssistant-JamesLandry
   cd Cloud9DraftAssistant-JamesLandry
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the environment**
   
   **Windows (PowerShell)**:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   
   If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   
   **Windows (Command Prompt)**:
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   **macOS/Linux**:
   ```bash
   source .venv/bin/activate
   ```

4. **Upgrade pip** (recommended)
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all required packages including FastAPI, Streamlit, PyTorch, and ML libraries.

6. **Verify installation** (optional but recommended)
   ```bash
   python verify_setup.py
   ```
   
   This will check that all critical files and dependencies are present.

### Running the demo

**Note**: The demo works without a `.env` file. All required data files and model checkpoints are included in the repository.

**Important**: The Streamlit UI can run standalone for the core drafting assistant. The FastAPI backend is optional and only needed for the `/health/demo` endpoint and full API access.

1. **Start the FastAPI backend** (optional, in one terminal)
   ```bash
   uvicorn api.main:app --reload
   ```
   
   You should see:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```

2. **Start the Streamlit UI** (in a second terminal, with the virtual environment activated)
   ```bash
   streamlit run app.py
   ```
   
   The UI will automatically open in your browser at `http://localhost:8501`.

### Optional: Environment variables

If you want to run data ingestion scripts or use GRID-backed workflows, create a `.env` file in the project root:

```bash
GRID_API_KEY=your_grid_api_key_here
MODEL_PATCH_START=25.16
MODEL_PATCH_END=26.2
```

**Note**: The demo works without these variables. They are only needed for data ingestion.

## How to use the demo UI

1. **Select teams** in the left sidebar:
   - Choose Blue Side Team (default: C9)
   - Choose Red Side Team (your opponent)

2. **Step through the draft**:
   - Search for a champion in the search box
   - Click **Confirm Pick** or **Confirm Ban** when ready
   - The draft will advance automatically

3. **Review recommendations** on the right panel:
   - Top 5 suggestions are shown with confidence levels
   - Click the expander to see detailed explanations
   - Each recommendation shows score breakdowns (policy, comfort, counter, synergy, meta)

4. **Auto-play mode**:
   - Enable "Auto-play Blue Side" or "Auto-play Red Side" to simulate automatic picks/bans
   - Useful for quickly seeing a full draft

5. **Series mode**:
   - After completing a draft, click "Save Draft" to add picked champions to "Fearless Bans"
   - These champions will be automatically banned in subsequent drafts (simulating a best-of series)

## Troubleshooting

### Import errors
If you see import errors, ensure:
- The virtual environment is activated (you should see `(.venv)` in your terminal prompt)
- All dependencies are installed: `pip install -r requirements.txt`
- You're running commands from the project root directory

### Port already in use
If port 8000 or 8501 is already in use:
- **FastAPI**: Change the port: `uvicorn api.main:app --reload --port 8001`
- **Streamlit**: Change the port: `streamlit run app.py --server.port 8502`

### Model files not found
The demo includes all required model checkpoints and data files. If you see errors about missing files:
- Ensure you cloned the full repository (including the `models/` and `data/` directories)
- Check that `models/policy_checkpoints/` contains `.pt` files
- Check that `data/meta/` contains CSV files

### Windows PowerShell execution policy
If you get "cannot be loaded because running scripts is disabled" when activating the virtual environment:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## What Makes This Special
- **Real-time explainability**: Every recommendation shows *why* it's suggested with score breakdowns
- **Multi-signal fusion**: Combines 5 different data sources (policy, comfort, counter, synergy, meta) - not just winrates
- **Opponent-aware**: Adapts recommendations based on opponent team selection and player comfort
- **Instant recommendations**: Generates suggestions in real-time (<100ms per step)

## Known limitations
- **Data coverage**: results depend on available historical data and the coverage of the ingestion pipeline.
- **Data Longevity**: Data is a snapshot of the current meta and patch history rather than a live model.
- **UI is functional, not final**: Streamlit is great for demos but not ideal for a coach-facing production tool.

## What we would do next
- **Deeper team composition analysis**: reason about win conditions, damage profiles (AD/AP), engage/disengage, scaling, objective control, and draft "shape" over time.
- **More advanced counter-pick modeling**: role- and matchup-aware counter scoring, lane matchup prediction, and conditional counters (counter depends on team context).
- **Cleaner, more intuitive frontend**: a dedicated web UI with faster interactions, clearer state visualization, and better coach workflow (e.g., pinboards, scenario planning, and draft timelines).
- **Production hardening**: confidence calibration, monitoring, multi-worker state storage, and a model registry for versioned model metadata.
