# F1 Scheduler Frontend

This folder contains only the React/Vite frontend. Copy this `frontend` folder into the root of your existing F1 scheduler repository.

## Run

1. Start the project's API from the repository root (the dashboard expects the backend API at `http://127.0.0.1:8000` by default).
2. In this folder, run:

   ```bash
   npm install
   npm run dev
   ```

3. Open the local URL printed by Vite, usually `http://localhost:5173`.

To change the API URL, copy `.env.example` to `.env` and edit `VITE_API_BASE_URL`.

Note: this package contains frontend files only. It does not include the backend API adapter (`api.py`) or the original scheduler and dataset.
