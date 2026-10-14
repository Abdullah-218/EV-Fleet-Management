# EV Fleet Command Center

React/Vite frontend for the EV Fleet Prediction API.

## Run locally

Start the FastAPI backend from the repository root:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Install and start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/*` to
`http://localhost:8000/*`. To use another backend URL, set
`VITE_API_URL`, for example:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Included features

- Fleet overview dashboard with model health status.
- Range intelligence form and physical/practical range outputs.
- Battery degradation form and degradation-per-100-cycle output.
- Remaining useful life form with driver and mechanical indicators.
- Combined `/predict/all` assessment workflow.
- Responsive layout for desktop, tablet, and mobile.
- Animated CSS 3D EV visualization.
- API loading, validation-error, offline, and reset states.
