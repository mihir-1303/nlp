# Supply Chain Dashboard

Lightweight Next.js 15 frontend for the existing FastAPI machine-learning backend. This is a capstone demonstration interface, not a commercial enterprise product.

## Setup

From this directory:

```powershell
npm install
Copy-Item .env.example .env.local
```

Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` to the running FastAPI server. The default example is `http://localhost:8000`.

## Run

```powershell
npm run dev
```

Open `http://localhost:3000`.

For a production build:

```powershell
npm run lint
npm run build
npm run start
```

## Backend integrations

- `GET /api/health` powers the connection and model-availability status on Overview.
- `POST /api/incidents/classify` powers Incident Analysis and sends `{ "description": "..." }`.
- `POST /api/delivery/predict` powers Delivery Risk and sends the backend’s exact 14 model feature names plus `threshold`.

The frontend does not create supplier recommendations, disruption probabilities, incident severity scores, statistics, or sample prediction responses. Supplier Decisions is intentionally a placeholder because that backend functionality is not implemented.
