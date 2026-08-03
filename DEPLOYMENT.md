# MahaGR Assist — deployment runbook (Phase 7)

Two targets from one codebase:

- **Track A — On-prem bundle** (the government/NIC story): backend + a local LLM
  (Ollama) + frontend, brought up with one `docker compose up`, works air-gapped.
- **Track B — Public cloud demo** (the resume URL): backend on a container host
  using Groq, frontend on Vercel.

The only difference between them is the `LLM_PROVIDER` env var — no code change.

---

## 0. Prerequisites

- Docker + Docker Compose.
- A **prebuilt index** at `backend/index/` (run `scripts/ingest_text.py` then
  `scripts/add_fixtures.py` once on a machine with the models). This is shipped
  into the image / mounted as a volume so the container never re-embeds.
- For GPU on-prem: NVIDIA driver + NVIDIA Container Toolkit.
- Model weights: bge-m3 (~2.2 GB) + bge-reranker-v2-m3 (~2.3 GB). Downloaded to a
  Hugging Face cache **volume** on first run (or pre-seeded for air-gap — see §4).

Add a `backend/.dockerignore` and `frontend/.dockerignore` excluding `.venv/`,
`node_modules/`, `.next/`, `data/grs_text/`, `__pycache__/` so images stay small.

---

## 1. Backend image — `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
# OCR for scanned Marathi GRs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-mar tesseract-ocr-hin tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY engine ./engine
COPY app ./app
RUN pip install --no-cache-dir -e .          # engine package now on path
COPY index ./index                            # prebuilt FAISS index (or mount a volume)
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

> Smaller image on CPU-only hosts: install the CPU build of torch instead of the
> default CUDA wheel — `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> before the requirements step. On a GPU host keep the default CUDA wheel.

## 2. Frontend image — `frontend/Dockerfile`

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "start"]
```

> `NEXT_PUBLIC_*` is inlined at **build** time, so the backend URL must be passed
> as a build ARG (see compose below), not just at runtime.

---

## Track A — On-prem bundle (air-gap capable)

### 3. `docker-compose.yml` (repo root)

```yaml
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama:/root/.ollama
    # GPU (optional): uncomment to give Ollama the NVIDIA GPU
    # deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] } } }

  backend:
    build: ./backend
    environment:
      - LLM_PROVIDER=ollama
      - OLLAMA_BASE_URL=http://ollama:11434
      - OLLAMA_MODEL=llama3.1:8b
      - MAHAGR_INDEX=index
    volumes:
      - hf:/root/.cache/huggingface      # embedding + reranker weights (persist across restarts)
      - ./backend/index:/app/index       # FAISS index
      - sqlite:/app/data/db              # Phase 3 persistence (conversations/feedback)
    depends_on: [ollama]
    ports: ["8000:8000"]

  frontend:
    build:
      context: ./frontend
      args:
        NEXT_PUBLIC_API_URL: http://localhost:8000
    depends_on: [backend]
    ports: ["3000:3000"]

volumes:
  ollama:
  hf:
  sqlite:
```

### 4. Bring it up

```bash
docker compose up -d --build
docker compose exec ollama ollama pull llama3.1:8b     # one-time; lands in the ollama volume
curl http://localhost:8000/health                       # expect llm_provider: "ollama"
# open http://localhost:3000
```

**Air-gapped install:** on a machine WITH internet, run the two commands above
once so the `hf` and `ollama` volumes fill with weights, then export those
volumes (`docker run --rm -v hf:/data -v $PWD:/backup alpine tar czf /backup/hf.tgz -C /data .`,
same for `ollama`) and the built images (`docker save`). Import them on the
offline server (`docker load`, and restore the volumes). Nothing then reaches
the internet — the whole stack runs inside the department network.

### 5. HTTPS / reverse proxy (production)

Put Caddy or Nginx in front, terminating TLS and routing `/` → frontend:3000 and
`/api` → backend:8000. Add `restart: unless-stopped` to each service and a
`healthcheck` hitting `/health` on the backend.

---

## Track B — Public cloud demo (resume URL)

### 6. Backend on a container host (Render / Railway / GCP Cloud Run)

- Deploy the `backend/` image.
- Env: `GROQ_API_KEY=...` (leave `LLM_PROVIDER` unset → defaults to `groq`),
  `MAHAGR_INDEX=index`.
- **Cold-start caveat:** the models are ~4.5 GB. Either bake them into the image
  (build step: run a tiny script that loads bge-m3 + the reranker so the weights
  are cached in a layer) or attach a persistent disk mounted at the HF cache path.
  Otherwise every cold start re-downloads them.
- CPU-only host works (bge-m3 on CPU is fine for demo traffic); set a generous
  request timeout.

### 7. Frontend on Vercel

- Import `frontend/`.
- Set `NEXT_PUBLIC_API_URL` to the deployed backend URL (Production env var).
- Redeploy so the value is inlined into the build.
- Confirm the backend's CORS allows the Vercel origin (it currently allows `*`;
  tighten to the Vercel domain for production).

---

## 8. Environment variable reference

| Variable | Where | Purpose |
|---|---|---|
| `LLM_PROVIDER` | backend | `groq` (default) or `ollama` |
| `GROQ_API_KEY` | backend | needed when provider is groq |
| `OLLAMA_BASE_URL` | backend | e.g. `http://ollama:11434` (on-prem) |
| `OLLAMA_MODEL` | backend | e.g. `llama3.1:8b` |
| `MAHAGR_INDEX` | backend | index dir inside the container (default `index`) |
| `NEXT_PUBLIC_API_URL` | frontend (build) | backend base URL |

## 9. Verify & troubleshoot

- `GET /health` → shows `indexed_vectors`, `embedding_model`, `llm_provider`.
- `GET /documents` (no key needed) → confirms the index mounted.
- Empty `/documents` → the index volume/copy is missing or `MAHAGR_INDEX` is wrong.
- Ollama errors → model not pulled yet (`docker compose exec ollama ollama pull ...`)
  or the GPU/CPU is out of memory (use a smaller model or CPU).
- Chat 429 on Groq → daily token limit; switch to a smaller model or to Ollama.

**Done when:** `docker compose up` serves the whole stack offline with a local
LLM, and a public URL is live for the resume.
