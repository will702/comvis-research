# Acne Detection CV Playground

An **interactive web playground** that walks through every stage of a computer-vision acne severity classification pipeline — from raw pixels to prediction — with live parameter sliders, a 42-dim feature inspector, and 21 classifier comparisons.

> **Models on Hugging Face →** [will702/acne-cv-models](https://huggingface.co/will702/acne-cv-models)

---

## How It Works

### System Architecture

```mermaid
graph TD
    User["👤 User (Browser)"]
    FE["React Frontend\nVite + Tailwind + Framer Motion"]
    BE["FastAPI Backend\nPort 8000 / $PORT"]
    PP["TracedPipeline\nsrc/ OpenCV ops"]
    ML["21 Classifiers\nsklearn + CatBoost .pkl"]
    HF["🤗 Hugging Face Hub\nwill702/acne-cv-models"]
    FS["Demo Images\ndata_split_6535/test/"]

    User -->|"pick demo / upload photo\nadjust sliders"| FE
    FE -->|"POST /api/run\n{image, params}"| BE
    BE --> PP
    BE --> ML
    PP -->|"18 stage images b64\n42-dim feature vector\nblob metadata"| BE
    ML -->|"21 predictions\n+ confidence"| BE
    BE -->|"trace + predictions JSON"| FE
    FE -->|"animated stage cards\nfeature grid\nmodel bars"| User
    ML -.->|"download on cold start\nif local models missing"| HF
    BE -->|"GET /api/demos"| FS
```

### ML Pipeline (12 Stages)

```mermaid
flowchart LR
    IMG["📷 Input Image"] --> S1

    subgraph S1["Stage 1–3: Preprocessing"]
        direction TB
        R["Resize 512×512"] --> G["Grayscale"]
        G --> C["CLAHE\nclipLimit=3.0\ntile=8×8"]
    end

    subgraph S2["Stage 4–5: Face ROI"]
        direction TB
        FD["Haar Cascade\nFace Detection\n4-attempt chain"]
        FD --> ROI["ROI Mask\neyes · nose · lips\nexcluded"]
    end

    subgraph S3["Stage 6–7: Skin Segmentation"]
        direction TB
        YC["YCrCb Range\nCr∈[133,173]\nCb∈[77,127]"]
        YC --> MC["Morph Close 9×9\n+ Open 5×5"]
        MC --> CM["skin ∩ ROI mask"]
    end

    subgraph S4["Stage 8–9: Lesion Detection"]
        direction TB
        AT["Adaptive Threshold\nthr_cr = max(145, μCr+1.5σ)\nthr_a  = max(133, μa+1.2σ)"]
        AT --> RM["Red Mask\nCr AND a* elevated"]
        RM --> CC["Connected Components\nshape filter\naspect · fill · contrast"]
    end

    subgraph S5["Stage 10–12: Feature Extraction"]
        direction TB
        ST["8 Structural\ncount·area·density\ncircularity·intensity"]
        LB["27 LBP\nR=1,2,3 × 9 bins\nuniform patterns"]
        GL["3 GLCM\ncontrast·homogeneity\nenergy"]
        RD["4 Global Redness\nμa·σa·μCr·σCr"]
    end

    S1 --> S2 --> S3 --> S4 --> S5

    S5 --> VEC["42-dim\nFeature Vector"]
    VEC --> SC["StandardScaler"]
    SC --> CL["21 Classifiers"]
    CL --> PRED["🎯 Severity Prediction\nGrade I · II · III"]
```

### Feature Vector Breakdown

```mermaid
pie title 42-Dim Feature Vector
    "LBP Texture R=1 (9 dims)" : 9
    "LBP Texture R=2 (9 dims)" : 9
    "LBP Texture R=3 (9 dims)" : 9
    "Structural (8 dims)" : 8
    "Global Redness (4 dims)" : 4
    "GLCM Texture (3 dims)" : 3
```

### Model Performance

```mermaid
xychart-beta
    title "Top 10 Models — Weighted F1 Score"
    x-axis ["SGD", "CatBoost", "Cal.SVM", "SVM RBF", "LDA", "LogReg", "MLP", "Stacking", "Ridge", "KNN"]
    y-axis "F1 Score" 0.60 --> 0.80
    bar [0.7504, 0.7494, 0.7466, 0.7399, 0.7371, 0.7360, 0.7301, 0.7297, 0.7233, 0.7222]
```

### Cold Start vs Warm Request Flow

```mermaid
sequenceDiagram
    participant C as Container
    participant HF as HF Hub
    participant DB as Local Cache /tmp

    note over C: Cold start (first deploy)
    C->>DB: check /tmp/hf_cache
    DB-->>C: not found
    C->>HF: snapshot_download(will702/acne-cv-models)
    HF-->>C: 21 × .pkl (~98MB)
    C->>DB: cache models
    note over C: Warm (subsequent requests)
    C->>DB: load models from cache ✓

    note over C: Local dev
    C->>C: models/model (42 dim)/*.pkl
    note right of C: HF not called
```

---

## Project Structure

```
comvis-research/
├── src/
│   ├── preprocessing.py     # FaceProcessor — CLAHE, face detect, skin seg
│   ├── features.py          # FeatureExtractor — 42-dim vector
│   └── config.py            # paths, class names
├── backend/
│   ├── app.py               # FastAPI routes + static file serving
│   ├── pipeline.py          # TracedPipeline — captures all intermediates
│   ├── models.py            # load 21 .pkl + HF Hub download fallback
│   ├── params.py            # PipelineParams dataclass (14 knobs)
│   ├── encode.py            # ndarray → base64 PNG helpers
│   └── test_pipeline.py     # fidelity guard: traced ≈ original
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── StageStrip.tsx       # 12 animated pipeline stages
│   │   │   ├── ParamPanel.tsx       # 14 live sliders
│   │   │   ├── FeatureInspector.tsx # 42-dim interactive grid
│   │   │   ├── ModelCompare.tsx     # 21 classifier table
│   │   │   └── ImagePicker.tsx      # demo gallery + drag-drop upload
│   │   └── lib/api.ts               # typed fetch client
│   └── ...
├── models/
│   └── model (42 dim)/      # 21 × .pkl (local dev only, not in git)
├── data_split_6535/test/    # demo images (acne1/2/3)
├── scripts/
│   └── upload_models_hf.py  # one-time HF Hub upload
├── Dockerfile               # multi-stage: node build → python slim
├── fly.toml                 # Fly.io config
├── railway.toml             # Railway config
├── .github/workflows/
│   └── cloud-run.yml        # Google Cloud Run CI/CD
├── requirements.txt
└── start.sh                 # local dev server
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 20+
- [uv](https://github.com/astral-sh/uv) (or pip)

### 1. Clone & create venv

```bash
git clone https://github.com/will702/comvis-research
cd comvis-research
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python3
```

### 2. Get models

**Option A — local** (if you have the trained models):
```bash
# models/model (42 dim)/*.pkl must exist
# Already present if you trained locally
```

**Option B — download from Hugging Face**:
```bash
export HF_MODEL_REPO=will702/acne-cv-models
# Backend downloads automatically on first start
```

### 3. Build frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Start server

```bash
./start.sh
# → http://localhost:8000
```

### Dev mode (hot reload on both sides)

```bash
# Terminal 1 — backend
.venv/bin/python3 -m uvicorn backend.app:app --reload

# Terminal 2 — frontend
cd frontend && npm run dev
# → http://localhost:5173  (proxies API to :8000)
```

### Run fidelity tests

```bash
.venv/bin/python3 -m pytest backend/test_pipeline.py -v
# 5/5 pass: TracedPipeline ≈ FeatureExtractor.extract() within 1e-5
```

---

## Deployment

> **Models are NOT in the Docker image.** Set `HF_MODEL_REPO=will702/acne-cv-models` on your host — container downloads ~98MB from HF Hub on first cold start.

### Option 1 — Railway (easiest)

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

Set env var in Railway dashboard → Service → Variables:
```
HF_MODEL_REPO=will702/acne-cv-models
```

Requires **1GB RAM** minimum (set in Railway service settings).

---

### Option 2 — Fly.io

```bash
brew install flyctl
fly auth login
fly launch        # detects Dockerfile, edit fly.toml app name first
fly secrets set HF_MODEL_REPO=will702/acne-cv-models
fly deploy
```

Config in `fly.toml` already sets `memory = "2gb"`.

---

### Option 3 — Google Cloud Run

```bash
# One-time setup
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com

# Build & deploy
docker build -t gcr.io/YOUR_PROJECT_ID/acne-cv-playground .
docker push gcr.io/YOUR_PROJECT_ID/acne-cv-playground

gcloud run deploy acne-cv-playground \
  --image gcr.io/YOUR_PROJECT_ID/acne-cv-playground \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars HF_MODEL_REPO=will702/acne-cv-models
```

**CI/CD (auto-deploy on push to `main`):** add these GitHub secrets:
| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | your GCP project ID |
| `GCP_SA_KEY` | JSON key with Cloud Run Admin + Storage Admin roles |

`.github/workflows/cloud-run.yml` handles the rest.

---

### Resource requirements

| | Min RAM | Cold start | Notes |
|---|---|---|---|
| Railway | 1 GB | ~45s | $5/mo free credit |
| Fly.io | 2 GB | ~45s | `shared-cpu-1x` + `2gb` memory |
| Cloud Run | 2 GB | ~45s | scales to 0, pay-per-request |

---

## Retrain & Re-upload Models

```bash
# 1. Retrain (uses existing src/ pipeline)
.venv/bin/python3 src/train.py

# 2. Re-upload to HF Hub
huggingface-cli login
python scripts/upload_models_hf.py --repo will702/acne-cv-models

# 3. Restart deployed container — it pulls fresh models on next cold start
```

---

## Dataset

[ACNE04](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Z8tMaJR3PJtVblSH) — 3 severity grades used (acne1/2/3 = mild/moderate/severe).

```
Wu, X. et al. "Joint Acne Image Grading and Counting via Label Distribution Learning." ICCV 2019.
```

---

## Tech Stack

| Layer | Stack |
|---|---|
| Frontend | React 18 · Vite · TypeScript · Tailwind CSS · Framer Motion |
| Backend | FastAPI · Uvicorn · Python 3.13 |
| CV | OpenCV · scikit-image (LBP/GLCM) |
| ML | scikit-learn 1.7.2 · CatBoost · joblib |
| Models | Hugging Face Hub ([will702/acne-cv-models](https://huggingface.co/will702/acne-cv-models)) |
| Deploy | Docker · Google Cloud Run · Railway · Fly.io |
