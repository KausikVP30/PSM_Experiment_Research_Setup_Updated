# Running the PSM Experiment — Kaggle & Google Colab Guide

This guide walks you through running the **Predictive Semantic Memory (PSM)** experiment
pipeline on two free cloud platforms.  
Pick the platform that suits you best, then follow the steps for that platform.

---

## Table of Contents

1. [Overview — what this experiment does](#overview)
2. [Platform comparison](#platform-comparison)
3. [Running on Kaggle](#running-on-kaggle)
   - [Step 1 — Fork / open the notebook](#kaggle-step-1)
   - [Step 2 — Attach the PSM source code](#kaggle-step-2)
   - [Step 3 — Add a free LLM from Kaggle Models](#kaggle-step-3)
   - [Step 4 — Set the model path in the config cell](#kaggle-step-4)
   - [Step 5 — Run the notebook](#kaggle-step-5)
   - [Step 6 — Download results](#kaggle-step-6)
   - [Popular Kaggle model paths](#kaggle-model-paths)
   - [Kaggle troubleshooting](#kaggle-troubleshooting)
4. [Running on Google Colab](#running-on-colab)
   - [Step 1 — Open or upload the notebook](#colab-step-1)
   - [Step 2 — Enable GPU](#colab-step-2)
   - [Step 3 — Mount / clone the PSM source code](#colab-step-3)
   - [Step 4 — Install dependencies](#colab-step-4)
   - [Step 5 — Pick your LLM backend](#colab-step-5)
     - [Option A — Use a HuggingFace model (free, any size)](#colab-option-a)
     - [Option B — Use Ollama inside Colab (Llama 3, Mistral, etc.)](#colab-option-b)
   - [Step 6 — Run the experiment](#colab-step-6)
   - [Step 7 — Save results to Google Drive](#colab-step-7)
   - [Colab troubleshooting](#colab-troubleshooting)
5. [Environment variables reference](#env-vars)
6. [Output files explained](#output-files)
7. [Pushing results to GitHub](#pushing-to-github)

---

## Overview — what this experiment does <a name="overview"></a>

The PSM pipeline is a **retrieval-augmented question-answering** system with a
*persistent memory* layer.  For each incoming question it:

1. **Checks memory** — if a similar question was answered before with high confidence,
   the stored answer is returned immediately (fast path).
2. **Retrieves context** — if memory confidence is too low, a hybrid BM25 + dense FAISS
   retriever fetches the most relevant document chunks.
3. **Generates an answer** — the retrieved context is fed to an LLM to produce a final answer.
4. **Stores the result** — the question + answer are written back into memory so future
   similar queries can skip retrieval.

Metrics collected per question: **Exact Match (EM)**, **Token F1**, **ROUGE-L**,
**BLEU**, confidence score, latency, and routing path (memory vs. retrieval).

---

## Platform comparison <a name="platform-comparison"></a>

| | **Kaggle** | **Google Colab** |
|---|---|---|
| Free GPU | ✅ T4 / P100, up to 30 h/week | ✅ T4, up to ~12 h/session |
| Free LLM hosting | ✅ Models tab (Gemma, Phi, Llama, …) | ⚠️ Download from HuggingFace Hub or install Ollama |
| HuggingFace token needed? | ❌ Not for public models | ❌ Not for public models |
| Persistent storage | `/kaggle/working` (saved automatically) | Google Drive (needs mounting) |
| Best for | Running the ready-made notebook | Iterating / debugging / larger experiments |

---

## Running on Kaggle <a name="running-on-kaggle"></a>

### Step 1 — Fork / open the notebook <a name="kaggle-step-1"></a>

**Option A — Use the repo notebook (recommended)**

1. Go to [kaggle.com/notebooks](https://www.kaggle.com/notebooks) and click
   **New Notebook**.
2. In the new notebook editor choose **File → Import Notebook**.
3. Select **GitHub** and paste the repo URL:
   ```
   https://github.com/KausikVP30/PSM_Experiment_Research_Setup_Updated
   ```
4. Select `PSM_Experiment_Setup_Research/PSM_Kaggle_Experiment.ipynb` and click
   **Import**.

**Option B — Upload manually**

1. Download `PSM_Kaggle_Experiment.ipynb` from the repo.
2. On Kaggle click **New Notebook → File → Import Notebook → Upload** and select
   the file.

---

### Step 2 — Attach the PSM source code <a name="kaggle-step-2"></a>

The notebook needs the PSM Python source tree available at runtime.

1. In the notebook editor, click the **Add-ons** (➕) icon in the top-right sidebar.
2. Choose **Datasets → Add Dataset**.
3. Search for `PSM Experiment Research Setup` and add the dataset that contains this
   repository, **or** create your own:
   - Go to [kaggle.com/datasets/new](https://www.kaggle.com/datasets/new).
   - Upload a ZIP of the repo (or link it to GitHub via the Kaggle CLI — see
     [docs](https://www.kaggle.com/docs/api#interacting-with-datasets)).
   - The dataset will be mounted at `/kaggle/input/<your-dataset-slug>/`.
4. In the **config cell** (Cell 2) of the notebook set `PSM_ROOT` to point at the
   mounted path, e.g.:
   ```python
   PSM_ROOT = "/kaggle/input/psm-experiment-research-setup/PSM_Experiment_Setup_Research"
   ```
   Cell 3 (imports) already contains fallback logic that detects the path automatically,
   so this is only needed if auto-detection fails.

---

### Step 3 — Add a free LLM from Kaggle Models <a name="kaggle-step-3"></a>

Kaggle hosts dozens of open-weight LLMs at no cost.  You **do not** need a HuggingFace
token or any paid API key.

1. In the notebook editor click the **Add-ons** (➕) icon → **Models**.
2. Click **Add Model**.
3. Search for the model you want (see table below) and click **Add**.
4. Accept the model's licence if prompted (one-time per account).
5. The model will be available inside the notebook at a path like
   `/kaggle/input/<model-slug>/transformers/<variant>/<version>/`.

> **Tip:** Click the **ⓘ icon** next to the model name in the sidebar after adding it
> to see its exact mounted path.

---

### Step 4 — Set the model path in the config cell <a name="kaggle-step-4"></a>

Open **Cell 2** (the Config cell) and set `KAGGLE_MODEL_PATH` to the path shown by
Kaggle:

```python
# ── CHANGE THIS to your model path ───────────────────────────────────────────
KAGGLE_MODEL_PATH = "/kaggle/input/gemma/transformers/2b-it/3"   # ← edit this
```

Everything else can be left at the defaults for a first run.

| Setting | Default | Meaning |
|---------|---------|---------|
| `SMOKE_SIZE` | `10` | Number of synthetic questions for the smoke test |
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum memory-match score to skip retrieval |
| `OUTPUT_BASE` | `/kaggle/working/psm_results` | Where all CSVs and reports are saved |
| `PSM_NUM_PREDICT` | `64` | Max new tokens the LLM generates per answer |
| `PSM_TEMPERATURE` | `0.1` | Near-greedy generation (good for factual Q&A) |

---

### Step 5 — Run the notebook <a name="kaggle-step-5"></a>

1. Make sure **Accelerator** is set to **GPU T4 x2** (or **GPU P100**) in
   **Notebook settings** (top-right ⚙ icon).  
   *CPU-only is possible but ~10× slower.*
2. Click **Run → Run All** (or press `Shift+Enter` through each cell).
3. The notebook will:
   - Install missing packages (≈ 2 min on first run, cached afterwards)
   - Validate the model path — a clear error is raised if it is wrong
   - Run a **smoke test** (10 synthetic questions) to confirm the pipeline works
   - Run the **full TriviaQA experiment** (`smoke` profile: 5 questions,
     `pilot` profile: 8 questions)
   - Display all result tables inline
   - Print a file listing of every output file

Total runtime: **~15–25 min** with GPU (Gemma 2B IT).

---

### Step 6 — Download results <a name="kaggle-step-6"></a>

All output files land in `/kaggle/working/psm_results/`.  To download:

- Click the **Output** tab (right sidebar) → expand `psm_results/` → click the ⬇
  icon next to any file or folder.
- To download everything at once: click **Download All** at the top of the Output tab.

Key files:

| File | Description |
|------|-------------|
| `experiment_runs/all_predictions_combined.csv` | Every question, generated answer, and metric |
| `experiment_runs/all_runs_metrics.csv` | One row per experiment profile |
| `experiment_runs/<run_id>/summary_metrics.csv` | Flat one-row summary for a single run |
| `experiment_runs/<run_id>/run_readable_report.txt` | Human-readable narrative report |

---

### Popular Kaggle model paths <a name="kaggle-model-paths"></a>

After adding a model in the Models tab these are the paths Kaggle assigns.
*(Version numbers may change — always verify with the ⓘ icon.)*

| Model | Size | Path |
|-------|------|------|
| **Gemma 2B IT** *(recommended)* | 2B | `/kaggle/input/gemma/transformers/2b-it/3` |
| Gemma 7B IT | 7B | `/kaggle/input/gemma/transformers/7b-it/3` |
| Phi-2 | 2.7B | `/kaggle/input/phi-2/transformers/default/1` |
| Phi-3 Mini 4K | 3.8B | `/kaggle/input/phi-3/transformers/4b-instruct/1` |
| Mistral 7B Instruct v0.2 | 7B | `/kaggle/input/mistral-7b-instruct-v0.2/transformers/default/1` |
| Llama 3 8B Instruct | 8B | `/kaggle/input/llama-3/transformers/8b-instruct/1` |
| Llama 3.1 8B Instruct | 8B | `/kaggle/input/llama-3.1/transformers/8b-instruct/1` |

---

### Kaggle troubleshooting <a name="kaggle-troubleshooting"></a>

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: Model path not found` | The model is not attached or the path is wrong. Click ⓘ next to the model in the sidebar to copy the exact path. |
| `RuntimeError: PSM source tree not found` | The PSM dataset is not attached. Go to Add-ons → Datasets and attach the repo dataset. |
| `CUDA out of memory` | Use a smaller model (Gemma 2B IT or Phi-2) or reduce `PSM_NUM_PREDICT` to `32`. |
| `ModuleNotFoundError: faiss` | Re-run the install cell (Cell 2). Kaggle caches installed packages only within the same session. |
| Notebook session timeout | Kaggle sessions last up to ~9 hours. For longer runs, increase profile sizes gradually and save intermediate CSVs. |
| Slow generation | Lower `PSM_NUM_PREDICT` (e.g. `32`) and/or set `PSM_TEMPERATURE = "0.0"` for greedy decoding. |

---

## Running on Google Colab <a name="running-on-colab"></a>

### Step 1 — Open or upload the notebook <a name="colab-step-1"></a>

**Option A — Open directly from GitHub**

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Choose **File → Open notebook → GitHub**.
3. Paste the repo URL:
   ```
   https://github.com/KausikVP30/PSM_Experiment_Research_Setup_Updated
   ```
4. Select `PSM_Experiment_Setup_Research/PSM_Kaggle_Experiment.ipynb` and click
   **Open in Colab**.

**Option B — Upload**

1. Download `PSM_Kaggle_Experiment.ipynb` from the repo.
2. In Colab choose **File → Upload notebook** and select the file.

---

### Step 2 — Enable GPU <a name="colab-step-2"></a>

1. In the top menu click **Runtime → Change runtime type**.
2. Set **Hardware accelerator** to **T4 GPU** (free) or **A100 / L4** (Colab Pro).
3. Click **Save**.

> ⚠️ Without a GPU, models like Gemma 7B are extremely slow.  
> Use Gemma 2B IT or Phi-2 on CPU if a GPU is not available.

---

### Step 3 — Mount / clone the PSM source code <a name="colab-step-3"></a>

The notebook needs the PSM Python source tree.  Add a cell at the very top of the
notebook and run it **once**:

```python
# Clone the PSM repo into the Colab session storage
import os

REPO_URL  = "https://github.com/KausikVP30/PSM_Experiment_Research_Setup_Updated.git"
REPO_DIR  = "/content/PSM_Experiment_Research_Setup_Updated"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")

PSM_ROOT = f"{REPO_DIR}/PSM_Experiment_Setup_Research"
print(f"PSM root: {PSM_ROOT}")
```

> **Using Google Drive instead?**  Mount Drive first, then clone/copy the repo there
> so it persists between sessions:
> ```python
> from google.colab import drive
> drive.mount("/content/drive")
> REPO_DIR = "/content/drive/MyDrive/PSM_Experiment_Research_Setup_Updated"
> ```

---

### Step 4 — Install dependencies <a name="colab-step-4"></a>

Colab pre-installs `torch`, `numpy`, and many common libraries.  Run this cell once:

```python
%%capture
import subprocess, sys

PACKAGES = [
    "faiss-cpu",
    "sentence-transformers",
    "rank-bm25",
    "datasets",
    "rouge-score",
    "nltk",
    "accelerate",
    "transformers>=4.38",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *PACKAGES])

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
print("✓ All packages installed.")
```

---

### Step 5 — Pick your LLM backend <a name="colab-step-5"></a>

On Colab you have two options.  Choose **one** and run it.

#### Option A — HuggingFace model (recommended, free) <a name="colab-option-a"></a>

Use any public HuggingFace model directly.  No token is needed for public models.

```python
import os, sys
sys.path.insert(0, PSM_ROOT)   # set PSM_ROOT in Step 3

# ── Choose a model ─────────────────────────────────────────────────────────
# Gemma 2B IT   (fast, good quality, needs ~6 GB VRAM)
HF_MODEL_ID = "google/gemma-2b-it"

# Other good options:
# HF_MODEL_ID = "microsoft/phi-2"             # 2.7B, very fast
# HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"  # 7B, needs 16 GB VRAM / offloading
# HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # 8B (needs HF token for Llama)

os.environ["PSM_LLM_BACKEND"]       = "kaggle"   # KaggleLLM works on Colab too
os.environ["PSM_KAGGLE_MODEL_PATH"] = HF_MODEL_ID
os.environ["PSM_NUM_PREDICT"]       = "64"
os.environ["PSM_TEMPERATURE"]       = "0.1"

print(f"Backend : {os.environ['PSM_LLM_BACKEND']}")
print(f"Model   : {os.environ['PSM_KAGGLE_MODEL_PATH']}")
```

> **Note for gated models (Llama 3, Gemma):**  
> You need a HuggingFace access token for gated model repos.
> ```python
> from huggingface_hub import login
> login(token="hf_YOUR_TOKEN_HERE")  # get a free token at huggingface.co/settings/tokens
> ```
> Non-gated models (Phi-2, Mistral) work with no token.

#### Option B — Ollama inside Colab (Llama 3, Mistral, etc.) <a name="colab-option-b"></a>

Ollama lets you run quantised models efficiently.  It uses the default `LocalLLM`
backend so no extra config is needed.

```python
import subprocess, time, os, sys
sys.path.insert(0, PSM_ROOT)

# 1. Install Ollama
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

# 2. Start Ollama server in the background
proc = subprocess.Popen(["ollama", "serve"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)   # wait for server to start

# 3. Pull the model you want (first run only, ~4 GB download for llama3)
MODEL = "llama3"   # or "mistral", "phi3", "gemma:2b", etc.
subprocess.run(["ollama", "pull", MODEL], check=True)

# 4. Tell PSM to use Ollama (this is already the default)
os.environ["PSM_LLM_BACKEND"]  = "ollama"
os.environ["PSM_MODEL_NAME"]   = MODEL
os.environ["PSM_NUM_PREDICT"]  = "64"
os.environ["PSM_TEMPERATURE"]  = "0.0"

print(f"Ollama running | model: {MODEL}")
```

---

### Step 6 — Run the experiment <a name="colab-step-6"></a>

After completing Steps 3–5 you can run the experiment cells directly.  
If you are using `PSM_Kaggle_Experiment.ipynb`, **skip Cell 2** (Kaggle-specific config)
and instead run your Step 5 cell first.  Everything else runs identically.

Alternatively, run the experiment script directly from a code cell:

```python
import os, sys
sys.path.insert(0, PSM_ROOT)
os.chdir(PSM_ROOT)

from run_psm_experiments import PSMExperimentRunner
runner = PSMExperimentRunner(base_results_dir="/content/psm_results", threshold=0.55)
results = runner.execute()
print(results)
```

---

### Step 7 — Save results to Google Drive <a name="colab-step-7"></a>

Colab sessions are ephemeral — **always copy results to Google Drive** before the
session times out.

```python
from google.colab import drive
import shutil, os

# Mount Drive (if not already mounted)
drive.mount("/content/drive")

# Copy results
SRC = "/content/psm_results"
DST = "/content/drive/MyDrive/psm_results"
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

print(f"✓ Results saved to Google Drive: {DST}")
```

---

### Colab troubleshooting <a name="colab-troubleshooting"></a>

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: faiss` | Re-run the install cell (Step 4). |
| `CUDA out of memory` | Use a smaller model (Phi-2 or Gemma 2B) or restart the runtime and reduce `PSM_NUM_PREDICT`. |
| Gated model `401` / `403` error | Log in with a HuggingFace token: `from huggingface_hub import login; login("hf_...")`. |
| Ollama `connection refused` | Wait a few more seconds after `ollama serve`, then retry. |
| Session disconnected / results lost | Always copy to Google Drive (Step 7) before closing. |
| Slow on CPU | Go to **Runtime → Change runtime type → GPU** and run all cells again. |
| `[LLM_ERROR] ...` in answers | The LLM process crashed. Check CUDA OOM errors above it and switch to a smaller model. |

---

## Environment variables reference <a name="env-vars"></a>

All PSM settings are controlled via environment variables so no source files need to be
edited between runs.

| Variable | Default | Description |
|----------|---------|-------------|
| `PSM_LLM_BACKEND` | `ollama` | `ollama` → Ollama HTTP server · `kaggle` → HuggingFace `transformers` pipeline |
| `PSM_MODEL_NAME` | `llama3` | Model name passed to Ollama (only when `PSM_LLM_BACKEND=ollama`) |
| `PSM_OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `PSM_KAGGLE_MODEL_PATH` | `/kaggle/input/gemma/transformers/2b-it/3` | Local path **or** HuggingFace Hub model ID (when `PSM_LLM_BACKEND=kaggle`) |
| `PSM_NUM_PREDICT` | `24` (Ollama) / `64` (KaggleLLM) | Max new tokens per generation |
| `PSM_TEMPERATURE` | `0.0` | Sampling temperature (0 = greedy) |
| `PSM_TOP_P` | `0.8` | Nucleus sampling threshold (Ollama only) |
| `PSM_NUM_CTX` | `1024` | Context window size (Ollama only) |
| `PSM_REQUEST_TIMEOUT_S` | `45` | HTTP request timeout for Ollama (seconds) |
| `PSM_EMBEDDING_DEVICE` | `cpu` | Device for the sentence-transformer embedding model |

---

## Output files explained <a name="output-files"></a>

After a run the following files are created under `<results_dir>/<run_id>/`:

```
<run_id>/
├── predictions.csv            ← Per-question: question, generated answer, EM, F1, ROUGE-L, BLEU,
│                                 confidence, source (memory/retrieval), latency
├── predictions.jsonl          ← Same as above in JSON Lines format
├── planned_queries.csv        ← Questions planned before the run starts
├── summary_metrics.csv        ← Flat one-row summary of the entire run (easy to open in Excel)
├── summary_metrics.json       ← Full nested summary including per-path breakdowns
├── run_config.json            ← Reproducibility: all settings used
├── corpus_info.txt            ← Corpus statistics and first 10 chunks
└── run_readable_report.txt    ← Human-readable narrative report with interpretation notes
```

At the top level of `<results_dir>`:

```
experiment_runs/
├── all_runs_metrics.csv       ← One row per profile/run — compare runs at a glance
├── all_runs_metrics.json      ← Same as above in JSON
├── all_predictions_combined.csv ← All questions from all runs in one file
└── experiment_master_report.txt ← Cross-run master report with best-run summary
```

---

## Pushing results to GitHub <a name="pushing-to-github"></a>

Once you have downloaded (Kaggle) or saved to Drive (Colab) your results:

```bash
# On your local machine
git clone https://github.com/KausikVP30/PSM_Experiment_Research_Setup_Updated.git
cd PSM_Experiment_Research_Setup_Updated

# Create a results folder
mkdir -p results/<run_id>

# Copy the CSVs you want to track
cp ~/Downloads/psm_results/experiment_runs/all_runs_metrics.csv results/
cp ~/Downloads/psm_results/experiment_runs/<run_id>/summary_metrics.csv results/<run_id>/
cp ~/Downloads/psm_results/experiment_runs/<run_id>/predictions.csv results/<run_id>/

# Commit and push
git add results/
git commit -m "Add experiment results: <run_id>"
git push
```

---

*Last updated: April 2026*
