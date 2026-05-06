# PSM Experiment Research Setup

**Predictive Semantic Memory (PSM)** — a retrieval-augmented QA pipeline with a
persistent memory layer, evaluated on TriviaQA.

## Quick links

- 📓 **Kaggle / Colab run guide** →
  [`PSM_Experiment_Setup_Research/RUNNING_ON_KAGGLE_AND_COLAB.md`](PSM_Experiment_Setup_Research/RUNNING_ON_KAGGLE_AND_COLAB.md)
- 📒 **Ready-made notebook** →
  [`PSM_Experiment_Setup_Research/PSM_Kaggle_Experiment.ipynb`](PSM_Experiment_Setup_Research/PSM_Kaggle_Experiment.ipynb)
- 🗺️ **Repo knowledge graph** →
  [`REPO_KNOWLEDGE_GRAPH.md`](REPO_KNOWLEDGE_GRAPH.md)
- 🗂️ **Source code** → [`PSM_Experiment_Setup_Research/`](PSM_Experiment_Setup_Research/)

## What this repo contains

| Path | Description |
|------|-------------|
| `PSM_Experiment_Setup_Research/PSM_Kaggle_Experiment.ipynb` | Self-contained notebook — runs on Kaggle or Colab |
| `PSM_Experiment_Setup_Research/run_psm_experiments.py` | Full experiment runner (smoke + pilot TriviaQA profiles) |
| `PSM_Experiment_Setup_Research/llm/kaggle_llm.py` | HuggingFace `transformers` backend (no API key needed) |
| `PSM_Experiment_Setup_Research/llm/llm_interface.py` | `get_llm()` factory — switches between Ollama and Kaggle/HF backends |
| `PSM_Experiment_Setup_Research/router/router.py` | PSM routing logic (memory → retrieval → LLM) |
| `PSM_Experiment_Setup_Research/evaluation/` | EM, F1, ROUGE-L, BLEU metrics |
| `PSM_Experiment_Setup_Research/retrieval/` | Hybrid BM25 + FAISS retriever |
| `PSM_Experiment_Setup_Research/memory/` | FAISS-backed persistent memory store |

## Running on Kaggle or Colab

See the detailed step-by-step guide:  
**[RUNNING_ON_KAGGLE_AND_COLAB.md](PSM_Experiment_Setup_Research/RUNNING_ON_KAGGLE_AND_COLAB.md)**

It covers:
- Adding a free LLM from Kaggle Models (no token required)
- Using HuggingFace Hub models on Colab
- Using Ollama inside Colab
- All environment variable settings
- Saving and pushing results to GitHub
