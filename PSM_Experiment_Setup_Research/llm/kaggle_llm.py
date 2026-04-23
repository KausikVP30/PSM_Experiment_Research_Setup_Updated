"""Kaggle-compatible LLM backend using a HuggingFace model loaded from Kaggle's model store.

Usage in Kaggle notebooks
--------------------------
1. Open your notebook settings → "Models" → add any free model (e.g. Gemma 2B IT).
2. Set PSM_KAGGLE_MODEL_PATH to the model directory before importing, e.g.::

       import os
       os.environ["PSM_KAGGLE_MODEL_PATH"] = "/kaggle/input/gemma/transformers/2b-it/3"
       os.environ["PSM_LLM_BACKEND"] = "kaggle"

3. The Router will then call this class instead of LocalLLM (Ollama).

Popular Kaggle model paths (add the model from the Models tab first)
----------------------------------------------------------------------
Gemma 2B IT   : /kaggle/input/gemma/transformers/2b-it/3
Gemma 7B IT   : /kaggle/input/gemma/transformers/7b-it/3
Phi-2         : /kaggle/input/phi-2/transformers/default/1
Mistral 7B    : /kaggle/input/mistral-7b-instruct-v0.2/transformers/default/1
Llama-3 8B    : /kaggle/input/llama-3/transformers/8b-instruct/1
"""

from __future__ import annotations

import os

import torch
from transformers import pipeline


class KaggleLLM:
    """Text-generation LLM backend for Kaggle.

    Loads a HuggingFace causal-LM from a local path (Kaggle model directory).
    Has the same interface as ``LocalLLM``: a single ``generate(prompt)`` method.

    The pipeline is loaded lazily on the first call to ``generate`` so that
    import time stays fast and partial initialisation (e.g. during unit tests)
    does not consume GPU memory.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path: str = model_path or os.getenv(
            "PSM_KAGGLE_MODEL_PATH",
            "/kaggle/input/gemma/transformers/2b-it/3",
        )
        self.max_new_tokens: int = int(os.getenv("PSM_NUM_PREDICT", "64"))
        self.temperature: float = float(os.getenv("PSM_TEMPERATURE", "0.1"))
        self._pipe = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazily load the transformers pipeline on first use."""
        if self._pipe is not None:
            return

        use_gpu = torch.cuda.is_available()
        dtype = torch.float16 if use_gpu else torch.float32

        print(f"[KaggleLLM] Loading model from: {self.model_path}")
        print(f"[KaggleLLM] Device: {'GPU (float16)' if use_gpu else 'CPU (float32)'}")

        self._pipe = pipeline(
            "text-generation",
            model=self.model_path,
            torch_dtype=dtype,
            device_map="auto" if use_gpu else None,
        )

        print("[KaggleLLM] Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public interface (mirrors LocalLLM)
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a response for *prompt* and return it as a plain string."""
        try:
            self._load()
            outputs = self._pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return (outputs[0]["generated_text"] or "").strip()
        except Exception as exc:  # noqa: BLE001
            return f"[LLM_ERROR] {exc}"
