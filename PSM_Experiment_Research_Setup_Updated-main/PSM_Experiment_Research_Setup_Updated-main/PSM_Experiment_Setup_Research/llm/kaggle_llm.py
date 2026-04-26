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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


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
        self._model = None
        self._tokenizer = None
        self._generation_config = None

    def _resolve_pad_token_id(self) -> int:
        """Return a safe pad_token_id across model/tokenizer variants."""
        tokenizer = self._tokenizer
        model = self._model

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(getattr(model, "config", object()), "eos_token_id", None)

        if eos_id is not None:
            # Align tokenizer pad token when absent (common for decoder-only models).
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return int(eos_id)

        # Last-resort fallback; avoids runtime attribute errors in generation helpers.
        return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazily load the transformers pipeline on first use."""
        if self._model is not None and self._tokenizer is not None:
            return

        use_gpu = torch.cuda.is_available()
        dtype = torch.float16 if use_gpu else torch.float32

        print(f"[KaggleLLM] Loading model from: {self.model_path}")
        print(f"[KaggleLLM] Device: {'GPU (float16)' if use_gpu else 'CPU (float32)'}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto" if use_gpu else None,
            trust_remote_code=True,
        )

        # Decoder-only models may not define pad token ids. Ensure both tokenizer
        # and model config have one to avoid generation-time config attribute errors.
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        # Some remote-code model configs (including certain Phi variants) can expose
        # generation config objects that do not carry pad_token_id. Build and store
        # an explicit GenerationConfig so model.generate never depends on missing attrs.
        gen_cfg = GenerationConfig.from_model_config(model.config)
        if getattr(gen_cfg, "eos_token_id", None) is None:
            gen_cfg.eos_token_id = tokenizer.eos_token_id or getattr(model.config, "eos_token_id", None)
        if getattr(gen_cfg, "pad_token_id", None) is None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id or gen_cfg.eos_token_id or 0

        self._generation_config = gen_cfg
        model.generation_config = gen_cfg

        self._model = model
        self._tokenizer = tokenizer

        print("[KaggleLLM] Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public interface (mirrors LocalLLM)
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a response for *prompt* and return it as a plain string."""
        try:
            self._load()
            inputs = self._tokenizer(prompt, return_tensors="pt")
            model_device = next(self._model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._resolve_pad_token_id(),
                    eos_token_id=getattr(self._generation_config, "eos_token_id", None),
                    generation_config=self._generation_config,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            generated_only = output_ids[0][prompt_len:]
            text = self._tokenizer.decode(generated_only, skip_special_tokens=True)
            return (text or "").strip()
        except Exception as exc:  # transformers/CUDA errors have many forms; surface them as LLM errors
            return f"[LLM_ERROR] {exc}"
