import requests
import os


class LocalLLM:
    def __init__(self, model_name="llama3"):
        self.model_name = os.getenv("PSM_MODEL_NAME", model_name)
        self.url = os.getenv("PSM_OLLAMA_URL", "http://localhost:11434/api/generate")
        self.temperature = float(os.getenv("PSM_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("PSM_TOP_P", "0.8"))
        self.num_predict = int(os.getenv("PSM_NUM_PREDICT", "24"))
        self.num_ctx = int(os.getenv("PSM_NUM_CTX", "1024"))
        self.timeout_s = int(os.getenv("PSM_REQUEST_TIMEOUT_S", "45"))

    def generate(self, prompt):
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.num_predict,
                        "num_ctx": self.num_ctx,
                    },
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as exc:
            return f"[LLM_ERROR] {exc}"


def get_llm():
    """Return the appropriate LLM backend based on the ``PSM_LLM_BACKEND`` env var.

    Backends
    --------
    ``ollama`` (default)
        :class:`LocalLLM` — calls a local Ollama HTTP server (default port 11434).
    ``kaggle``
        :class:`~llm.kaggle_llm.KaggleLLM` — loads a HuggingFace model from a
        local directory (set ``PSM_KAGGLE_MODEL_PATH`` to the model folder).
    """
    backend = os.getenv("PSM_LLM_BACKEND", "ollama").lower()
    if backend == "kaggle":
        from llm.kaggle_llm import KaggleLLM
        return KaggleLLM()
    return LocalLLM()