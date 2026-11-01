"""Language-model interface and a local Hugging Face Transformers implementation.

Nothing heavy is imported at module import time. `HuggingFaceLLM` downloads and loads the
model on the first `load()` or `generate()` call, so the API and unit tests never pull a
multi-gigabyte model unless they explicitly ask for one.
"""

import logging
import os
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from app.rag.deps import require

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class LLMLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float


class LLM(Protocol):
    model_id: str

    def generate(
        self, messages: list[dict[str, str]], *, max_new_tokens: int = 256
    ) -> Generation: ...


class SerializedLLM:
    """Shares one loaded model across requests, running one generation at a time.

    A single model instance is not safe for concurrent `generate` calls from several threads.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self.model_id = llm.model_id
        self._lock = threading.Lock()

    def load(self) -> None:
        load = getattr(self._llm, "load", None)
        if load is not None:
            with self._lock:
                load()

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> Generation:
        with self._lock:
            return self._llm.generate(messages, max_new_tokens=max_new_tokens)


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    dtype: str
    reason: str
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None


def _cuda_arch_supported(torch: Any) -> tuple[bool, str]:
    major, minor = torch.cuda.get_device_capability(0)
    capability = major * 10 + minor
    archs = torch.cuda.get_arch_list()
    if not archs:  # some builds do not report their architectures
        return True, f"sm_{capability}"
    native = f"sm_{capability}" in archs
    # A PTX ("compute_XY") build for an older architecture can be JIT-compiled for newer GPUs.
    ptx = any(a.startswith("compute_") and int(a[8:]) <= capability for a in archs)
    return native or ptx, f"sm_{capability}"


def select_device(preference: str = "auto", dtype: str = "auto", torch: Any = None) -> DeviceConfig:
    """Choose where to run the model and with which precision.

    - CUDA GPU: float16, or bfloat16 on GPUs that support it (Ampere and newer).
      Halves memory versus float32 with negligible quality loss for inference.
    - Apple MPS: float16.
    - CPU: float32. Half precision on CPU is slow or unsupported for many ops.

    A CUDA GPU that the installed PyTorch build was not compiled for (e.g. an old Kaggle
    P100 with a newer PyTorch) falls back to CPU with an explanation instead of crashing.
    """
    torch = torch or require("torch", "local LLM inference")
    preference = preference.lower()
    gpu_name = gpu_memory = None
    reason = ""

    device = "cpu"
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        supported, arch = _cuda_arch_supported(torch)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        if supported:
            device, reason = "cuda", f"CUDA GPU {gpu_name} ({arch})"
        else:
            reason = (
                f"GPU {gpu_name} ({arch}) is not supported by this PyTorch build "
                f"({', '.join(torch.cuda.get_arch_list())}); using CPU instead"
            )
    elif preference == "cuda":
        reason = "CUDA was requested but no CUDA GPU is available; using CPU"
    elif preference in ("auto", "mps") and _mps_available(torch):
        device, reason = "mps", "Apple Silicon GPU (MPS)"
    elif preference not in ("auto", "cpu", "mps"):
        raise ValueError(f"Unknown device preference: {preference!r}")
    if device == "cpu" and not reason:
        reason = "CPU requested" if preference == "cpu" else "No compatible GPU found"

    if dtype == "auto":
        if device == "cuda":
            bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
            dtype = "bfloat16" if bf16 else "float16"
        else:
            dtype = "float16" if device == "mps" else "float32"
    return DeviceConfig(device, dtype, reason, gpu_name, gpu_memory)


def _mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps and mps.is_available())


class HuggingFaceLLM:
    """A causal language model run locally with Transformers. Greedy decoding by default."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id or os.environ.get("LLM_MODEL_ID") or DEFAULT_LLM_MODEL
        self.device_preference = device
        self.dtype_preference = dtype
        self.local_files_only = local_files_only
        self.device_config: DeviceConfig | None = None
        self.tokenizer = None
        self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "HuggingFaceLLM":
        if self.is_loaded:
            return self
        torch = require("torch", "local LLM inference")
        transformers = require("transformers", "local LLM inference")
        config = select_device(self.device_preference, self.dtype_preference, torch)
        logger.info(
            "Loading %s on %s (%s): %s", self.model_id, config.device, config.dtype, config.reason
        )

        # transformers renamed `torch_dtype` to `dtype` in 4.56.
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kwarg = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        started = perf_counter()
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True,
                **{dtype_kwarg: getattr(torch, config.dtype)},
            )
            model.to(config.device)
            model.eval()
            # Greedy decoding: the same prompt always gives the same answer, which makes runs
            # reproducible and evaluation meaningful. Clearing the checkpoint's sampling
            # defaults (temperature, top_p, top_k) avoids warnings, since greedy ignores them.
            defaults = model.generation_config
            defaults.do_sample = False
            defaults.temperature = defaults.top_p = defaults.top_k = None
        except Exception as exc:
            raise LLMLoadError(self._load_error_message(exc, config)) from exc

        self.tokenizer, self.model, self.device_config = tokenizer, model, config
        logger.info("Model loaded in %.1fs", perf_counter() - started)
        return self

    def _load_error_message(self, exc: Exception, config: DeviceConfig) -> str:
        oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        if oom:
            return (
                f"Ran out of memory loading '{self.model_id}' on {config.device} "
                f"({config.dtype}). Try a GPU runtime, a smaller model (set LLM_MODEL_ID, e.g. "
                "'Qwen/Qwen2.5-0.5B-Instruct'), or restart the kernel to free memory."
            )
        return (
            f"Could not load model '{self.model_id}': {type(exc).__name__}: {exc}\n"
            "Check that:\n"
            "  1. The model id is correct, or LLM_MODEL_ID points to a local model folder.\n"
            "  2. Internet is enabled so files can download (Kaggle: Settings > Internet > On),\n"
            "     or the model is attached as a Kaggle Model/Dataset input and you pass its path.\n"
            "  3. HF_HUB_OFFLINE / local_files_only is not set unless the files are cached.\n"
            "  4. transformers is recent enough for this model (Qwen2.5 needs >= 4.37)."
        )

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"model_id": self.model_id, "loaded": self.is_loaded}
        if self.device_config:
            info.update(
                device=self.device_config.device,
                dtype=self.device_config.dtype,
                reason=self.device_config.reason,
                gpu=self.device_config.gpu_name,
                gpu_memory_gb=self.device_config.gpu_memory_gb,
            )
        if self.model is not None:
            info["parameters_millions"] = round(
                sum(p.numel() for p in self.model.parameters()) / 1e6, 1
            )
        return info

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> Generation:
        self.load()
        torch = require("torch", "local LLM inference")
        tokenizer, model = self.tokenizer, self.model

        # The chat template wraps messages in the role markers the model was trained with.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device_config.device)

        pad_id = tokenizer.pad_token_id
        started = perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id if pad_id is not None else tokenizer.eos_token_id,
            )
        duration_ms = (perf_counter() - started) * 1000

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[0, prompt_length:]
        return Generation(
            text=tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            prompt_tokens=int(prompt_length),
            completion_tokens=int(new_tokens.shape[0]),
            duration_ms=round(duration_ms, 1),
        )
