"""LLM loader tests with fake torch/transformers modules. No model is downloaded or run."""

import sys
from types import SimpleNamespace

import pytest

from app.rag.deps import MissingDependencyError
from app.services.llm import DEFAULT_LLM_MODEL, HuggingFaceLLM, LLMLoadError, select_device


def fake_torch(*, cuda=False, capability=(7, 5), archs=("sm_75", "sm_80"), bf16=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_capability=lambda index=0: capability,
            get_arch_list=lambda: list(archs),
            get_device_name=lambda index=0: "Tesla T4",
            get_device_properties=lambda index=0: SimpleNamespace(total_memory=16 * 1024**3),
            is_bf16_supported=lambda: bf16,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_cuda_t4_uses_float16():
    config = select_device(torch=fake_torch(cuda=True))

    assert (config.device, config.dtype) == ("cuda", "float16")
    assert config.gpu_name == "Tesla T4" and config.gpu_memory_gb == 16.0


def test_ampere_gpu_uses_bfloat16():
    torch = fake_torch(cuda=True, capability=(8, 0), bf16=True)

    assert select_device(torch=torch).dtype == "bfloat16"


def test_gpu_unsupported_by_pytorch_build_falls_back_to_cpu():
    # e.g. a P100 (sm_60) with a PyTorch build compiled only for sm_70+
    config = select_device(torch=fake_torch(cuda=True, capability=(6, 0), archs=("sm_70",)))

    assert (config.device, config.dtype) == ("cpu", "float32")
    assert "not supported by this PyTorch build" in config.reason


def test_ptx_build_supports_newer_gpu():
    torch = fake_torch(cuda=True, capability=(9, 0), archs=("sm_80", "compute_80"))

    assert select_device(torch=torch).device == "cuda"


def test_cpu_and_mps_selection():
    assert select_device(torch=fake_torch()).device == "cpu"
    assert select_device(torch=fake_torch(mps=True)).device == "mps"
    assert select_device("cpu", torch=fake_torch(cuda=True)).device == "cpu"
    assert select_device("auto", "float32", torch=fake_torch(cuda=True)).dtype == "float32"
    cpu_fallback = select_device("cuda", torch=fake_torch())
    assert cpu_fallback.device == "cpu" and "no CUDA GPU" in cpu_fallback.reason
    with pytest.raises(ValueError):
        select_device("tpu", torch=fake_torch())


def test_constructing_llm_is_lazy(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)  # any import attempt would fail

    llm = HuggingFaceLLM()

    assert llm.model_id == DEFAULT_LLM_MODEL and not llm.is_loaded
    assert llm.describe() == {"model_id": DEFAULT_LLM_MODEL, "loaded": False}


def test_model_id_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    assert HuggingFaceLLM().model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert HuggingFaceLLM("local/path").model_id == "local/path"


def test_missing_transformers_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(MissingDependencyError, match="transformers"):
        HuggingFaceLLM().load()


def test_download_failure_gives_actionable_error(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    transformers = SimpleNamespace(
        __version__="4.46.0",
        AutoTokenizer=SimpleNamespace(from_pretrained=offline),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=offline),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(LLMLoadError) as exc_info:
        HuggingFaceLLM("Qwen/Qwen2.5-1.5B-Instruct").load()

    message = str(exc_info.value)
    assert "Qwen/Qwen2.5-1.5B-Instruct" in message
    assert "Internet" in message and "LLM_MODEL_ID" in message
