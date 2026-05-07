"""Models subpackage. VLLMWrapper is lazy-imported to keep CPU-only tests light."""
from conformal_routing.models.base import FirstTokenProbe, ModelWrapper, StepOutput
from conformal_routing.models.factory import build_model

__all__ = ["FirstTokenProbe", "ModelWrapper", "StepOutput", "VLLMWrapper", "build_model"]


def __getattr__(name):
    if name == "VLLMWrapper":
        from conformal_routing.models.vllm_wrapper import VLLMWrapper
        return VLLMWrapper
    raise AttributeError(name)

