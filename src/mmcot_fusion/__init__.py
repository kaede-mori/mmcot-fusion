"""mmcot-fusion: MM-CoT reimplementation with pluggable cross-modal fusion."""

__version__ = "0.1.0"

from .models.fusion import (  # noqa: F401
    FUSION_REGISTRY,
    SigmoidGatedFusion,
    TanhGatedMHAFusion,
    build_fusion,
)


def __getattr__(name):
    # heavyweight import (transformers) kept lazy
    if name == "T5ForMultimodalGeneration":
        from .models.t5_multimodal import T5ForMultimodalGeneration

        return T5ForMultimodalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
