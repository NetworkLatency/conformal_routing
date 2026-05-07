from conformal_routing.signals.base import SignalContext, SignalExtractor
from conformal_routing.signals.h_init import HInitSignal
from conformal_routing.signals.logit_confidence import LogitConfidenceSignal
from conformal_routing.signals.self_consistency import SelfConsistencySignal

SIGNAL_REGISTRY: dict[str, type[SignalExtractor]] = {
    "h_init": HInitSignal,
    "logit_confidence": LogitConfidenceSignal,
    "self_consistency": SelfConsistencySignal,
}


def build_signal(name: str, **kwargs) -> SignalExtractor:
    if name not in SIGNAL_REGISTRY:
        raise KeyError(f"Unknown signal {name}. Available: {list(SIGNAL_REGISTRY)}")
    return SIGNAL_REGISTRY[name](**kwargs)


__all__ = [
    "SIGNAL_REGISTRY",
    "SignalContext",
    "SignalExtractor",
    "HInitSignal",
    "LogitConfidenceSignal",
    "SelfConsistencySignal",
    "build_signal",
]

