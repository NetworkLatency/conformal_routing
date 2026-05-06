"""Calibration subpackage."""
from conformal_routing.calibration.base import Calibrator, FitInputs, RouteDecision
from conformal_routing.calibration.conformal import ConformalCalibrator
from conformal_routing.calibration.gmm import GMMCalibrator
from conformal_routing.calibration.question_conditional import QuestionConditionalCalibrator

CALIBRATOR_REGISTRY: dict[str, type[Calibrator]] = {
    "gmm": GMMCalibrator,
    "conformal": ConformalCalibrator,
    "qcond": QuestionConditionalCalibrator,
}


def build_calibrator(name: str, **kwargs) -> Calibrator:
    if name not in CALIBRATOR_REGISTRY:
        raise KeyError(f"Unknown calibrator {name}. Available: {list(CALIBRATOR_REGISTRY)}")
    return CALIBRATOR_REGISTRY[name](**kwargs)


__all__ = [
    "CALIBRATOR_REGISTRY",
    "Calibrator",
    "ConformalCalibrator",
    "FitInputs",
    "GMMCalibrator",
    "QuestionConditionalCalibrator",
    "RouteDecision",
    "build_calibrator",
]


def __getattr__(name):
    # Lazy: collect.py uses tqdm and ModelWrapper which pulls torch transitively in some setups.
    if name in {"CalibrationExample", "collect_with_outcome_propagation",
                "collect_with_agreement", "to_fit_inputs"}:
        from conformal_routing.calibration import collect as _collect
        return getattr(_collect, name)
    raise AttributeError(name)
