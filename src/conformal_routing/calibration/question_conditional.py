"""Question-conditional calibrator (Route 2  - backup).

Idea: STEER fits ONE GMM globally. Difficulty distributions differ across question
types (math vs code vs multi-hop QA). We cluster questions in embedding space,
then fit a per-cluster calibrator (we delegate to ANY base calibrator: GMM or
Conformal  - this composes nicely).

This file is intentionally short  - it's a wrapper that holds K base calibrators.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from conformal_routing.calibration.base import Calibrator, FitInputs, RouteDecision


class QuestionConditionalCalibrator(Calibrator):
    name = "qcond"

    def __init__(
        self,
        base_factory,                # callable () -> Calibrator
        n_clusters: int = 8,
        random_state: int = 0,
        min_per_cluster: int = 30,
    ):
        self.base_factory = base_factory
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.min_per_cluster = min_per_cluster
        self.kmeans: KMeans | None = None
        self.calibrators_: dict[int, Calibrator] = {}
        self.fallback_: Calibrator | None = None

    def fit(self, data: FitInputs) -> None:
        if data.question_embeds is None or data.question_ids is None:
            raise ValueError(
                "QuestionConditionalCalibrator requires question_embeds and question_ids."
            )
        # Cluster at the QUESTION level (not step level), then assign step labels.
        unique_qids, inv = np.unique(data.question_ids, return_inverse=True)
        # Take the embedding of each unique question (assumes question_embeds is per-step
        # but constant within question; pick the first occurrence for each q).
        q_embeds = np.zeros((len(unique_qids), data.question_embeds.shape[1]), dtype=np.float32)
        seen = set()
        for i, qid in enumerate(data.question_ids):
            if qid not in seen:
                q_embeds[np.searchsorted(unique_qids, qid)] = data.question_embeds[i]
                seen.add(qid)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(q_embeds)
        q_clusters = self.kmeans.predict(q_embeds)        # (Q,)
        step_clusters = q_clusters[inv]                   # (N_steps,)

        # Always fit a fallback global calibrator on full data.
        self.fallback_ = self.base_factory()
        self.fallback_.fit(data)

        # Per-cluster calibrators.
        for c in range(self.n_clusters):
            mask = step_clusters == c
            if mask.sum() < self.min_per_cluster:
                continue
            sub = FitInputs(
                scores=data.scores[mask],
                small_correct=data.small_correct[mask],
            )
            cal = self.base_factory()
            try:
                cal.fit(sub)
                self.calibrators_[c] = cal
            except Exception:
                # If fitting fails (e.g. too few correct examples for conformal), skip.
                continue

    def _cluster_for(self, embed: np.ndarray | None) -> int | None:
        if embed is None or self.kmeans is None:
            return None
        c = int(self.kmeans.predict(embed.reshape(1, -1))[0])
        return c if c in self.calibrators_ else None

    def decide(self, score: float, question_embed=None) -> RouteDecision:
        c = self._cluster_for(question_embed)
        cal = self.calibrators_.get(c, self.fallback_) if c is not None else self.fallback_
        assert cal is not None
        return cal.decide(score, question_embed)

    def confidence(self, score: float, question_embed=None) -> float:
        c = self._cluster_for(question_embed)
        cal = self.calibrators_.get(c, self.fallback_) if c is not None else self.fallback_
        assert cal is not None
        return cal.confidence(score, question_embed)

