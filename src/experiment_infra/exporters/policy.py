"""Explicit sampling and experiment-lineage split policy for training exports."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class ExportPolicy(BaseModel):
    """Canonical exports are lossless; this policy applies only to training rows.

    Replays are audits rather than fresh agent decisions. Including them requires an
    explicit opt-in and carries a visible weight. Incomplete SFT contexts likewise
    require opt-in; missing context is never reconstructed from execution state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    include_unknown_purpose: bool = False
    include_exact_replays: bool = False
    exact_replay_weight: float = Field(default=0.1, gt=0, le=1)
    include_partial_context: bool = False
    include_unknown_context: bool = False
    test_fraction: float = Field(default=0.2, ge=0, le=1)
    split_seed: str = "experiment-infra-v2"

    def split(self, lineage_root_experiment_id: str) -> str:
        # No action, run, branch, or export identifier may affect the split.
        key = f"{self.split_seed}\0{lineage_root_experiment_id}".encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest(), "big")
        return "test" if bucket < self.test_fraction * (1 << 256) else "train"

    def weight(self, purpose: str) -> float | None:
        if purpose == "unknown":
            return 1.0 if self.include_unknown_purpose else None
        if purpose == "exact_replay":
            return self.exact_replay_weight if self.include_exact_replays else None
        return 1.0


def training_metadata(trajectory, policy):
    root = trajectory.lineage_root_experiment_id or trajectory.experiment_id
    return {
        "lineage_root_experiment_id": root,
        "split_group": root,
        "split": policy.split(root),
        "run_purpose": trajectory.purpose,
        "sample_weight": policy.weight(trajectory.purpose),
    }
