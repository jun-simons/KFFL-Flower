"""Typed view over the Flower run config (``[tool.flwr.app.config]``).

Every field maps 1:1 to a key in ``pyproject.toml`` 
The whole experiment is reproducible from a single ``flwr run --run-config`` string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class KFFLConfig:
    """Run configuration for KFFL / FedAvg."""

    # algorithm
    algorithm: str

    # data (App. B.2 / B.3)
    dataset: str
    sensitive_attribute: str
    partition: str
    non_iid_majority: float
    non_iid_mode: str
    exclude_sensitive_from_features: bool

    # model / task (App. B.4)
    model: str
    task: str

    # optimisation (App. B.1)
    num_server_rounds: int
    local_epochs: int
    batch_size: int
    local_lr: float
    local_optimizer: str
    alpha: float
    prox_mu_raw: float
    aggregation: str

    # fairness regulariser
    fairness_lambda: float
    rfm_dim: int
    rfm_kind: str
    kernel_gamma_s: float
    kernel_gamma_f: float
    rfm_resample: str
    fairness_output: str
    sensitive_encoding: str

    # fidelity switches
    psi_normalization: str
    omega_variant: str
    fairness_step_size_raw: float

    # misc
    seed: int
    results_path: str

    # -- derived ------------------------------------------------------------

    @property
    def prox_mu(self) -> float:
        """Coefficient of the proximal term in Algorithm 1, line 5.

        A negative ``prox-mu`` selects the literal coupling of Algorithm 1,
        ``1/(2*alpha)``; ``prox-mu = 0`` disables the proximal term entirely
        (a plain FedAvg-style local solver).  See the README for why the
        literal coupling cannot reproduce the paper's own accuracies.
        """
        return self.prox_mu_raw if self.prox_mu_raw >= 0.0 else 1.0 / (2.0 * self.alpha)

    @property
    def fairness_step_size(self) -> float:
        """Step size on the fairness gradient, ``alpha`` in Algorithm 1 line 3.

        Algorithm 3 line 14 drops it (``w_{t+1/2} <- w_t - sum_i g^i``); we keep
        it, and allow it to be decoupled from the proximal ``alpha``.
        """
        return (
            self.fairness_step_size_raw
            if self.fairness_step_size_raw >= 0.0
            else self.alpha
        )

    @property
    def uses_fairness(self) -> bool:
        return self.algorithm == "kffl"

    @classmethod
    def from_run_config(cls, run_config: Mapping[str, Any]) -> "KFFLConfig":
        g = run_config.__getitem__
        return cls(
            algorithm=str(g("algorithm")),
            dataset=str(g("dataset")),
            sensitive_attribute=str(g("sensitive-attribute")),
            partition=str(g("partition")),
            non_iid_majority=float(g("non-iid-majority")),
            non_iid_mode=str(g("non-iid-mode")),
            exclude_sensitive_from_features=bool(g("exclude-sensitive-from-features")),
            model=str(g("model")),
            task=str(g("task")),
            num_server_rounds=int(g("num-server-rounds")),
            local_epochs=int(g("local-epochs")),
            batch_size=int(g("batch-size")),
            local_lr=float(g("local-lr")),
            local_optimizer=str(g("local-optimizer")),
            alpha=float(g("alpha")),
            prox_mu_raw=float(g("prox-mu")),
            aggregation=str(g("aggregation")),
            fairness_lambda=float(g("fairness-lambda")),
            rfm_dim=int(g("rfm-dim")),
            rfm_kind=str(g("rfm-kind")),
            kernel_gamma_s=float(g("kernel-gamma-s")),
            kernel_gamma_f=float(g("kernel-gamma-f")),
            rfm_resample=str(g("rfm-resample")),
            fairness_output=str(g("fairness-output")),
            sensitive_encoding=str(g("sensitive-encoding")),
            psi_normalization=str(g("psi-normalization")),
            omega_variant=str(g("omega-variant")),
            fairness_step_size_raw=float(g("fairness-step-size")),
            seed=int(g("seed")),
            results_path=str(g("results-path")),
        )
