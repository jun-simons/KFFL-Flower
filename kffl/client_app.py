"""KFFL - Client Side (Algorithm 2).

Each global iteration inlcudes three rounds with messages coming to the client 

    query.fair1        -> Phi^i(w_t) = {M^i(w_t), mu_{s,i}, mu_{f,i}}   (Eqs. 14, 19, 20)
    query.fair2        -> g^i(w_t)   = J_{Omega_i}(w_t)^T G(w_t)        (Eqs. 15, 16)
    train.local_update -> w^i_{t+1}  = prox step on w_{t+1/2}           (Alg. 1, line 5)

``w_t`` and the shared RFM seed ``zeta`` arrive with FAIR1 and are cached in
``context.state`` for FAIR2, which keeps the downlink ``2|w| + D^2``
"""

from __future__ import annotations

from functools import lru_cache

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from kffl.config import KFFLConfig
from kffl.dataset import get_dataset, partition
from kffl.fairness import RandomFeatureMap, derive_seed
from kffl.fairness.khsic import local_interaction, local_terms
from kffl.ml import build_loss, build_model, solve_local
from kffl.utils import record_to_state, state_to_record, tensors_to_record

app = ClientApp()


# --------------------------------------------------------------------------
# Local data and model
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _load_partition(
    dataset: str,
    sensitive_attribute: str,
    exclude_sensitive: bool,
    scheme: str,
    seed: int,
    majority: float,
    mode: str,
    partition_id: int,
    num_partitions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """The client's share of the training data (Appendix B.3)."""
    ds = get_dataset(dataset, sensitive_attribute, exclude_sensitive)
    idx = partition(
        ds.train.s,
        num_partitions=num_partitions,
        scheme=scheme,
        seed=seed,
        majority=majority,
        mode=mode,
    )[partition_id]
    local = ds.train.subset(idx)
    return (
        torch.from_numpy(local.X),
        torch.from_numpy(local.s),
        torch.from_numpy(local.y),
        ds.num_groups,
    )


def _setup(context: Context):
    """Config, a fresh model, and this client's local data"""
    cfg = KFFLConfig.from_run_config(context.run_config)
    X, s, y, num_groups = _load_partition(
        cfg.dataset,
        cfg.sensitive_attribute,
        cfg.exclude_sensitive_from_features,
        cfg.partition,
        cfg.seed,
        cfg.non_iid_majority,
        cfg.non_iid_mode,
        int(context.node_config["partition-id"]),
        int(context.node_config["num-partitions"]),
    )
    model = build_model(cfg.model, X.shape[1])
    return cfg, model, X, s, y, num_groups


def _encode_sensitive(
    s: torch.Tensor, num_groups: int, encoding: str
) -> torch.Tensor:
    """Input to ``phi_s``.  One-hot matches the "one-hot columns" of Sec. 6.5."""
    if encoding == "one-hot":
        return torch.nn.functional.one_hot(s.long(), num_groups).float()
    if encoding == "raw":
        return s.float().unsqueeze(-1)
    raise ValueError(f"unknown sensitive encoding {encoding!r}")


def _build_rfms(cfg: KFFLConfig, in_dim_s: int, zeta: int):
    """``phi_s`` and ``phi_f``, both mapping into R^D from the shared seed zeta."""
    phi_s = RandomFeatureMap(
        in_dim_s, cfg.rfm_dim, cfg.kernel_gamma_s, derive_seed(zeta, "s"), cfg.rfm_kind
    )
    # phi_f acts on the scalar model score f(x; w).
    phi_f = RandomFeatureMap(
        1, cfg.rfm_dim, cfg.kernel_gamma_f, derive_seed(zeta, "f"), cfg.rfm_kind
    )
    return phi_s, phi_f


def _model_output(model, X: torch.Tensor, how: str) -> torch.Tensor:
    """``f(X; w)`` as fed to ``phi_f`` (App. G: "the model output, e.g. logits")."""
    scores = model(X)
    return torch.sigmoid(scores) if how == "prob" else scores


# --------------------------------------------------------------------------
# FAIR1 (Algorithm 2, lines 1-4)
# --------------------------------------------------------------------------


@app.query("fair1")
def fair1(msg: Message, context: Context) -> Message:
    cfg, model, X, s, y, num_groups = _setup(context)
    model.load_state_dict(record_to_state(msg.content["arrays"]))

    zeta = int(msg.content["config"]["zeta"])
    s_enc = _encode_sensitive(s, num_groups, cfg.sensitive_encoding)
    phi_s, phi_f = _build_rfms(cfg, s_enc.shape[1], zeta)

    with torch.no_grad():
        Z_s = phi_s(s_enc)
        Z_f = phi_f(_model_output(model, X, cfg.fairness_output))
        terms = local_terms(Z_s, Z_f)  # Eqs. (14), (19), (20)

    # Cache w_t and zeta: FAIR2 needs both and the server does not resend them.
    context.state["w_t"] = msg.content["arrays"]
    context.state["fair"] = ConfigRecord({"zeta": zeta})

    return Message(
        content=RecordDict(
            {
                "arrays": tensors_to_record(
                    M=terms.M, mu_s=terms.mu_s, mu_f=terms.mu_f
                ),
                "metrics": MetricRecord({"n_i": terms.n_i}),
            }
        ),
        reply_to=msg,
    )


# --------------------------------------------------------------------------
# FAIR2 (Algorithm 2, lines 5-8)
# --------------------------------------------------------------------------


@app.query("fair2")
def fair2(msg: Message, context: Context) -> Message:
    cfg, model, X, s, y, num_groups = _setup(context)
    model.load_state_dict(record_to_state(context.state["w_t"]))

    zeta = int(context.state["fair"]["zeta"])
    s_enc = _encode_sensitive(s, num_groups, cfg.sensitive_encoding)
    phi_s, phi_f = _build_rfms(cfg, s_enc.shape[1], zeta)

    G = torch.from_numpy(msg.content["arrays"]["G"].numpy())
    mu_s = torch.from_numpy(msg.content["arrays"]["mu_s"].numpy())
    mu_f = torch.from_numpy(msg.content["arrays"]["mu_f"].numpy())

    # Recompute M^i(w_t) under autograd.  Algorithm 2 line 2 says to *store*
    # M^i from FAIR1, but a stored value carries no graph; the forward pass has
    # to be replayed to get the Jacobian. Same seed zeta => same feature space.
    Z_s = phi_s(s_enc)
    Z_f = phi_f(_model_output(model, X, cfg.fairness_output))
    M = Z_s.T @ Z_f  # Eq. (14)

    Omega = local_interaction(  # Eq. (15)
        M=M,
        mu_s=mu_s,
        mu_f_local=Z_f.mean(dim=0),
        mu_f_global=mu_f,
        n_i=X.shape[0],
        variant=cfg.omega_variant,
    )

    # Eq. (16): g^i = J_{Omega_i}(w)^T G, i.e. the gradient of <Omega_i, G>_F
    # with G held fixed.
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    grads = torch.autograd.grad((Omega * G).sum(), params)

    return Message(
        content=RecordDict({"arrays": state_to_record(dict(zip(names, grads)))}),
        reply_to=msg,
    )


# --------------------------------------------------------------------------
# Local Update (Algorithm 2, lines 9-11)
# --------------------------------------------------------------------------


@app.train("local_update")
def local_update(msg: Message, context: Context) -> Message:
    cfg, model, X, s, y, _ = _setup(context)
    model.load_state_dict(record_to_state(msg.content["arrays"]))

    prox_mu = float(msg.content["config"]["prox_mu"])
    server_round = int(msg.content["config"]["round"])

    generator = torch.Generator().manual_seed(
        cfg.seed + 1000 * server_round + int(context.node_config["partition-id"])
    )
    loss = solve_local(  # Alg. 1, line 5
        model,
        X,
        y,
        build_loss(cfg.task),
        epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.local_lr,
        optimizer=cfg.local_optimizer,
        prox_mu=prox_mu,
        generator=generator,
    )

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord({"n_i": int(X.shape[0]), "loss": loss}),
            }
        ),
        reply_to=msg,
    )
