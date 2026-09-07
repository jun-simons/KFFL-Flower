# KFFL on Flower

A implementation of **KFFL** — *Group Fair Federated Learning via Stochastic
Kernel Regularization* (Arif, Murugesan, Chen & Gittens) with the Flower **Message
API**.

KFFL needs three client interactions per global iteration (FAIR1 / FAIR2 / Local
Update), which Flower's strategy-level API (currently) doesnt support so we use
`grid.send_and_receive` directly and clients register three named actions.

```
kffl/
├── server_app.py     Algorithm 3 (server side)
├── client_app.py     Algorithm 2 (client side)
├── config.py         typed view over [tool.flwr.app.config]
├── dataset/          loaders (registry), preprocessing, IID / non-IID partitioners
├── ml/               models (registry), losses, the proximal local solver
├── fairness/         random feature maps, the KHSIC regulariser, fairness metrics
└── utils/            RecordDict <-> tensor helpers
```

Everything is selected from `pyproject.toml`, so you can swap the dataset or model
config key here. Add a loader to `kffl/dataset/__init__.py::LOADERS` or a model to
`kffl/ml/models.py::MODELS` to extend either registry.

## Running

```bash
flwr run . --run-config 'algorithm="kffl" fairness-lambda=330.0 partition="non-iid"'
```

```bash
flwr run . --run-config 'algorithm="fedavg"'
```

Sweep the fairness weight over the Appendix B.5 grid and write a CSV:

```bash
PYTHONPATH=. python scripts/sweep.py --partition iid
```

Check the paper's decompositions against their centralised definitions:

```bash
PYTHONPATH=. python scripts/verify_lemmas.py
```

## Protocol

One global iteration `t`, with per-client communication cost in brackets
(reproducing Table 2's `2|w| + D^2` in each direction):

| Sub-round | Server → client | Client computes | Client → server |
|---|---|---|---|
| **FAIR1** | `w_t`, seed `zeta` [`\|w\|`] | `M^i = Z_{s,i}^T Z_{f,i}` (14), `mu_{s,i}` (19), `mu_{f,i}` (20) | `Phi^i` [`D^2 + 2D`] |
| — | server: `mu_s` (17), `mu_f` (18), `G = sum_i M^i - n mu_s mu_f^T` (13) | | |
| **FAIR2** | `G`, `mu_s`, `mu_f` `[D^2 + 2D]` | `Omega_i` (15), `g^i = J_{Omega_i}^T G` (16) | `g^i` [`\|w\|`] |
| — | server: `w_{t+1/2} = w_t - alpha lambda sum_i g^i` (Alg. 1 line 3) | | |
| **Update** | `w_{t+1/2}` `[\|w\|]` | `argmin_w l^i(w) + (1/2a)||w - w_{t+1/2}||^2` (Alg. 1 line 5) | `w^i_{t+1}` [`\|w\|`] |
| — | server: `w_{t+1} = average(w^i_{t+1})` (Alg. 1 line 7) | | |

`w_t` and `zeta` are cached in `context.state` at FAIR1 and reused at FAIR2, so the
server never resends them — that is what holds the downlink at `2|w| + D^2`.
