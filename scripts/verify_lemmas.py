"""Check the federated decompositions against their centralised definitions.

On a small synthetic problem, for the paper's KHSIC (one stratum) and for the
label-conditional extension (one stratum per label y), verifies:

  Lemma 1   sum_i M^{i,y} - n_y mu_{s,y} mu_{f,y}^T  ==  Z_{s,y}^T H_y Z_{f,y}   (Eq. 13)
  Theorem 2 sum_y w_y ||G_y||_F^2 / (n_y-1)^2  ~=  the exact kernel value        (Eq. 9)
  Lemma 2   sum_i sum_y c_y J_{Omega_{i,y}}^T G_y  ==  grad psi                   (Eq. 16)

and reports how far the literal reading of Eq. (15) lands from the exact
gradient.  Run in float64: the app itself uses float32, where the identities
hold only to ~1e-4 relative because the centralised reference cancels O(n)
terms against each other.

Run:  PYTHONPATH=. python scripts/verify_lemmas.py
"""

from __future__ import annotations

import torch

from kffl.fairness.khsic import (
    global_interaction,
    local_interaction,
    local_terms,
    psi_values,
    strata_onehot,
    stratum_scales,
    stratum_weights,
)
from kffl.fairness.rfm import RandomFeatureMap, derive_seed
from kffl.ml import build_model

DTYPE = torch.float64
N, D_IN, M_CLIENTS, D, K, GAMMA = 400, 6, 4, 32, 2, 0.5
ZETA, MODE = 12345, "n-minus-1-sq"

torch.manual_seed(0)
X = torch.randn(N, D_IN, dtype=DTYPE)
s = (torch.rand(N) < 0.4).long()
y = ((X[:, 0] + 0.5 * s + 0.5 * torch.randn(N)) > 0).float()  # labels correlated with s
s_enc = torch.nn.functional.one_hot(s, K).to(DTYPE)
shards = torch.chunk(torch.arange(N), M_CLIENTS)

model = build_model("logistic-regression", D_IN).to(DTYPE)
params = [p for _, p in model.named_parameters()]
phi_s = RandomFeatureMap(K, D, GAMMA, derive_seed(ZETA, "s")).to(DTYPE)
phi_f = RandomFeatureMap(1, D, GAMMA, derive_seed(ZETA, "f")).to(DTYPE)


def centred(Z: torch.Tensor) -> torch.Tensor:
    return Z - Z.mean(dim=0)  # H Z


def central_G(R: torch.Tensor) -> torch.Tensor:
    """``G_y = Z_{s,y}^T H_y Z_{f,y}`` from the pooled data, one stratum at a time."""
    Z_s, Z_f = phi_s(s_enc), phi_f(model(X))
    return torch.stack([centred(Z_s[R[:, c] > 0]).T @ Z_f[R[:, c] > 0] for c in range(R.shape[1])])


def central_psi(R: torch.Tensor, weights: str) -> torch.Tensor:
    n = R.sum(0)
    return (stratum_weights(n, weights) * psi_values(central_G(R), n, MODE)).sum()


def exact_psi(R: torch.Tensor, weights: str) -> torch.Tensor:
    """The n_y x n_y kernel form of Eq. (9) inside each stratum."""
    total, n = 0.0, R.sum(0)
    w = stratum_weights(n, weights)
    f = model(X)
    for c in range(R.shape[1]):
        idx = R[:, c] > 0
        K_s = torch.exp(-GAMMA * torch.cdist(s_enc[idx], s_enc[idx]) ** 2)
        K_f = torch.exp(-GAMMA * torch.cdist(f[idx], f[idx]) ** 2)
        H = torch.eye(int(n[c]), dtype=DTYPE) - 1.0 / n[c]
        total = total + w[c] * torch.trace(K_s @ H @ K_f @ H) / (n[c] - 1) ** 2
    return total


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


for conditioning, weights in (("none", "prevalence"), ("label", "prevalence"), ("label", "uniform")):
    print(f"--- khsic-conditioning={conditioning}  khsic-stratum-weights={weights}")
    R = strata_onehot(y, conditioning, 2).to(DTYPE)

    # -- Lemma 1 ------------------------------------------------------------
    terms = [local_terms(phi_s(s_enc[i]), phi_f(model(X[i])), R[i]) for i in shards]
    G, mu_s, mu_f, n = global_interaction(terms)
    err = rel_err(G.detach(), central_G(R).detach())
    print(f"Lemma 1     G_y federated vs centralised      rel err = {err:.2e}")
    assert err < 1e-10, "Lemma 1 failed"

    # -- Theorem 2: how well the D-dimensional RFM tracks the exact kernel ---
    with torch.no_grad():
        print(f"Theorem 2   psi exact = {float(exact_psi(R, weights)):.3e}   "
              f"psi via RFM (D={D}) = {float(central_psi(R, weights)):.3e}")

    # -- Lemma 2 ------------------------------------------------------------
    grad_ref = torch.autograd.grad(central_psi(R, weights), params)
    G_const, scale = G.detach(), stratum_scales(n, MODE, weights)
    for variant in ("exact", "paper-literal"):
        total = [torch.zeros_like(p) for p in params]
        for i in shards:
            local = local_terms(phi_s(s_enc[i]), phi_f(model(X[i])), R[i])
            Omega = local_interaction(  # Eq. (15)
                M=local.M, mu_s=mu_s.detach(), mu_f_local=local.mu_f,
                mu_f_global=mu_f.detach(), n_i=local.n, variant=variant,
            )
            # Eq. (16) with the stratum scales c_y applied on the client
            objective = (scale[:, None, None] * Omega * G_const).sum()
            for acc, gi in zip(total, torch.autograd.grad(objective, params)):
                acc += gi
        err = max(rel_err(a, b) for a, b in zip(total, grad_ref))
        print(f"Lemma 2     variant={variant:<14} rel err = {err:.2e}")
        if variant == "exact":
            assert err < 1e-10, "Lemma 2 (exact) failed"

# -- Sanity: a predictor that is a function of the label alone -----------------
R = strata_onehot(y, "label", 2).to(DTYPE)
Z_f = phi_f(3.0 * y.to(DTYPE).unsqueeze(-1) - 1.0)  # f(x) = 3y - 1
G_perfect, _, _, n = global_interaction([local_terms(phi_s(s_enc), Z_f, R)])
G_uncond, _, _, n1 = global_interaction([local_terms(phi_s(s_enc), Z_f, torch.ones(N, 1, dtype=DTYPE))])
print("--- predictor f = 3y - 1 (depends on the label only)")
print(f"conditional psi = {float(psi_values(G_perfect, n, MODE).sum()):.2e}   "
      f"unconditional psi = {float(psi_values(G_uncond, n1, MODE).sum()):.2e}")
assert float(psi_values(G_perfect, n, MODE).sum()) < 1e-20

print("\nOK")
