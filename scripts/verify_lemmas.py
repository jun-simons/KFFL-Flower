"""Check the federated decompositions against their centralised definitions.

On a small synthetic problem, verifies:

  Lemma 1   sum_i M^i - n mu_s mu_f^T  ==  Z_s^T H Z_f                (Eq. 13)
  Theorem 2 ||G||_F^2 / (n-1)^2        ~=  Tr(K_s H K_f H) / (n-1)^2  (Eq. 9)
  Lemma 2   sum_i J_{Omega_i}^T G      ==  (1/2) grad ||G(w)||_F^2    (Eq. 16)

and reports how far the literal reading of Eq. (15) lands from the exact
gradient.  Run in float64: the app itself uses float32, where the identities
hold only to ~1e-4 relative because the centralised reference forms a dense
n x n H and cancels O(n) terms against each other.

Run:  PYTHONPATH=. python scripts/verify_lemmas.py
"""

from __future__ import annotations

import torch

from kffl.fairness.khsic import global_interaction, local_interaction, local_terms
from kffl.fairness.rfm import RandomFeatureMap, derive_seed
from kffl.ml import build_model

DTYPE = torch.float64
N, D_IN, M_CLIENTS, D, K, GAMMA = 400, 6, 4, 32, 2, 0.5
ZETA = 12345

torch.manual_seed(0)
X = torch.randn(N, D_IN, dtype=DTYPE)
s = (torch.rand(N) < 0.4).long()
s_enc = torch.nn.functional.one_hot(s, K).to(DTYPE)
shards = torch.chunk(torch.arange(N), M_CLIENTS)
H = torch.eye(N, dtype=DTYPE) - torch.ones(N, N, dtype=DTYPE) / N

model = build_model("logistic-regression", D_IN).to(DTYPE)
params = [p for _, p in model.named_parameters()]
phi_s = RandomFeatureMap(K, D, GAMMA, derive_seed(ZETA, "s")).to(DTYPE)
phi_f = RandomFeatureMap(1, D, GAMMA, derive_seed(ZETA, "f")).to(DTYPE)


def centralised_G() -> torch.Tensor:
    """``G = Z_s^T H Z_f``, the definition Lemma 1 decomposes."""
    return phi_s(s_enc).T @ H @ phi_f(model(X))


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


# -- Lemma 1 ----------------------------------------------------------------
terms = [local_terms(phi_s(s_enc[i]), phi_f(model(X[i]))) for i in shards]
G_fed, mu_s, mu_f, n = global_interaction(terms)
G_ctr = centralised_G()
err = rel_err(G_fed.detach(), G_ctr.detach())
print(f"Lemma 1     G_fed vs G_central          rel err = {err:.2e}")
assert err < 1e-10, "Lemma 1 failed"

# -- Theorem 2: how well the D-dimensional RFM tracks the true KHSIC ---------
with torch.no_grad():
    f_out = model(X)
    K_s = torch.exp(-GAMMA * torch.cdist(s_enc, s_enc) ** 2)
    K_f = torch.exp(-GAMMA * torch.cdist(f_out, f_out) ** 2)
    exact = torch.trace(K_s @ H @ K_f @ H) / (N - 1) ** 2
    approx = (G_ctr**2).sum() / (N - 1) ** 2
print(f"Theorem 2   psi exact = {exact:.3e}   psi via RFM (D={D}) = {approx:.3e}")

# -- Lemma 2 ----------------------------------------------------------------
grad_ctr = torch.autograd.grad(0.5 * (centralised_G() ** 2).sum(), params)
G_const = G_fed.detach()

for variant in ("exact", "paper-literal"):
    total = [torch.zeros_like(p) for p in params]
    for idx in shards:
        Z_s, Z_f = phi_s(s_enc[idx]), phi_f(model(X[idx]))
        Omega = local_interaction(  # Eq. (15)
            M=Z_s.T @ Z_f,
            mu_s=mu_s.detach(),
            mu_f_local=Z_f.mean(0),
            mu_f_global=mu_f.detach(),
            n_i=len(idx),
            variant=variant,
        )
        # Eq. (16): g^i = J_{Omega_i}^T G
        for acc, gi in zip(total, torch.autograd.grad((Omega * G_const).sum(), params)):
            acc += gi
    err = max(rel_err(a, b) for a, b in zip(total, grad_ctr))
    print(f"Lemma 2     variant={variant:<14} rel err = {err:.2e}")
    if variant == "exact":
        assert err < 1e-10, "Lemma 2 (exact) failed"

print("\nOK")
