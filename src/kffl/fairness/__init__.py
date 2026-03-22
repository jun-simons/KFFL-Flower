"""Fairness regularisation components for KFFL.

Planned contents:
- Orthogonal Random Feature Maps (ORFMs) for shift-invariant kernels
  (Yu et al. 2016) to approximate K_{f(ω)} and K_S
- KHSIC computation: ψ(ω; X, S) = (1/(n-1)²) Tr(H K_S H K_{f(ω)} H)
- Local interaction matrix Mᵢ(ω) = Z_{S,i}ᵀ Z_{f(X),i}
- Global interaction matrix G(ω) aggregation (Lemma 1)
- Fairness gradient g(ω) = ∇‖G(ω)‖²_F (Corollary 1, Lemma 2)
- Fairness metrics: SPD, EO, KS distance
"""
