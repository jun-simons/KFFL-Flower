"""Client data distributions (Appendix B.3).

IID
    "each client is provided with an equal number of samples and a consistent
    data distribution for local training".

Non-IID
    "half of the clients have 90% of A_0 and 10% of A_1 while the other half
    has 90% of A_1 and 10% of A_0".
"""

from __future__ import annotations

import numpy as np


def partition(
    s: np.ndarray,
    num_partitions: int,
    scheme: str,
    seed: int,
    majority: float = 0.9,
    mode: str = "exact-ratio",
) -> list[np.ndarray]:
    """Return one index array per client.

    Parameters
    ----------
    mode : {"exact-ratio", "use-all"}
        Only used for ``scheme="non-iid"``.  The paper's 90/10 prescription is
        infeasible at full size whenever the global groups are imbalanced (in
        ADULT, sex is ~2:1), and the paper does not say how it resolves this.

        ``exact-ratio`` gives every client exactly the requested composition by
        shrinking the (uniform) client size until the demand fits the pools;
        surplus samples of the majority group are dropped.

        ``use-all`` keeps every sample by splitting each group's pool in
        proportion to the clients' demand for it, which uses all the data at
        the price of only approximating the 90/10 ratios.
    """
    rng = np.random.default_rng(seed)

    if scheme == "iid":
        idx = rng.permutation(len(s))
        return [np.sort(part) for part in np.array_split(idx, num_partitions)]

    if scheme != "non-iid":
        raise ValueError(f"unknown partition scheme {scheme!r}")

    groups = np.unique(s)
    if len(groups) != 2:
        raise ValueError(
            "the non-IID scheme of App. B.3 is defined for a binary sensitive "
            f"attribute; got K={len(groups)}"
        )
    pools = {g: rng.permutation(np.flatnonzero(s == g)) for g in groups}

    # Client j takes `majority` of group `maj[j]` and `1 - majority` of the other.
    half = num_partitions // 2
    maj = [groups[0]] * half + [groups[1]] * (num_partitions - half)
    # demand[j][g] is client j's share of its own size drawn from group g.
    demand = [
        {g: (majority if g == maj[j] else 1.0 - majority) for g in groups}
        for j in range(num_partitions)
    ]

    parts: list[list[np.ndarray]] = [[] for _ in range(num_partitions)]
    if mode == "exact-ratio":
        # Largest uniform client size for which every group's demand fits.
        size = min(
            int(len(pools[g]) // sum(d[g] for d in demand)) for g in groups
        )
        cursor = {g: 0 for g in groups}
        for j in range(num_partitions):
            for g in groups:
                take = int(round(demand[j][g] * size))
                parts[j].append(pools[g][cursor[g] : cursor[g] + take])
                cursor[g] += take
    elif mode == "use-all":
        for g in groups:
            weights = np.array([d[g] for d in demand], dtype=float)
            cuts = np.cumsum(weights / weights.sum() * len(pools[g])).astype(int)[:-1]
            for j, chunk in enumerate(np.split(pools[g], cuts)):
                parts[j].append(chunk)
    else:
        raise ValueError(f"unknown non-iid mode {mode!r}")

    return [np.sort(np.concatenate(p)) for p in parts]
