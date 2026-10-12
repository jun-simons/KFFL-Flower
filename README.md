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
├── dataset/          loaders (registry: adult, compas), preprocessing, IID / non-IID partitioners
├── ml/               models (registry), losses, the proximal local solver
├── fairness/         random feature maps, the KHSIC regulariser, fairness metrics
└── utils/            RecordDict <-> tensor helpers
```

Everything is configurerd in `pyproject.toml`, so you can swap the dataset or model
config key here. Add a loader to `kffl/dataset/__init__.py::LOADERS` or a model to
`kffl/ml/models.py::MODELS` to extend either registry.

## Running

```bash
flwr run . --run-config 'algorithm="kffl" fairness-lambda=330.0 partition="non-iid"'
```

```bash
flwr run . --run-config 'algorithm="fedavg"'
```

Sweep the fairness weight over the Appendix B.5 grid for five seeds. Rows are
appended to one CSV as runs finish, and re-running the same command resumes:

```bash
PYTHONPATH=. python scripts/sweep.py --partition iid --seeds 0,1,2,3,4 --jobs 3
```

Plot mean ± sd over seeds and print a summary table:

```bash
PYTHONPATH=. python scripts/plot_tradeoff.py results/sweep_kffl_adult_iid.csv results/sweep_kffl_adult_non-iid.csv
```

COMPAS takes one of three sensitive-attribute variants (`african-american`,
`race-sex`, `race`); the multi-group ones use Section 6.5's larger `D`:

```bash
PYTHONPATH=. python scripts/sweep.py --dataset compas --sensitive african-american --grid compas-extended --seeds 0,1,2,3,4 --jobs 2
```

```bash
PYTHONPATH=. python scripts/sweep.py --dataset compas --sensitive race-sex --grid compas-multi --extra "rfm-dim=16" --seeds 0,1,2,3,4 --jobs 2
```

```bash
PYTHONPATH=. python scripts/plot_tradeoff.py --gaps spd_multi,eo_multi results/sweep_kffl_compas_race-sex_iid.csv
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

## Reproduction

Every run uses logistic regression, 4 clients, 10 global rounds, 5 local epochs,
batch size 64, Adam at 0.01 and `D = 10`, with the `pyproject.toml` defaults.
Five seeds (0-4) vary the client partition, initial model, random features and
batch order. Tables report mean ± sd over seeds.

### ADULT

FedAvg baseline:

| | accuracy | SPD | EO |
|---|---|---|---|
| paper (Table 6, non-IID) | 84.30 ± 1.41 | 0.18 ± 0.06 | 0.22 ± 0.03 |
| ours, IID | 85.26 ± 0.04 | 0.171 ± 0.006 | 0.064 ± 0.008 |
| ours, non-IID | 85.19 ± 0.04 | 0.169 ± 0.008 | 0.057 ± 0.021 |

KFFL at selected points of the Appendix B.5 grid (every point: run
`scripts/plot_tradeoff.py` on `results/sweep_kffl_adult_*.csv`):

| lambda | IID acc | IID \|SPD\| | IID \|EO\| | non-IID acc | non-IID \|SPD\| | non-IID \|EO\| |
|---|---|---|---|---|---|---|
| 0 | 0.8537 ± 0.0005 | 0.172 ± 0.005 | 0.068 ± 0.006 | 0.8527 ± 0.0003 | 0.163 ± 0.005 | 0.052 ± 0.012 |
| 123.16 | 0.8528 ± 0.0008 | 0.156 ± 0.017 | 0.031 ± 0.025 | 0.8495 ± 0.0027 | 0.105 ± 0.025 | 0.085 ± 0.052 |
| 329.47 | 0.8499 ± 0.0029 | 0.138 ± 0.028 | 0.050 ± 0.028 | 0.8444 ± 0.0052 | 0.080 ± 0.023 | 0.151 ± 0.050 |
| 638.95 | 0.8472 ± 0.0028 | 0.140 ± 0.028 | 0.037 ± 0.042 | 0.8349 ± 0.0163 | 0.109 ± 0.064 | 0.180 ± 0.162 |
| 1000.00 | 0.8394 ± 0.0125 | 0.116 ± 0.057 | 0.081 ± 0.091 | 0.8115 ± 0.0427 | 0.169 ± 0.140 | 0.339 ± 0.245 |

- **Non-IID shows a genuine but short trade-off.** Up to `lambda ~ 330`, mean
  |SPD| falls from 0.163 to 0.080 for about 0.8 points of accuracy. Beyond that
  it reverses: from `lambda ~ 740` mean |SPD| returns to FedAvg levels while the
  seeds diverge. At `lambda = 1000` one seed reaches |SPD| 0.006, another 0.357,
  and one seed's accuracy drops to 0.737.
- **IID does much less on average.** Mean |SPD| only reaches 0.116 at
  `lambda = 1000`, ranging from 0.026 to 0.166 across seeds. A single-seed
  sweep (seed 0) had suggested a clean 0.168 -> 0.026 curve; seed 0 turns out to
  be the most favourable seed.
- **|EO| overshoots** once `lambda` is large, most strongly in the non-IID
  setting. KFFL targets statistical parity; with unequal base rates (men earn
  >50K at 2.8x the rate of women) equalising selection rates pushes the
  true-positive rates apart.
- **The paper's EO does not match.** The paper reports FedAvg EO 0.22; ours is
  about 0.06. Accuracy and SPD agree. This gap is unexplained.

Absolute `lambda` values are not comparable across implementations in any case:
the kernel bandwidth is never specified in the paper (item 7 below), and it sets
the scale of `psi`.

### COMPAS

The label is two-year recidivism, so always predicting "no" scores 54.5 % on the
test set. The binary sweep uses Appendix B.5's COMPAS grid extended up to
`lambda = 5000`, because nothing changes at `lambda <= 71.58`. The multi-group
sweeps use Section 6.5's `{0, 1000, 1100, ..., 2000}` with `D = 16` (race x sex)
and `D = 24` (six races).

FedAvg baseline:

| sensitive attribute | partition | accuracy | gap 1 | gap 2 |
|---|---|---|---|---|
| paper, African-American (Table 3, non-IID) | non-IID | 61.13 ± 1.25 | SPD 0.16 ± 0.02 | EO 0.23 ± 0.02 |
| African-American vs rest | IID | 67.71 ± 0.26 | SPD 0.254 ± 0.004 | EO 0.274 ± 0.006 |
| African-American vs rest | non-IID | 68.29 ± 0.46 | SPD 0.263 ± 0.011 | EO 0.284 ± 0.013 |
| race x sex (K = 4) | IID | 68.39 ± 0.22 | SPD_multi 0.369 ± 0.007 | EO_multi 0.400 ± 0.008 |
| race x sex (K = 4) | non-IID | 68.76 ± 0.31 | SPD_multi 0.359 ± 0.012 | EO_multi 0.391 ± 0.019 |
| six races (K = 6) | IID | 67.71 ± 0.26 | SPD_multi 0.278 ± 0.005 | EO_multi 0.640 ± 0.004 |
| six races (K = 6) | non-IID | 68.29 ± 0.46 | SPD_multi 0.291 ± 0.013 | EO_multi 0.631 ± 0.009 |

KFFL, African-American vs rest:

| lambda | IID acc | IID \|SPD\| | IID \|EO\| | non-IID acc | non-IID \|SPD\| | non-IID \|EO\| |
|---|---|---|---|---|---|---|
| 0 | 0.678 ± 0.003 | 0.257 ± 0.006 | 0.277 ± 0.008 | 0.683 ± 0.005 | 0.264 ± 0.009 | 0.288 ± 0.010 |
| 123.16 | 0.653 ± 0.028 | 0.206 ± 0.031 | 0.225 ± 0.039 | 0.657 ± 0.030 | 0.204 ± 0.049 | 0.233 ± 0.052 |
| 250 | 0.637 ± 0.038 | 0.167 ± 0.073 | 0.193 ± 0.078 | 0.651 ± 0.024 | 0.195 ± 0.048 | 0.213 ± 0.056 |
| 500 | 0.640 ± 0.060 | 0.178 ± 0.081 | 0.192 ± 0.068 | 0.629 ± 0.087 | 0.187 ± 0.069 | 0.186 ± 0.103 |
| 1000 | 0.639 ± 0.085 | 0.193 ± 0.090 | 0.204 ± 0.114 | 0.613 ± 0.131 | 0.229 ± 0.020 | 0.235 ± 0.037 |
| 5000 | 0.607 ± 0.142 | 0.265 ± 0.029 | 0.226 ± 0.066 | 0.606 ± 0.138 | 0.274 ± 0.025 | 0.239 ± 0.067 |

KFFL, race x sex (Section 6.5, Figure 5):

| lambda | IID acc | IID SPD_multi | IID EO_multi | non-IID acc | non-IID SPD_multi | non-IID EO_multi |
|---|---|---|---|---|---|---|
| 0 | 0.684 ± 0.002 | 0.369 ± 0.007 | 0.400 ± 0.008 | 0.687 ± 0.003 | 0.361 ± 0.011 | 0.388 ± 0.023 |
| 1000 | 0.594 ± 0.052 | 0.153 ± 0.071 | 0.160 ± 0.068 | 0.601 ± 0.051 | 0.172 ± 0.068 | 0.181 ± 0.065 |
| 1300 | 0.571 ± 0.078 | 0.163 ± 0.083 | 0.184 ± 0.075 | 0.544 ± 0.121 | 0.187 ± 0.095 | 0.209 ± 0.080 |
| 1600 | 0.588 ± 0.112 | 0.217 ± 0.077 | 0.221 ± 0.069 | 0.602 ± 0.119 | 0.218 ± 0.068 | 0.210 ± 0.066 |
| 2000 | 0.567 ± 0.160 | 0.269 ± 0.058 | 0.289 ± 0.048 | 0.589 ± 0.146 | 0.260 ± 0.073 | 0.291 ± 0.050 |

KFFL, six races (Section 6.5, Figure 6), with the supplementary gaps over
groups of at least 20 test samples:

| lambda | partition | accuracy | SPD_multi | EO_multi | SPD_multi_min20 | EO_multi_min20 |
|---|---|---|---|---|---|---|
| 0 | IID | 0.678 ± 0.003 | 0.280 ± 0.006 | 0.640 ± 0.000 | 0.280 ± 0.006 | 0.325 ± 0.008 |
| 1000 | IID | 0.658 ± 0.021 | 0.415 ± 0.196 | 0.769 ± 0.214 | 0.233 ± 0.017 | 0.260 ± 0.034 |
| 1300 | IID | 0.614 ± 0.088 | 0.485 ± 0.198 | 0.571 ± 0.044 | 0.181 ± 0.064 | 0.172 ± 0.078 |
| 2000 | IID | 0.630 ± 0.091 | 0.560 ± 0.154 | 0.703 ± 0.174 | 0.221 ± 0.068 | 0.238 ± 0.065 |
| 0 | non-IID | 0.683 ± 0.005 | 0.293 ± 0.012 | 0.633 ± 0.010 | 0.290 ± 0.015 | 0.331 ± 0.011 |
| 1000 | non-IID | 0.621 ± 0.093 | 0.506 ± 0.179 | 0.622 ± 0.083 | 0.203 ± 0.066 | 0.189 ± 0.078 |
| 1300 | non-IID | 0.647 ± 0.031 | 0.462 ± 0.179 | 0.674 ± 0.187 | 0.221 ± 0.026 | 0.218 ± 0.034 |
| 2000 | non-IID | 0.642 ± 0.063 | 0.532 ± 0.129 | 0.706 ± 0.173 | 0.218 ± 0.072 | 0.239 ± 0.066 |

- **At the paper's 10-round schedule, no setting gives a usable trade-off.**
  With the binary attribute, seeds
  that train normally stay at |SPD| 0.20-0.23 with 65-67 % accuracy. The
  low-gap points come from collapsed models, mostly seeds 2 and 3, at 38-59 %
  accuracy, at or below the majority-class rate. The paper's Table 3 KFFL point
  (60.30 %, SPD 0.06, EO 0.07) is not reproduced.
- **FedAvg does not match the paper either.** We get 67.7-68.3 % and SPD
  0.25-0.26, against Table 3's 61.13 % and SPD 0.16. The paper disagrees with
  itself here: its Figure 4 puts FedAvg near 0.72 accuracy and SPD 0.29.
- **Race x sex halves both gaps, but only by giving up accuracy.** The gaps
  fall from 0.37 to 0.15-0.17 at `lambda = 1000` while accuracy drops from 0.68
  to 0.59-0.60. Past `lambda ~ 1300` the gaps climb back and the accuracy spread
  widens. Section 6.5's headline point (both gaps below 0.05 at about 0.65
  accuracy, `lambda = 1600`, IID) is not reproduced: we get 0.217 and 0.221 at
  0.588.
- **The paper's six-race gaps get worse as `lambda` grows** (0.28 -> 0.41-0.62),
  because the 2 Native American and 7 Asian test samples dominate the max-min gap
  once predictions become more extreme. Over groups with at least 20 test
  samples, SPD_multi falls from 0.28-0.29 to about 0.18-0.20 at best, again with
  lost accuracy and high variance.
- All of this matches the step-size instability described next.

Figures: `results/adult_tradeoff_5seeds.png`,
`results/compas_african-american_5seeds.png`, `results/compas_race-sex_5seeds.png`,
`results/compas_race_5seeds.png` and `results/compas_race_min20_5seeds.png`.

### Why the trade-off reverses: an oversized fairness step

KFFL applies the fairness gradient as one full step per round,
`w_{t+1/2} = w_t - lambda grad psi`. Logging the size of that step (runs in
`results/diagnostics/`) shows:

- **The first fairness step is 4-40x larger than any later one.** On binary
  COMPAS it has norm 9 (`lambda = 500`), 38 (`lambda = 2000`) and 54
  (`kernel-gamma-f = 0.1`, `lambda = 20000`), against at most about 2
  afterwards. It is taken on the untrained initial model and pushes the train
  loss to 1.8-11.
- **A large step overshoots parity.** SPD flips from +0.25 to -0.25 within a
  single round.
- **The local epochs then pull the model back.** The final metrics depend on
  where round 10 happens to fall in that oscillation, which differs from seed to
  seed.

Remedies tried on binary COMPAS (single seed):

| change | outcome |
|---|---|
| `model-init = "zeros"`, so `G(w_0) = 0` | the oversized step simply moves to round 2 |
| `fairness-output = "prob"` | steps stay bounded, but SPD flips sign from round to round |
| `kernel-gamma-f = 0.1` | same first-round kick; SPD stays at 0.19 or above |
| `prox-mu = 1` | the kick is locked in: loss stays high, SPD unchanged |
| step 0.1, `prox-mu = 5`, 50 rounds, `lambda >= 1000` | loss rises over 50 rounds; SPD improves 0.27 -> 0.19 at best. That `lambda` was 30-300x too large for this regime: see the audit below |

ADULT is not immune. With `model-init = "zeros"` at `lambda = 1000` (IID), the
fairness steps reach 16 and 11, and SPD swings between -0.06 and +0.15 from
round to round.

Theorem 1 guarantees convergence only for a step size below
`min{1/20, 1/(2L_-), 1/(120 L(B+1)), 1/(5 L B^2)}`, and only as the number of
rounds grows. The sweeps above break both conditions: step size 1 with `lambda`
in the hundreds or thousands, and 10 rounds. The audit below shows that the same
code converges with a small consistent `alpha`, a few hundred rounds, and a
`lambda` 30-300x smaller. No clipping is needed.

### Is it the code? An audit on COMPAS

Checks on the binary COMPAS task (seed 0; the Flower runs are saved as
`results/diagnostics/audit_*.json`):

- **Gradient.** The fairness gradient assembled by our client and server
  functions equals centralised autograd of the random-feature KHSIC to `1e-15`,
  at the initial model and at a trained one.
- **Data and metrics.** scikit-learn's logistic regression on our features gives
  67.7 % accuracy and SPD 0.258, as our FedAvg does. SPD and EO recomputed by
  hand match `metrics.py`.
- **Convergence to Eq. (10).** The table compares minimising Eq. (10) centrally
  with the exact `n x n` kernel (no federation, no random features) against our
  Flower KFFL running Algorithm 1 with one consistent `alpha = 0.1`
  (`fairness-step-size=0.1`, `prox-mu=5`, the paper's local solver, `D = 10`)
  for 300 rounds:

  | lambda | centralised exact KHSIC: acc / SPD | Flower KFFL, 300 rounds: acc / SPD |
  |---|---|---|
  | 0 | 0.679 / +0.258 | 0.681 / +0.260 |
  | 3 | 0.679 / +0.251 | 0.680 / +0.229 |
  | 10 | 0.615 / +0.114 | 0.600 / +0.121 |
  | 30 | 0.533 / -0.013 | 0.543 / -0.010 |

  The federated runs are stable: over the last 50 rounds the sd of SPD is at
  most 0.016.
- **Rounds needed.** With `alpha = 1` or `0.3` the same runs are far from those
  values after 10 rounds and not settled after 100. For `lambda = 30` and
  `alpha = 1`, SPD is 0.219 at round 10, 0.201 at round 30 and 0.047 at round
  100.

So the implementation is correct and Algorithm 1 works. What does not reproduce
is the paper's schedule: Appendix B.1 says 10 global rounds were enough for
convergence on every dataset, and in this implementation KFFL on COMPAS needs a
few hundred. The 10-round sweeps above therefore report transients, not
solutions of Eq. (10). Their `lambda` scale (hundreds to thousands) is an
artefact of that: once converged, the whole trade-off lies in `lambda` 3-30,
inside the paper's COMPAS grid of 0.01-123.

On the converged curve, SPD 0.06 costs about 9 points of accuracy (0.68 to about
0.59). The paper's Table 3 KFFL point (60.3 %, SPD 0.06) lies close to that
curve. What differs is the paper's FedAvg baseline (61.1 %, SPD 0.16), which
makes fairness look almost free; the paper's own Figure 4 puts FedAvg near 0.72
accuracy with SPD 0.29.

## Extension: label-conditional KHSIC

This is not in the paper, whose Limitations section names "conditional variants
of the KHSIC to ensure equalized odds" as future work. It is off by default.

```bash
flwr run . --run-config 'khsic-conditioning="label" khsic-stratum-weights="prevalence" fairness-lambda=10.0'
```

**Objective.** The KHSIC of Eq. (9) is computed separately inside each true-label
stratum `y`, on the `n_y` samples with that label, and the results are summed:

```
psi_cond(w) = sum_y w_y psi_y(w),     psi_y(w) = E ||G_y(w)||_F^2 / (n_y - 1)^2
```

`khsic-stratum-weights="prevalence"` sets `w_y = n_y / n`, so `psi_cond` is the
average conditional dependence over the empirical distribution of `Y`.
`"uniform"` sets `w_y = 1/C`. The penalty is zero when `f(x; w)` is independent
of `s` given `y`, which is equalised odds; it does not ask for statistical
parity. A predictor that depends only on the label has zero penalty.

**Algorithm.** Lemmas 1 and 2 apply inside each stratum unchanged, because each
`psi_y` is Eq. (9) on the sub-dataset `{j : y_j = y}`:

| Step | Per stratum `y` |
|---|---|
| FAIR1, client | `M^{i,y} = Z_{s,i,y}^T Z_{f,i,y}`, `mu_{s,i,y}`, `mu_{f,i,y}`, `n_{i,y}` |
| FAIR1, server | `G_y = sum_i M^{i,y} - n_y mu_{s,y} mu_{f,y}^T` with the stratum means |
| FAIR2, client | `g^i = sum_y c_y J_{Omega_{i,y}}^T G_y`, `Omega_{i,y} = M^{i,y} - n_{i,y} mu_{s,y} mu_{f,i,y}^T` |
| FAIR2, server | `w_{t+1/2} = w_t - alpha lambda sum_i g^i` |

The scales `c_y = w_y * 2 / (n_y - 1)^2` are applied on the client inside one
backward pass, so the client still returns a single gradient. The fairness
payload grows from `D^2` to `C D^2` (`C = 2` for binary labels); the `2|w|` part
is unchanged. The unconditional method is the same code with one stratum.

**Relation to the paper's penalty.** The unconditional interaction matrix splits
into a within-label and a between-label part,
`G = sum_y G_y + sum_y n_y (mu_{s,y} - mu_s)(mu_{f,y} - mu_f)^T`. The conditional
penalty keeps only the first: it ignores dependence that runs through the label.

**Requirements and caveats.** Labels must be discrete; a regression task raises
an error. A stratum with fewer than two samples globally is skipped. Clients now
reveal their per-label counts and per-label feature means to the server.

**Verification.** `scripts/verify_lemmas.py` checks the per-stratum
decompositions against centralised references (to `1e-14`), and
`scripts/centralized_khsic.py` minimises the same objective without federation.

### Results: conditional vs unconditional KFFL

All runs use Algorithm 1 with one consistent `alpha = 0.1`
(`fairness-step-size=0.1`, `prox-mu=5`), 300 rounds, the paper's local solver,
`D = 10`, IID clients and 3 seeds. Tables give mean ± sd; the equalised-odds gap
is the larger of the TPR and FPR gaps.

```bash
PYTHONPATH=. python scripts/sweep.py --dataset adult --seeds 0,1,2 --jobs 4 --lambdas 3,10,30,100,300 --extra 'num-server-rounds=300 fairness-step-size=0.1 prox-mu=5.0 khsic-conditioning="label"' --out results/cond_adult_label-prevalence_iid.csv
```

ADULT (sex):

| method | lambda | accuracy | \|SPD\| | EO (signed) | eq. odds gap |
|---|---|---|---|---|---|
| no penalty | 0 | 0.837 ± 0.001 | 0.187 ± 0.010 | +0.227 ± 0.019 | 0.227 ± 0.019 |
| unconditional | 10 | 0.829 ± 0.003 | 0.089 ± 0.008 | -0.049 ± 0.013 | 0.049 ± 0.013 |
| unconditional | 30 | 0.824 ± 0.000 | 0.043 ± 0.004 | -0.158 ± 0.009 | 0.158 ± 0.009 |
| unconditional | 300 | 0.808 ± 0.008 | 0.005 ± 0.005 | -0.222 ± 0.072 | 0.222 ± 0.072 |
| conditional, prevalence | 30 | 0.830 ± 0.002 | 0.099 ± 0.004 | -0.003 ± 0.007 | 0.026 ± 0.002 |
| conditional, prevalence | 100 | 0.827 ± 0.001 | 0.084 ± 0.003 | -0.008 ± 0.008 | 0.015 ± 0.002 |
| conditional, uniform | 100 | 0.827 ± 0.001 | 0.087 ± 0.002 | +0.001 ± 0.003 | 0.015 ± 0.001 |
| conditional, uniform | 300 | 0.820 ± 0.002 | 0.079 ± 0.007 | -0.004 ± 0.015 | 0.013 ± 0.007 |

COMPAS (African-American vs rest):

| method | lambda | accuracy | \|SPD\| | eq. odds gap | positive rate |
|---|---|---|---|---|---|
| no penalty | 0 | 0.686 ± 0.005 | 0.257 ± 0.007 | 0.288 ± 0.006 | 0.37 |
| unconditional | 10 | 0.613 ± 0.013 | 0.116 ± 0.005 | 0.150 ± 0.024 | 0.25 |
| unconditional | 20 | 0.573 ± 0.012 | 0.038 ± 0.031 | 0.080 ± 0.015 | 0.24 |
| unconditional | 30 | 0.544 ± 0.017 | 0.024 ± 0.013 | 0.082 ± 0.020 | 0.24 |
| conditional, prevalence | 20 | 0.612 ± 0.005 | 0.092 ± 0.013 | 0.113 ± 0.003 | 0.20 |
| conditional, prevalence | 50 | 0.581 ± 0.008 | 0.048 ± 0.009 | 0.063 ± 0.012 | 0.10 |
| conditional, prevalence | 100 | 0.564 ± 0.010 | 0.019 ± 0.017 | 0.028 ± 0.028 | 0.08 |
| conditional, uniform | 100 | 0.563 ± 0.011 | 0.019 ± 0.015 | 0.030 ± 0.023 | 0.08 |

- **On ADULT the two penalties do different jobs.** The unconditional penalty
  drives SPD to 0.005 but pushes EO from +0.23 through zero to -0.22, so the
  equalised-odds gap ends where it started. The conditional penalty cuts that
  gap from 0.227 to 0.015 for one point of accuracy, and leaves SPD at about
  0.08, the part explained by the different base rates.
- **On COMPAS the conditional penalty is modestly better on both gaps** at equal
  accuracy (at about 61 %: equalised-odds gap 0.113 against 0.150). Both
  penalties are expensive here, and the conditional one gets fair largely by
  predicting recidivism for fewer people (positive rate 0.37 down to 0.08), so
  its low-gap end is close to a constant predictor.
- **Prevalence and uniform weights behave almost identically** on both datasets.
  Uniform is slightly steadier at the largest `lambda` on ADULT.
- **`lambda = 1000` is too large for `alpha = 0.1` on ADULT**: the train loss
  rises to 0.45-0.57 and accuracy falls to 0.79-0.80. A smaller `alpha` with more
  rounds would be needed there.
- **The ADULT schedule is not fully converged.** With `prox-mu = 5`, 300 rounds
  reach 83.7 % at `lambda = 0`, against 85.3 % for plain FedAvg and for the
  centralised optimum. The comparison between methods is like for like, but the
  absolute numbers sit about 1.5 points below what a longer run would give.
- At 300 rounds the seeds agree closely (sd about 0.01), unlike the 10-round
  sweeps.

Figures: `results/cond_adult_comparison.png` and
`results/cond_compas_comparison.png`. Non-IID clients and the multi-group
settings have not been run with this option.

## Errors and ambiguities found in the paper

Each is exposed as a config switch so both readings can be measured. Defaults are
listed first.

### 1. Eq. (15) gives a biased gradient as printed — `omega-variant`

Algorithm 3 line 11 ships the *product* `mu_s mu_f^T` as one constant `D x D`
block, and Eq. (15) reads `Omega_i(w) = M^i(w) - n_i mu_s mu_f^T`. Taken
literally the subtrahend is constant, so `J_{Omega_i} = J_{M^i}` and the sum
over clients drops the `-n mu_s d(mu_f)/dw` term. That contradicts Corollary 1
(unbiasedness) and Appendix A.6, which differentiate `G = Z_s^T H Z_f`.

The exact decomposition needs the *local* mean, differentiated:
`Omega_i(w) = M^i(w) - n_i mu_s mu_{f,i}(w)^T`. Since `Z_s` does not depend on
`w`, `mu_s` really is constant, `sum_i Omega_i = G` holds identically in `w`, and
`sum_i J_{Omega_i}^T G = J_G^T G` exactly.

`scripts/verify_lemmas.py` measures both against a centralised autograd
reference: the exact form matches to `1e-13`, the literal form is off by
**2590 %**. We send `mu_s` and `mu_f` as two `D`-vectors instead of their outer
product (2D floats vs `D^2`, so Table 2 is unaffected).

*Default `"exact"`; `"paper-literal"` reproduces Eq. (15) as written.*

### 2. The `2/(n-1)^2` factor and `lambda` vanish — `psi-normalization`

Eq. (9) defines `psi` with `1/(n-1)^2`, Appendix A.6 carries the factor 2 from
differentiating a squared Frobenius norm, and Section 3 "absorbs `lambda` into
`psi`" — but Eqs. (13)–(16) and Algorithms 2–3 show none of the three. Keeping
the normalization is what makes the published `lambda` grids the right order of
magnitude (`||G||_F^2` is `O(10^4..10^8)` at `D = 10`; without the `1/(n-1)^2`
the update would be `O(10^9)`).

*Default `"n-minus-1-sq"`; `"none"` reproduces the literal Eqs. (13)–(16).*

### 3. Algorithm 3 line 14 drops `alpha` — `fairness-step-size`

Algorithm 1 line 3 has `w_{t+1/2} = w_t - alpha g_t`; Algorithm 3 line 14 has
`w_{t+1/2} <- w_t - sum_i g^i`. This is only a reparameterization of `lambda`
(`alpha * lambda` is what acts), but it decides whether Appendix B.5's grids are
meaningful. **Only the Algorithm 3 form does**: with step size 1, ADULT's grid of
`lambda` in `[20, 1000]` spans the range where the regulariser acts, while with
`alpha = 0.01` the same range needs `lambda` up to `10^5`. (Over five seeds the
resulting curve is much less clean than a single seed suggested; see
Reproduction.)

*Default `1.0` (Algorithm 3); `-1` uses `alpha` (Algorithm 1).*

### 4. Algorithm 1's proximal weight cannot reproduce the paper's own accuracies — `prox-mu`

Algorithm 1 line 5 pairs the global step size with a `1/(2 alpha)` proximal
weight. Appendix B.1 gives `alpha = 0.01` — but describes it as the *Adam
learning rate*, and never states the Algorithm 1 `alpha` at all. Reading them as
the same quantity gives a proximal weight of **50**, which pins the model to
`w_{t+1/2}` so hard that 10 global rounds cannot train it. Measured on ADULT
(IID, seed 0, `lambda = 0`, so the fairness term is inactive):

| `prox-mu` | accuracy | SPD |
|---|---|---|
| 0.001 | 0.8523 | +0.168 |
| **0.01** (default) | **0.8536** | **+0.168** |
| 0.1 | 0.8409 | +0.195 |
| 1.0 | 0.8134 | +0.107 |
| 10 | 0.7619 | −0.001 |
| 50 = `1/(2*0.01)` | 0.6756 | −0.054 |

The paper reports KFFL at 83.14 % on ADULT, close to its FedAvg baseline (84.30 %),
which is only reachable with a weak proximal term. Note also that `prox-mu = 10`
already drives SPD to ~0 **with `lambda = 0`**: an over-strong proximal term keeps
the model near its initialisation, which is trivially fair. Any fairness result
measured at `prox-mu = 50` would be confounded by that.

*Default `0.01`; `-1` selects the literal `1/(2 alpha)`, `0` drops the term.*

### 5. The seed `zeta` is generated per client — structural

Algorithm 3 line 5 places "Generation of random seed `zeta`" *inside* the
`for all i in parallel` loop. If each client drew its own seed, the `Z_{s,i}` and
`Z_{f,i}` blocks would live in different random feature spaces and `G` would be
meaningless. The server generates one `zeta` per iteration and broadcasts it.

### 6. Appendix G contradicts itself on resampling — `rfm-resample`

"the random features are drawn **once at the beginning** of global training round
and are fixed throughout the optimization process" is self-contradictory, and
Algorithm 3 regenerates `zeta` every iteration. Corollary 1's unbiasedness and
Theorem 1's `sigma^2` noise model both need fresh draws.

*Default `"per-round"` (Algorithm 3); `"once"` follows the Appendix G sentence.*

### 7. The kernel bandwidth is never specified — `kernel-gamma-s`, `kernel-gamma-f`

Neither `sigma` nor `gamma` for `kappa_F` or `kappa_G` appears anywhere in the
paper. We default both to `1.0`. This is an unavoidable assumption and it shifts
the absolute scale of `lambda`.

### 8. Eq. (3) is malformed

`SPD_multi = max_s |r_s - max_s' |r_s - r_s'||` is not well formed. Definition 2's
`EO_multi` is the clean max-min gap and the text says both "reduce to Eqs. (1)–(2)
when K = 2", so `max_{s,s'} |r_s - r_s'|` is intended; that is what we compute.

### 9. Smaller transcription issues

- **Eq. (8)** has `H K_s H^2 K_y H` where Eq. (9) and Appendix A.3 have
  `Tr(K_s H K_f H)`. Harmless: `H` is idempotent, so they agree.
- **Eq. (21)** writes the local subproblem as
  `argmin_w [f_i(w_t) + (1/2a)||w - w_t||^2]` — the objective should be at `w`,
  not `w_t`, and the proximal centre should be `w_{t+1/2}`, not `w_t`. Algorithm
  1 line 5 and Definition 3 are both correct; we implement those.
- **Eq. (18)** puts a transpose on the whole expression though `mu_f` is a vector.
- **Algorithm 4 line 6** cites Eqs. 17/18 for `mu_{f,i}`/`mu_{s,i}` where it means
  20/19.
- **Appendix B.5**, ADULT: the list is `linspace(20, 1000, 20)` with `638.95`
  dropped. `scripts/sweep.py` uses the full grid.
- **Section 6.1** says ADULT's `lambda` ranges "from 0 to 0.01", contradicting
  both its own Figure 3 caption and Appendix B.5. We follow B.5.
- **Section 6.5** claims the ORF map "requires `D >= 2k`". No such constraint
  exists in Yu et al. (2016); the real one is that the map is built from
  `ceil(D/d)` orthogonal `d x d` blocks, which works for any `D`.

## Assumptions not stated in the paper

- **`phi_f` is applied to the logit**, not the sigmoid (Appendix G: "the model
  output (e.g., logits)"). Switch with `fairness-output`.
- **The sensitive attribute is one-hot encoded** before `phi_s` (implied by the
  "one-hot columns" discussion in Section 6.5) and excluded from the model's
  features (Appendix F states this for regression; applied to classification too).
  Switch with `sensitive-encoding` / `exclude-sensitive-from-features`.
- **Preprocessing is fitted on the full training split** and shared by all
  clients, so every `X` lives in one space. The paper does not describe its
  preprocessing.
- **`?` is kept as a category** rather than dropped, keeping `n = 32,561` as the
  "n = 32K" quoted in Section 4 (dropping would give 30,162).
- **Evaluation is central**, on the held-out test split, as the paper reports it.
- **The fairness terms use all local data** each round, not a minibatch (implied
  by `n_i` and `mu_{s,i}` in Eqs. 19–20).
- **Non-IID at full size is infeasible.** Appendix B.3 asks for 90/10 per client,
  but ADULT's sex attribute is ~2:1 globally, so the demand exceeds the pools.
  `non-iid-mode="exact-ratio"` (default) gives every client exactly 90/10 by
  shrinking the uniform client size, using 21,536 of 32,561 samples;
  `"use-all"` keeps every sample at approximate ratios. The paper is silent.
- **COMPAS** (`kffl/dataset/compas.py`). The standard ProPublica filter gives
  the paper's 6,172 rows, but the paper lists neither its "52 predictive
  features" nor a train/test split. We use AIF360's feature set minus the
  389-level `c_charge_desc` and minus `race` (12 features; 10 for `race-sex`,
  which also drops `sex`), and one 80/20 split stratified on race x label that
  every variant shares. The label is `two_year_recid`, so `yhat = 1` predicts
  recidivism, and a positive SPD means African-Americans are predicted to
  reoffend more often.
- **COMPAS sensitive-attribute variants.** `african-american` is
  African-American vs everyone else (K = 2). `race-sex` is Section 6.5's
  intersectional groups (K = 4). `race` is all six recorded races (K = 6).
- **Non-IID with K > 2.** Appendix B.3 defines the split only for a binary
  group, so the multi-group COMPAS variants are partitioned on the binary
  African-American indicator.
- **Tiny groups in the six-race setting.** The test set holds 7 Asian and 2
  Native American people (1 positive), so a single prediction decides
  `EO_multi`, which is routinely exactly 1.0. Evaluation also logs
  `spd_multi_min20` / `eo_multi_min20`: the same gaps over groups with at least
  20 test samples (positives for EO). They are a diagnostic, not a metric from
  the paper.
- **Degenerate classifiers.** Evaluation logs the positive-prediction rate
  overall and per group, plus per-group TPR. A model that predicts a single
  class scores SPD = EO = 0, which otherwise looks the same as a fair one.
- **Initial model.** The paper never gives `w_0`. Every reported run uses
  `model-init="default"` (PyTorch's random init). `"zeros"` makes
  `f(X; w_0)` constant for linear models, so `G(w_0) = 0`. We tried it as a
  remedy for the oversized first fairness step on COMPAS and it did not help:
  the large step simply moves to round 2.
