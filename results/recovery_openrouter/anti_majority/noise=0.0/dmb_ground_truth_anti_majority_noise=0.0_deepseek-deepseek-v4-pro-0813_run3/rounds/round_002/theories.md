# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** People invert the valence of binary expert ratings: a 1 is treated as a costly defect and a 0 as a clean, favorable signal. Across all features they compute an inverse-tally advantage for A, weakly weighting the inverse cues by the experts' stated validities. When the raw inverse tally is nonzero, choice is a noisy softmax over the two weakly validity-weighted inverse tallies. When the raw inverse tally is exactly tied, people do not guess uniformly; instead they break the tie lexicographically by consulting features in descending validity order and choosing the option that has a clean 0 on the first discriminating feature. Response noise is captured by a softmax plus an independent uniform lapse. This differs from Take The Best because it aggregates all cues rather than stopping at the first discriminator, and from plain inverse tallying because ties are resolved by validity-ordered clean-cue use instead of guessing.

**Rationale:** The proposed theory keeps the strong inverse-tally effect that drives the negative top-cue dependence in Experiment 1 and the inverse-tally contrasts in Experiments 2 and 4, while fixing two mechanistic failures of the earlier theories. Take The Best fails because it stops at the first discriminating cue and therefore cannot produce the observed inverse, all-cue aggregation effects. Plain inverse tallying fails in Experiment 3 because those single-discriminator trials have an exactly tied inverse tally, causing uniform guessing; here the validity-ordered clean-cue tie-break systematically chooses the option with a 0 on the first discriminating feature, which reproduces the observed negative TTB-agreement score. Adding weak validity weighting to the otherwise equal-weight inverse tally allows the magnitude of the inverse-tally effect to differ between Experiments 2 and 4 as observed, because a raw inverse-tally advantage of +/-2 is composed of different cue locations with different validity weights in the two designs. Narrow parameter ranges around the selected softmax temperatures, lapse rate, and weak validity-weighting strength keep the pooled predictions near the observed metric values while retaining a psychologically plausible amount of response noise.

**Parameters:**
  - `beta_tally`: `[0.43, 0.47]`
  - `beta_tiebreak`: `[1.00, 1.08]`
  - `epsilon`: `[0.09, 0.11]`
  - `validity_weight`: `[1.90, 2.10]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    beta_tally = float(parameters['beta_tally'])
    beta_tiebreak = float(parameters['beta_tiebreak'])
    epsilon = float(parameters['epsilon'])
    validity_weight = float(parameters['validity_weight'])

    clean_a = (b > a)
    clean_b = (a > b)
    raw_s = float(np.sum(clean_a) - np.sum(clean_b))

    if raw_s != 0:
        if validity_weight > 0.0 and n_features > 1:
            weights = 1.0 + validity_weight * (val - np.mean(val))
        else:
            weights = np.ones(n_features)
        s = float(np.sum(weights * clean_a) - np.sum(weights * clean_b))
        scores = np.array([s, -s])
        beta = beta_tally
    else:
        cue_order = np.argsort(-val, kind='stable')
        winner = None
        for j in cue_order:
            if a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * np.ones(n_opts) / n_opts
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_3` — KILLED ✗

**Description:** People invert the meaning of binary expert ratings: a rating of 1 is treated as a defect or costly negative signal, and a rating of 0 is treated as a clean or favorable signal. Across all features, they tally how often each option is defect-free relative to the other option, and choose the option with the larger defect-free tally. Ties lead to guessing. Choice probability is a noisy softmax over the two defect tallies with inverse temperature beta, plus an independent lapse process that produces uniform guessing with probability epsilon. The heuristic uses all features equally, ignores validities and past trials, and differs from Take The Best by aggregating rather than stopping at the first discriminating cue, and from standard Tallying by reversing the valence of the binary ratings.

**Rationale:** Standard positive Tallying predicted positive metric values because it chooses the option favored by more 1-valued cues, whereas the human data show negative sensitivity to that cue direction. Reversing cue polarity makes the model treat 1s as defects and 0s as favorable evidence, flipping the predicted metric sign to match the observed negative values. Using all features avoids the flat near-zero metric produced by Take The Best in these experiments, because lower-cue disagreement can still influence choice. The noise parameters are deliberately non-deterministic: the arbiter noted that deterministic defect-counting would produce roughly -0.83 and -1.00 on Experiments 1 and 2, while the observed values are -0.51 and -0.64. The beta range of 1 to 5 and epsilon range of 0.30 to 0.40 add sufficient softmax uncertainty and lapse guessing to bring the model close to those observed values while preserving the inverse-tallying mechanism.

**Parameters:**
  - `beta`: `[1.0, 5.0]`
  - `epsilon`: `[0.30, 0.40]`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Inverse-tallying expects a (2, n_features) stimulus; got shape {stim.shape}.')

    a, b = stim[0], stim[1]

    # Defect-free wins: A has a 0 where B has a 1.
    a_wins_defects = float(np.sum(b > a))
    # Defect-free wins: B has a 0 where A has a 1.
    b_wins_defects = float(np.sum(a > b))

    scores = np.array([a_wins_defects, b_wins_defects])

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Numerically stable softmax over inverse-cue tallies.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** People invert the valence of binary expert ratings: a 1 is treated as a defect and a 0 as a clean, favorable signal. Each feature j probabilistically enters an inverse clean-cue tally with inclusion probability p_include_j = logistic(alpha * validity_j + gamma); excluded cues receive exactly zero weight. The included clean-cue tally drives a noisy softmax choice, and exact zero tallies are resolved by consulting the highest-validity included discriminating cue and choosing the option with the clean 0 there. If no included cue discriminates, people guess. Subject-level alpha, gamma, beta_tally, beta_tiebreak, and epsilon create individual differences, with heavy-tailed/logit-transformed subject-level distributions so that some subjects are nearly deterministic while others remain noisy, preserving the pooled inverse-valence pattern and realistically large between-subject variance.

**Rationale:** This is a minimal parameter-prior edit on top of the accepted validity-pruned inverse-tally architecture. The architecture is unchanged, but the direct uniform subject-level parameters are replaced by heavy-tailed transformed draws: log-uniform for beta_tally and beta_tiebreak, a symmetric logit-centered transform for gamma within the requested [-4.5, -2.5] band, and a logit transform for epsilon. The effective beta_tally range is now about 0.74-1.65 and beta_tiebreak about 1.49-3.00, making small inverse tallies and clean-zero tie-breaks more decisive, which repairs the under-powered Exp2, Exp3, and Exp4 scores. Tightening gamma to [-4.5, -2.5] with an edge-concentrated shape restores a small positive Exp6 validity-gradient effect instead of the previous overcorrection to zero, while alpha [5,9] removes the very high-alpha tail that flattened validity sensitivity. The interval transforms also raise targeted between-subject variance, especially in Exp5/Exp6, without the uniform widening that previously inflated Exp1-Exp3 variance.

**Parameters:**
  - `alpha`: `[5.0, 9.0]`
  - `gamma_logit`: `[-2.50, 2.50]`
  - `beta_tally_log`: `[-0.30, 0.50]`
  - `beta_tiebreak_log`: `[0.40, 1.10]`
  - `epsilon_logit`: `[-3.00, -0.40]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
from scipy.special import expit

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Heavy-tailed subject-level transforms of the sampled raw parameters.
    beta_tally = float(np.exp(float(parameters['beta_tally_log'])))
    beta_tiebreak = float(np.exp(float(parameters['beta_tiebreak_log'])))
    epsilon = float(expit(float(parameters['epsilon_logit'])))
    alpha = float(parameters['alpha'])
    gamma = -4.5 + 2.0 * float(expit(float(parameters['gamma_logit'])))

    p_include = expit(alpha * val + gamma)
    include = np.random.random(n_features) < p_include

    clean_a = ((b > a) & include).astype(float)
    clean_b = ((a > b) & include).astype(float)
    s = float(np.sum(clean_a) - np.sum(clean_b))

    if s != 0.0:
        p_a = expit(2.0 * beta_tally * s)
        p_core = np.array([p_a, 1.0 - p_a])
    else:
        winner = None
        cue_order = np.argsort(-val, kind='stable')
        for j in cue_order:
            if include[j] and a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        p_a = expit(beta_tiebreak) if winner == 0 else expit(-beta_tiebreak)
        p_core = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```
