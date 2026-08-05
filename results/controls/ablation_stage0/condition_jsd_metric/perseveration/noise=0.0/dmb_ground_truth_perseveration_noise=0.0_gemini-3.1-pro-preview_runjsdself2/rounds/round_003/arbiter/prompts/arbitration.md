# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Soft Strategy Selection: Decision makers probabilistically mix between a non-compensatory one-reason heuristic (Take The Best) and a compensatory approach (Weighted Additive rule) across trials. The probability of deploying WADD versus TTB is modeled as a smooth logistic function of the cue dispersion (the absolute difference in WADD scores). This soft mixture allows for both the compensatory integration of WADD at higher dispersions and the non-compensatory dominance of TTB at lower dispersions, without forcing a stark discontinuity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)

    beta_wadd = float(parameters["beta_wadd"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    dispersion_threshold = float(parameters["dispersion_threshold"])
    steepness = float(parameters["steepness"])

    # --- WADD Component ---
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)

    # --- TTB Component ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)

    # --- Soft Mixture based on Cue Dispersion ---
    dispersion = abs(score_a_wadd - score_b_wadd)
    x = -steepness * (dispersion - dispersion_threshold)
    
    # Numerically stable logistic function
    if x > 100:
        trial_wadd_prob = 0.0
    elif x < -100:
        trial_wadd_prob = 1.0
    else:
        trial_wadd_prob = 1.0 / (1.0 + np.exp(x))
    
    p_core = trial_wadd_prob * p_wadd + (1.0 - trial_wadd_prob) * p_ttb
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
People evaluate options by computing a weighted sum of their feature values, where the weights correspond to the subjective or objective validities of the cues (Weighted Additive rule, WADD). Rather than relying on a single discriminating cue (like Take The Best) or ignoring validities entirely (like Tallying), this compensatory strategy allows multiple lower-validity cues to jointly outweigh a single high-validity cue. Choices are made probabilistically via a softmax over the options' weighted sums, along with a lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Soft Strategy Selection theory from pure Weighted Additive (WADD) theory, we systematically vary the 'cue dispersion' (absolute difference in WADD scores) across trials where Take The Best (TTB) and WADD make opposing predictions. In pure WADD, when the dispersion is very small, the choice probability approaches 50/50 because the weighted sums are nearly identical. In contrast, the Soft Strategy Selection theory predicts a non-linear shift: at low dispersions, the decision-maker will revert to the non-compensatory TTB heuristic, causing the choice probability to shoot up toward 100% for the TTB-favored option despite the small WADD difference. By providing a gradient of dispersions (from 0.15 to 1.85) where TTB favors Option A but WADD favors Option B, we can clearly observe whether choice probabilities smoothly approach 50/50 (pure WADD) or exhibit a sharp resurgence of TTB at low dispersions (Advocated theory).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.05563093622795115, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.18518518518518517, '((1, 0, 1, 0, 0), (0, 1, 0, 0, 0))|0': 0.057676685621445976, '((1, 0, 1, 0, 0), (0, 1, 0, 0, 0))|1': 0.08403361344537816, '((1, 0, 0, 0, 0), (0, 0, 1, 0, 1))|0': 0.06257668711656442, '((1, 0, 0, 0, 0), (0, 0, 1, 0, 1))|1': 0.18823529411764706, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.11834561834561834, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.40310077519379844, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.07235772357723577, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.21666666666666667, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|0': 0.06280587275693311, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|1': 0.1814516129032258, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.08299430431244914, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.3884297520661157, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.10661322645290582, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.33658536585365856, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.06282722513089005, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.17511520737327188, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.05218446601941747, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.11403508771929824}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0


### RESULTS
- Predicted under pi_5 (simulated): 0.0012 (var=0.0008)
- Predicted under pi_3 (simulated): 0.1623 (var=0.0031)
- Observed on real data: 0.2709 (var=0.0337)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate the pure Weighted Additive (WADD) theory from the Soft Strategy Selection theory, we systematically vary the 'cue dispersion' (absolute difference in WADD scores) across trials where Take The Best (TTB) and WADD make opposing predictions. According to the advocated pure WADD theory, when the dispersion is very small (e.g., 0.1), the choice probability will be near 50/50 because the weighted sums are nearly identical, and it will smoothly increase for the WADD-favored option as dispersion increases. In contrast, the competing Soft Strategy Selection theory predicts a non-linear shift: at low dispersions, the decision-maker will fall back on the non-compensatory TTB heuristic, causing the choice probability to sharply favor the TTB-predicted option. By providing a gradient of dispersions where TTB favors Option A but WADD favors Option B (and vice versa), we can clearly observe whether choice probabilities smoothly track the WADD score differences (pure WADD) or exhibit a sharp resurgence of TTB at low dispersions (Soft Strategy Selection).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.8124672603457308, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.8235904531304047, '((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|0': 0.4947154471544715, '((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|1': 0.49188034188034185, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.33564814814814814, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.3330078125, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8603793466807166, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.864920744314266, '((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|0': 0.18195488721804512, '((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|1': 0.18110516934046345, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.6511009870918755, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.6682743837084673}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0


### RESULTS
- Predicted under pi_5 (simulated): 0.2618 (var=0.0026)
- Predicted under pi_3 (simulated): 0.0007 (var=0.0001)
- Observed on real data: 0.2454 (var=0.0011)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.14435946462715105, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.1596409055425449, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.1597576530612245, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.15912636505460218, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.8542864283929018, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.8399058269570335, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.1660264618011097, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.15847482871611557, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.8423000284656988, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.8277689454160042}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2769 (var=0.0015)
**Predicted under pi_5:** 0.0090 (var=0.0007)
**Predicted under pi_3:** 0.2532 (var=0.0067)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8737623762376238, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8550632911392405, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.8492590700051099, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.8212956068503351, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.14496833216045038, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.1294171638366223, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8678861788617886, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8342105263157895, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.13399280575539568, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.14752906976744187, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.12575841147269717, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.13430330162283155, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.1375, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.18194444444444444, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|0': 0.14272970561998216, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|1': 0.16715758468335787}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2751 (var=0.0028)
**Predicted under pi_5:** 0.3414 (var=0.0050)
**Predicted under pi_3:** 0.0011 (var=0.0002)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 1, 1), (0, 0, 1, 1, 1))|0': 0.14172709294660515, '((0, 1, 0, 1, 1), (0, 0, 1, 1, 1))|1': 0.22614840989399293, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.15440210249671485, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.22482014388489208, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.1329155672823219, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.18661971830985916, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.15480895915678525, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.2322695035460993, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.13498349834983497, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.24912280701754386, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.13749183540169824, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.19144981412639406, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15349294850770745, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.2395644283121597, '((0, 1, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.14965741074648395, '((0, 1, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.21062618595825428}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2223 (var=0.0310)
**Predicted under pi_5:** 0.0071 (var=0.0015)
**Predicted under pi_3:** 0.1537 (var=0.0036)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.7996438112199465, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8194988253719655, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.838126540673788, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8732905386547586, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.1927236971484759, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.14327253502511234, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8148951554591468, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8589405911618379, '((0, 1, 1, 0, 0), (0, 0, 1, 1, 1))|0': 0.7894736842105263, '((0, 1, 1, 0, 0), (0, 0, 1, 1, 1))|1': 0.8147261299012272, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.799184505606524, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8554595443833464}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2346 (var=0.0145)
**Predicted under pi_5:** 0.2663 (var=0.0028)
**Predicted under pi_3:** 0.0006 (var=0.0001)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.18787033809689788, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.43912448700410395, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.15119549929676512, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.373015873015873, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.1584089323098395, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.3119891008174387, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|0': 0.1652276295133438, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|1': 0.44813829787234044, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))|0': 0.1494956772334294, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))|1': 0.30097087378640774, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.17841726618705037, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.4268292682926829, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.2128186968838527, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.4677835051546392, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.1541740674955595, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.32993630573248406}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.1605 (var=0.0121)
**Predicted under pi_5:** 0.0081 (var=0.0011)
**Predicted under pi_3:** 0.1393 (var=0.0029)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5051792828685259, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.4955017301038062, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8472032742155525, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8176661264181524, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.7561881188118812, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.7838266384778013, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|0': 0.7674157303370787, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|1': 0.830939226519337, '((1, 0, 0, 1, 1), (0, 1, 0, 1, 1))|0': 0.32517482517482516, '((1, 0, 0, 1, 1), (0, 1, 0, 1, 1))|1': 0.3354330708661417, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.6804308797127468, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.6658259773013872, '((1, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.17035110533159947, '((1, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.1901893287435456, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 0))|0': 0.8219354838709677, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 0))|1': 0.8347826086956521, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.23991195891415995, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.2670157068062827, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.2180746561886051, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.18162083936324167}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2360 (var=0.0005)
**Predicted under pi_5:** 0.1757 (var=0.0010)
**Predicted under pi_3:** 0.0012 (var=0.0002)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, a clear pattern emerges: when the metric is designed to test Theory 1 (pi_5) by measuring JSD to its own predictions, Theory 1 predicts ~0 but the observed data is much higher (~0.16-0.27), and Theory 2 (pi_3) predicts values much closer to the observed data. Conversely, when the metric tests Theory 2, Theory 2 predicts ~0 but the observed data is again high (~0.23-0.27), and Theory 1 provides a closer prediction. This indicates that neither the pure Weighted Additive (WADD) strategy nor the Soft Strategy Selection (mixture of TTB and WADD) accurately captures human behavior. The true data lies somewhere in between, exhibiting patterns that neither model can accommodate fully. This suggests that a completely different cognitive mechanism is at play, rather than a simple probabilistic mixture or a pure compensatory integration.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Sequential Cue Integration' or Evidence Accumulation theory. In this model, decision-makers evaluate cues sequentially in descending order of validity, accumulating evidence for each option. A decision is made as soon as the accumulated evidence difference exceeds a specific internal threshold. If all cues are exhausted without crossing the threshold, a guess is made or a simple tallying rule is used. This naturally unifies non-compensatory (low threshold, acts like TTB) and compensatory (high threshold, acts like WADD) behaviors in a single mechanistic process, without relying on an ad-hoc soft mixture of two distinct strategies."
}
```

## Usage

```json
{
  "prompt_token_count": 15751,
  "candidates_token_count": 356,
  "total_token_count": 17350
}
```
