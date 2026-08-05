# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People make choices by integrating all available feature information, weighting each cue by its subjective validity. Unlike Tallying (which weights all cues equally) or Take The Best (which stops at the first discriminating cue), the Weighted Additive (WADD) model computes an overall expected value for each option by summing the products of the feature values and their validities. Choice probabilities are then generated via a softmax function over these weighted sums, allowing for graded sensitivity to both the number of supporting features and their relative importance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: compute the weighted sum of features for each option
    score_a = np.sum(stim[0] * val)
    score_b = np.sum(stim[1] * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_5
Rank-Weighted Additive Model: People make choices by integrating all available feature information, but rather than weighting each cue by its exact subjective validity (as in WADD) or weighting all cues equally (as in Tallying), they weight cues by their ordinal rank. The most valid cue receives the highest integer weight, the next most valid receives the second highest, and so on. This captures a heuristic simplification where decision-makers respect the order of cue importance but fail to perfectly integrate precise cardinal probabilities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Ranks: ascending order, so highest validity gets largest integer weight.
    weights = rankdata(val)
    
    score_a = np.sum(stim[0] * weights)
    score_b = np.sum(stim[1] * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.6, 0.58, 0.56, 0.54]

**Trial pairs (n=10):**
  trial 1: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) model from the Rank-Weighted Additive model by exploiting the difference between cardinal validities and ordinal ranks. We use a set of validities with a highly non-linear spacing: one very high validity (0.95) and four clustered lower validities (0.60, 0.58, 0.56, 0.54). For WADD, the sum of two or three of the lower validities can easily outweigh or be outweighed by the top validity depending on the exact cardinal values. However, in the Rank model, the validities are transformed to strictly linear integer weights (5, 4, 3, 2, 1). This creates adversarial trials where WADD predicts one choice based on the exact sum of probabilities, while the Rank model predicts the opposite choice because the ordinal ranks dictate a different balance of power.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.21328866554997208, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.2988505747126437, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.8093197643277986, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.737094837935174, '((0, 1, 0, 1, 0), (1, 0, 0, 0, 0))|0': 0.23014586709886548, '((0, 1, 0, 1, 0), (1, 0, 0, 0, 0))|1': 0.3027090694935218, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.2755102040816326, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.32577565632458233, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.14285714285714285, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.1697459584295612, '((0, 1, 0, 0, 1), (1, 0, 0, 0, 0))|0': 0.22102689486552568, '((0, 1, 0, 0, 1), (1, 0, 0, 0, 0))|1': 0.31755725190839695, '((0, 0, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.23446658851113716, '((0, 0, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.2746478873239437, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.24703344120819848, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.30969267139479906, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.28171731789676796, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.3492822966507177, '((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|0': 0.24294086307938198, '((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|1': 0.29040097205346294}
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
- Predicted under pi_3 (simulated): 0.0007 (var=0.0002)
- Predicted under pi_5 (simulated): 0.0826 (var=0.0014)
- Observed on real data: 0.0313 (var=0.0006)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.98, 0.75, 0.65, 0.55, 0.51]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design systematically dissociates the Weighted Additive (WADD) model from the Rank-Weighted Additive model by leveraging a specific distribution of validities: one very high validity and four spaced-out lower validities. While WADD integrates the exact cardinal validities, the Rank model converts them into strictly ordinal integer weights (5, 4, 3, 2, 1). By carefully pairing options, we create adversarial scenarios where the sum of specific cardinal validities favors one option (e.g., 0.98 + 0.51 = 1.49 vs 0.75 + 0.65 = 1.40), but the sum of their ordinal ranks favors the other (e.g., 5 + 1 = 6 vs 4 + 3 = 7). Additional trials where the models predict ties versus strict preferences ensure a robust quantitative dissociation across the parameter space.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.5069767441860465, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.5013793103448276, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.4929906542056075, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.4945205479452055, '((1, 0, 0, 0, 0), (0, 0, 1, 0, 1))|0': 0.15668617739756865, '((1, 0, 0, 0, 0), (0, 0, 1, 0, 1))|1': 0.1406816533720087, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.12570093457943926, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1226027397260274, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.1412280701754386, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.16176470588235295, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|0': 0.5113759479956663, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|1': 0.49338146811071, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.8599009900990099, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.8493670886075949, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.11822660098522167, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.16323024054982818}
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
- Predicted under pi_3 (simulated): 0.0564 (var=0.0011)
- Predicted under pi_5 (simulated): 0.0004 (var=0.0001)
- Observed on real data: 0.1406 (var=0.0039)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.8432741116751269, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.841897233201581, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.16182937554969218, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.13407304669440592, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.855036855036855, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.8370607028753994, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 0))|0': 0.8577777777777778, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 0))|1': 0.84, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.15462868769074262, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.14940771876194114, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.13982213438735178, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.14657360406091371, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.8566864445458695, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.8481192334989354, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))|0': 0.1502231036192365, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))|1': 0.14718888186986734}
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

**Observed (real) value:** 0.2545 (var=0.0057)
**Predicted under pi_3:** 0.1999 (var=0.0042)
**Predicted under pi_5:** 0.2169 (var=0.0037)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.13475997686524002, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.15946348733233978, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 1))|0': 0.8554948391013965, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 1))|1': 0.8326693227091634, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.8582089552238806, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.8350083752093802, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.1322314049586777, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.1412535079513564, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8722910216718266, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8691335740072202, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8629191321499013, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.8593073593073594, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.48135874067937034, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.5180217937971501, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.13944954128440368, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.12748091603053435, '((1, 0, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.48088360237892946, '((1, 0, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.5110384300899428, '((1, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.12789827973074047, '((1, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.12699905926622765, '((1, 0, 0, 0, 1), (0, 1, 0, 1, 0))|0': 0.4930555555555556, '((1, 0, 0, 0, 1), (0, 1, 0, 1, 0))|1': 0.5163043478260869, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|0': 0.1639871382636656, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|1': 0.1610968294772922}
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

**Observed (real) value:** 0.0570 (var=0.0004)
**Predicted under pi_3:** 0.0093 (var=0.0001)
**Predicted under pi_5:** 0.0144 (var=0.0001)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|0': 0.2504970178926441, '((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|1': 0.23338115734098516, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.6881229235880398, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.6593959731543624, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15008090614886732, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.20567375886524822, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.40920554854981084, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.429493545183714, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8176943699731903, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8206831119544592, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.3243927125506073, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.35098522167487683, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.7578084997439836, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.7656344869459624, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.24202822791427078, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.26964671953857244}
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

**Observed (real) value:** 0.1078 (var=0.0018)
**Predicted under pi_3:** 0.0003 (var=0.0002)
**Predicted under pi_5:** 0.0170 (var=0.0003)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.14606741573033707, '((0, 0, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.14012738853503184, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.1379992091735864, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.12769704975781593, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.48846508766533375, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.5203357004519045, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.13786078098471988, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.14855305466237942, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.5008685581933989, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.49554234769687966, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8620078041858815, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8465421504290762}
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

**Observed (real) value:** 0.0288 (var=0.0006)
**Predicted under pi_3:** 0.0336 (var=0.0003)
**Predicted under pi_5:** 0.1797 (var=0.0029)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.7541576283441793, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.7762742444745151, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.7403598971722365, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.7965474722564735, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.7891529555149299, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8044920877998979, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.6914893617021277, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.7326051779935275, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8217270194986073, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8644470868014269, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.20877192982456141, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1638211382113821, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.8470005503577325, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.803776129467296, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.1629162916291629, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.15869827239855364}
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

**Observed (real) value:** 0.0598 (var=0.0010)
**Predicted under pi_3:** 0.0004 (var=0.0002)
**Predicted under pi_5:** 0.0566 (var=0.0008)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.2208515718265022, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.34314627414903404, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.2238478419897586, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.35219399538106233, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.21999242711094283, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.3534932221063608, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.2347860227718885, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.32763532763532766, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.24054180289584307, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.26387936943111723, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.22110972048393826, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.3059019118869493, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.7667826951765291, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.7222653219550039, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.21905891656781337, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.28291316526610644}
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

**Observed (real) value:** 0.1328 (var=0.0032)
**Predicted under pi_3:** 0.1290 (var=0.0020)
**Predicted under pi_5:** 0.1103 (var=0.0018)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the eight experiments, neither the Weighted Additive (WADD) model (THEORY 1) nor the Rank-Weighted Additive model (THEORY 2) universally captures the observed data. WADD performs very well on Experiments 1, 6, and 8, whereas the Rank model provides better predictions for Experiments 5 and 7. In Experiment 2, both models severely underpredict the observed metric, and in Experiment 3, both are somewhat misaligned. The failure of either strictly compensatory model (using cardinal or ordinal weights) to consistently explain the data suggests that human decision-makers might not be integrating all cues. Instead, they may be using a non-compensatory heuristic that ignores some information entirely.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace THEORY 2 with a non-compensatory 'Take-The-Best' (TTB) heuristic theory. In this theory, decision-makers do not integrate all features. Instead, they inspect cues sequentially in descending order of their validities. The search stops at the first cue that discriminates between the two options, and the choice is made strictly based on that single cue. If no cues discriminate, they guess randomly. A soft-max or error rate parameter can be added to allow for stochasticity. This starkly contrasts with the fully compensatory WADD model and may better capture the data in environments where subjects simplify their decision processes."
}
```

## Usage

```json
{
  "prompt_token_count": 16176,
  "candidates_token_count": 318,
  "total_token_count": 17417
}
```
