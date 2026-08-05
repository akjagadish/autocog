# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Weighted Additive (WADD) theory posits that decision-makers integrate all available cues by weighting them proportionally to their validities. The overall value for each option is the validity-weighted sum of its features. Choices are made probabilistically via a softmax over these aggregated values, incorporating response noise through an inverse temperature parameter and an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
    
    # Calculate validity-weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)
    
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate WADD and Tallying, we use 5 features with a steep drop-off in validities. WADD integrates all cues weighted by their validity, allowing a small number of highly valid cues to outweigh a larger number of less valid cues. Tallying, on the other hand, ignores validities and simply counts the number of winning features for each option. We design trials where Option A wins on fewer but highly valid features (strongly favored by WADD), while Option B wins on more but less valid features (favored by Tallying). We also include trials where Tallying predicts a perfect tie (because both options win on the same number of features), whereas WADD strongly predicts one option due to the disparity in cue validities.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.415614773258532, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.39308530627583615, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.6572261557684298, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.6523551479783243, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.333597150771666, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.354157501099868, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.5937165298107818, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.5707853926963482, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.2462406015037594, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.2575885328836425, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.74, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.7665306122448979}
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
- Predicted under pi_3 (simulated): 0.0004 (var=0.0002)
- Predicted under pi_2 (simulated): 0.0747 (var=0.0008)
- Observed on real data: 0.1644 (var=0.0058)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Tallying (advocated theory) from Weighted Additive (WADD) (competing theory), we use a 5-feature design with a significant disparity in validities. Tallying completely ignores cue validities and simply counts the number of winning features for each option, whereas WADD integrates all features weighted by their respective validities. We design trials where one option wins on fewer but highly valid features (strongly favored by WADD), while the other option wins on a greater number of less valid features (favored by Tallying). Additionally, we include trials where Tallying predicts a perfect tie (because both options win on the exact same number of features), but WADD makes a deterministic prediction due to the differing validities of those features.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5014989293361884, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5281947261663286, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15499070055796652, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.1492899203325251, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.8430979133226324, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.8453206239168111, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.13958060288335516, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.15560640732265446, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8514970059880239, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8496732026143791, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.48419721871049304, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.49814585908529047}
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
- Predicted under pi_3 (simulated): 0.0430 (var=0.0005)
- Predicted under pi_2 (simulated): 0.0003 (var=0.0002)
- Observed on real data: 0.0327 (var=0.0005)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0), (0, 0, 1, 1))|0': 0.1259320629660315, '((1, 1, 0, 0), (0, 0, 1, 1))|1': 0.14502529510961215, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1400454201362604, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.12531581606872158, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8697047496790757, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8687561214495593, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.1400296882731321, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.14186193793540217, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.1333997013439522, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.12696417347580138, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8359240069084629, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8751023751023751, '((0, 1, 0, 1), (1, 0, 0, 0))|0': 0.8776622090143635, '((0, 1, 0, 1), (1, 0, 0, 0))|1': 0.8621125869702719, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.85957213384531, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.8773213280810355}
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

**Observed (real) value:** 0.2429 (var=0.0060)
**Predicted under pi_3:** 0.1594 (var=0.0017)
**Predicted under pi_2:** 0.1888 (var=0.0025)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.15036743923120408, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1583833970507919, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.12419070041200707, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.13992635455023672, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8402915838303512, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8589191774270684, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8630310716550412, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8645575877409788, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.16549069916613213, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.14061734443900048, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.4925219185146983, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.473208910295003, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8657289002557544, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.861984282907662, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.855036855036855, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8638768638768639}
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

**Observed (real) value:** 0.0405 (var=0.0004)
**Predicted under pi_3:** 0.0042 (var=0.0001)
**Predicted under pi_2:** 0.0010 (var=0.0002)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, neither Weighted Additive (WADD) nor Tallying perfectly captures the observed data. In Experiment 1, the observed metric is 0.1644, while WADD predicts 0.0004 and Tallying predicts 0.0747. In Experiment 2, WADD predicts 0.0430 (close to the observed 0.0327), while Tallying predicts 0.0003. Experiments 3 and 4 also show mixed performance, with both theories often underpredicting the divergence or failing to capture the exact choice distributions. This suggests that people might not be using a purely compensatory strategy (WADD) nor a completely unweighted counting strategy (Tallying). A lexicographic strategy that leverages validities without full integration might better explain the variance in the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying with the 'Take-The-Best' (TTB) heuristic. TTB searches through the features in descending order of their cue validities. It stops at the first feature that discriminates between the two options and chooses the option that has the higher value on that cue. If no cues discriminate, it guesses randomly. This introduces a fast-and-frugal, non-compensatory lexicographic strategy that relies heavily on cue order, offering a strong theoretical alternative to both the fully compensatory WADD and the validity-ignoring Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 8452,
  "candidates_token_count": 337,
  "total_token_count": 9677
}
```
