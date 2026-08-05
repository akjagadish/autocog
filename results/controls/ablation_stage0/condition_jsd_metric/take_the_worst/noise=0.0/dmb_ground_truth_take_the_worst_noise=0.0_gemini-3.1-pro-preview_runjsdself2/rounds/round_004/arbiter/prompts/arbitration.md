# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Autocorrelated Weighted Additive (WADD) Theory: Decision-makers evaluate options by integrating all available cues proportionally to their validities, but their choices are also subject to sequential dependencies (inertia or alternation). A 'stickiness' parameter biases the subjective value of the option that corresponds to the action chosen in the immediately preceding trial, capturing autocorrelation in choice behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
    
    # Calculate validity-weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)
    
    stickiness = float(parameters["stickiness"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply choice stickiness from the previous trial
    if history and "response" in history and len(history["response"]) > 0:
        prev_choice = int(history["response"][-1])
        if 0 <= prev_choice < len(scores):
            scores[prev_choice] += stickiness
    
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


## THEORY 2 — pi_5
Integrated Utility Theory (WADD + Tallying): Decision-makers integrate both compensatory validity weighting (WADD) and non-compensatory feature counting (Tallying) into a single, unified subjective value for each option. Rather than stochastically switching between strategies on a trial-by-trial basis, they form a composite score by taking a weighted average of the WADD and Tallying scores. This composite score is then translated into choice probabilities via a single softmax process subject to decision noise and lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = float(parameters["w"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # WADD scores
    wadd_scores = np.sum(stim * val, axis=1)
    
    # Tallying scores (strict wins)
    tally_scores = np.array([np.sum(a > b), np.sum(b > a)], dtype=float)
    
    # Mix scores instead of probabilities
    mixed_scores = w * wadd_scores + (1.0 - w) * tally_scores
    
    z = beta * (mixed_scores - np.max(mixed_scores))
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


## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [1.0, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Autocorrelated Weighted Additive (WADD) theory from the Integrated Utility Theory (WADD + Tallying), we use a 5-feature design with one highly valid cue (1.0) and four cues of lower validity (0.5). This structure allows us to create 'WADD Tie' trials where the WADD score is exactly equal (e.g., one 1.0 cue vs. two 0.5 cues), but Tallying strongly favors the option with more winning features. In these trials, the Integrated Utility theory predicts a strong preference for the Tallying-favored option, whereas the Autocorrelated WADD theory predicts a 50/50 overall choice rate that is locally driven by stickiness to the previous trial's response. We also include 'Tallying Tie' trials where both options win on the same number of features but differ in WADD score, isolating the pure validity weighting without non-compensatory interference.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.4823529411764706, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.5075697211155379, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.3993517017828201, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.4474187380497132, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.39429530201342283, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.4566176470588235, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.46525776496958055, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.5013581684128832, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.47393822393822393, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.4965277777777778}
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
- Predicted under pi_6 (simulated): 0.0006 (var=0.0033)
- Predicted under pi_5 (simulated): 0.0386 (var=0.0020)
- Observed on real data: 0.0949 (var=0.0044)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.9, 0.9, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 8: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Integrated Utility Theory (WADD + Tallying) from the Autocorrelated Weighted Additive (WADD) theory, we use a 6-feature design with two high-validity cues (0.9) and four moderate-validity cues (0.6). This structure allows us to create 'WADD Tie' trials where the WADD score is exactly equal (e.g., two 0.9 cues vs. three 0.6 cues), but Tallying strongly favors the option with more winning features (3 vs. 2). In these trials, the Integrated Utility theory predicts a strong main effect for the Tallying-favored option, whereas the Autocorrelated WADD theory predicts a 50/50 overall choice rate that is locally driven by stickiness to the previous trial's response. We also include 'Tallying Tie' trials where both options win on the same number of features but differ in WADD score, as well as congruent trials where both models agree, ensuring a robust test of the compensatory vs. non-compensatory integration.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 0))|0': 0.8469387755102041, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 0))|1': 0.8311546840958606, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0))|0': 0.7644949193066348, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0))|1': 0.7644006227296316, '((0, 0, 1, 0, 1, 1), (1, 1, 0, 1, 0, 0))|0': 0.7409502262443439, '((0, 0, 1, 0, 1, 1), (1, 1, 0, 1, 0, 0))|1': 0.726528384279476, '((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0))|0': 0.7489247311827957, '((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0))|1': 0.7263888888888889, '((0, 1, 0, 1, 1, 0), (1, 0, 1, 0, 0, 0))|0': 0.16280806572068707, '((0, 1, 0, 1, 1, 0), (1, 0, 1, 0, 0, 0))|1': 0.167624944714728, '((0, 0, 1, 1, 1, 0), (1, 1, 0, 0, 0, 0))|0': 0.22853894258101193, '((0, 0, 1, 1, 1, 0), (1, 1, 0, 0, 0, 0))|1': 0.2351982618142314, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.2680131004366812, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.2528280542986425, '((1, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 1))|0': 0.2436247723132969, '((1, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 1))|1': 0.27136752136752135}
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
- Predicted under pi_6 (simulated): 0.0253 (var=0.0035)
- Predicted under pi_5 (simulated): 0.0015 (var=0.0002)
- Observed on real data: 0.1114 (var=0.0027)

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
**Predicted under pi_6:** 0.1029 (var=0.0026)
**Predicted under pi_5:** 0.1674 (var=0.0015)

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
**Predicted under pi_6:** 0.0364 (var=0.0042)
**Predicted under pi_5:** 0.0009 (var=0.0001)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.1644 (var=0.0058)
**Predicted under pi_6:** 0.0181 (var=0.0031)
**Predicted under pi_5:** 0.0334 (var=0.0013)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.0327 (var=0.0005)
**Predicted under pi_6:** 0.0459 (var=0.0021)
**Predicted under pi_5:** 0.0009 (var=0.0002)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.15749525616698293, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.1589895988112927, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.14057507987220447, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.1542997542997543, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8291316526610645, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8264746227709191, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.3562231759656652, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.3474114441416894, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.7488196411709159, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.7942882641677822, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.6647093364650617, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.6510279388508171, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.7594123048668503, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8104340900039825, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8479498861047836, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8378524945770065}
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

**Observed (real) value:** 0.0081 (var=0.0003)
**Predicted under pi_6:** 0.0275 (var=0.0033)
**Predicted under pi_5:** 0.0042 (var=0.0003)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.405341446923597, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.4815418023887079, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.34856351678781583, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.41601255886970173, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.3780487804878049, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.45619158878504673, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.27576503693281745, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.3546244251405212, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.3910427807486631, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.45741150442477874, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.3926247288503254, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.4783677482792527}
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

**Observed (real) value:** 0.1147 (var=0.0024)
**Predicted under pi_6:** 0.0216 (var=0.0023)
**Predicted under pi_5:** 0.0863 (var=0.0015)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.24265165728580362, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.22438780609695153, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.5080886758538047, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.5059554634904194, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.6605200945626477, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.6404040404040404, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.1933404940923738, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.18150208623087621, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.30350553505535055, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.23052464228934816, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 0))|0': 0.81474738279472, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 0))|1': 0.7754811119030649, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.6551724137931034, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.6716061185468452, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.7492781520692974, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.7582128777923784}
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

**Observed (real) value:** 0.0670 (var=0.0019)
**Predicted under pi_6:** 0.0266 (var=0.0030)
**Predicted under pi_5:** 0.0264 (var=0.0008)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8129390018484288, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.7923627684964201, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.18975515463917525, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.20695754716981132, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.20986547085201793, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.19299610894941635, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13988439306358383, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.18059701492537314, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8065159574468085, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.7896205357142857, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.2611336032388664, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.26798029556650244}
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

**Observed (real) value:** 0.0668 (var=0.0020)
**Predicted under pi_6:** 0.0447 (var=0.0025)
**Predicted under pi_5:** 0.0007 (var=0.0002)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, neither Autocorrelated WADD (Theory 1) nor Integrated Utility Theory (Theory 2) consistently captures the observed data. Theory 1 (pi_6) performs reasonably well on Experiments 4, 6, and 10, while Theory 2 (pi_5) performs better on Experiments 7 and 8. However, both models frequently underestimate the JSD metric, indicating they fail to capture the true variance and conditional choice probabilities in the human data. The presence of strong sequential dependencies (as modeled in Theory 1) and non-compensatory heuristic use (as modeled in Theory 2) both seem to be partial drivers of behavior, but mutually exclusive. This suggests the need for a unified theory that incorporates both heuristic integration and sequential choice inertia.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Heuristic-Inertia Hybrid Theory' that unifies both non-compensatory feature counting (Tallying) and compensatory weighting (WADD) while also applying a sequential choice stickiness parameter. Alternatively, explore a probabilistic strategy-selection model (e.g., Take-The-Best vs WADD) where the probability of selecting a strategy is influenced by the previous trial's outcome or choice. This will replace the purely compensatory Autocorrelated WADD theory with one that accounts for both heuristic shortcuts and temporal dependencies."
}
```

## Usage

```json
{
  "prompt_token_count": 17435,
  "candidates_token_count": 314,
  "total_token_count": 18638
}
```
