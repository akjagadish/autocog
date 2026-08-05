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
Choice Inertia (Autocorrelation) with Tallying: Decision-makers evaluate options by counting the total number of positive features for each option (unweighted sum), but their current choice is also biased by their motor response on the immediately preceding trial. A 'stickiness' parameter shifts the utility toward the previously chosen option index (A or B), reflecting the cognitive ease of repeating a past action regardless of the specific product features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) state.")
    
    a, b = stim[0], stim[1]
    
    # Base utility: unweighted sum of features (Tallying)
    scores = np.array([np.sum(a), np.sum(b)])
    
    # Choice Inertia: boost the score of the previously chosen action
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = int(history["response"][-1])
        if 0 <= last_resp < 2:
            phi = float(parameters["phi"])
            scores[last_resp] += phi
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the modified utilities with numerical stability
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p_core = p / np.sum(p)
    
    # Mix with uniform guessing (lapse rate)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure normalization
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_3
Weighted Additive (WADD) Model: People integrate all available information by computing a weighted sum of feature values for each option. The weights are proportional to the log-odds of the cue validities. This allows for compensatory decision-making, where multiple weakly predictive features can jointly override a single highly predictive feature. Choice probabilities are determined by applying a softmax function to the weighted sums, with an independent lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Extract validities and clip to avoid log(0) or division by zero
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-4, 1.0 - 1e-4)
    
    # Transform validities to weights using log-odds
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p_core = p / np.sum(p)
    
    # Mix with uniform guessing (lapse rate)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure normalization
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.95, 0.8, 0.74, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 13: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  trial 15: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 1, 1), (0, 0, 1, 0))|0': 0.360450563204005, '((0, 0, 1, 1), (0, 0, 1, 0))|1': 0.37962037962037964, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.5450191570881227, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.5701058201058201, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.40406976744186046, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.35881294964028776, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.6241299303944315, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.6609808102345416, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.6415094339622641, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.64293659621802, '((1, 0, 1, 1), (0, 1, 1, 0))|0': 0.3614103819784525, '((1, 0, 1, 1), (0, 1, 1, 0))|1': 0.4274711168164313, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.4919254658385093, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.4894472361809045, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.6143790849673203, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.6235827664399093, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.44812680115273773, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.3589511754068716, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.39308578745198464, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.37684003925417076, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.5104895104895105, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.4883227176220807, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.6416893732970027, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.6409921671018277, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.479064039408867, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.5101214574898786, '((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.6313465783664459, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.6331096196868009, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.6155632984901278, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.65814696485623, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.538135593220339, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.508177570093458}
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
- Predicted under pi_5 (simulated): 0.0021 (var=0.0014)
- Predicted under pi_3 (simulated): 0.0479 (var=0.0010)
- Observed on real data: 0.2219 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.64, 0.8, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 2: A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  trial 3: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 7: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 9: A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 10: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 12: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 13: A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 14: A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 15: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 16: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.844050258684405, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8232662192393736, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.1463917525773196, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.1285171102661597, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.12106918238993711, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.14351851851851852, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.11691762621789194, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.13263785394932937, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8626760563380281, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8689759036144579, '((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.1567489114658926, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.13051305130513052, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.7374658158614403, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.7610241820768137, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.1415478615071283, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.12102689486552567, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.13434452871072589, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.12314709236031927, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.14465408805031446, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.1359338061465721, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.13663663663663664, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.17094017094017094, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8563869992441421, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8259958071278826, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.8853006681514477, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.8691796008869179, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8566739606126915, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8668171557562077, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.14450867052023122, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.14711191335740073, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8706677937447168, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8508914100486223}
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
- Predicted under pi_5 (simulated): 0.0409 (var=0.0020)
- Predicted under pi_3 (simulated): 0.0012 (var=0.0002)
- Observed on real data: 0.2745 (var=0.0009)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (1, 1, 0, 1))|0': 0.8443465491923642, '((1, 0, 0, 1), (1, 1, 0, 1))|1': 0.8437118437118437, '((0, 1, 0, 0), (1, 1, 1, 0))|0': 0.8369098712446352, '((0, 1, 0, 0), (1, 1, 1, 0))|1': 0.8530734632683659, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8735177865612648, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8802163833075735, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8077753779697624, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8526551982049364, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.16240266963292546, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.13873473917869034, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.14475138121546963, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.15083798882681565, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.8528301886792453, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.8042105263157895, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.13489208633093525, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.15843023255813954, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.8515625, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.8517699115044248, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.16393442622950818, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.14526129317980513, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8734756097560976, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8483606557377049, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.14634146341463414, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.1402936378466558, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.17346938775510204, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.15798922800718132, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8537020517395183, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8527245949926362, '((0, 1, 1, 1), (1, 0, 1, 1))|0': 0.8435754189944135, '((0, 1, 1, 1), (1, 0, 1, 1))|1': 0.8475138121546961}
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

**Observed (real) value:** 0.2701 (var=0.0005)
**Predicted under pi_5:** 0.0553 (var=0.0010)
**Predicted under pi_3:** 0.0021 (var=0.0002)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.1539301310043668, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.1414027149321267, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.14615829000577701, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.13964686998394862, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8874259381171824, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.7758007117437722, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.8291015625, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8260309278350515, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.15993907083015993, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.1868583162217659, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.15517241379310345, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.15759312320916904, '((1, 0, 1, 0), (0, 0, 1, 1))|0': 0.48265460030165913, '((1, 0, 1, 0), (0, 0, 1, 1))|1': 0.5527426160337553, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.12248865845755022, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.22957198443579765, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8393316195372751, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8405088062622309, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.1339366515837104, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.16115107913669063, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.154006968641115, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.18904109589041096, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.15240384615384617, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.16475409836065574, '((0, 0, 1, 1), (0, 1, 1, 0))|0': 0.5254942767950052, '((0, 0, 1, 1), (0, 1, 1, 0))|1': 0.4803337306317044, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8481012658227848, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8198847262247838}
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

**Observed (real) value:** 0.2327 (var=0.0048)
**Predicted under pi_5:** 0.0146 (var=0.0020)
**Predicted under pi_3:** 0.0302 (var=0.0001)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 1, 0), (1, 1, 0, 1))|0': 0.1453900709219858, '((1, 0, 1, 0), (1, 1, 0, 1))|1': 0.15476190476190477, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8635863586358636, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8630751964085297, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.1334056399132321, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.16970387243735763, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.821656050955414, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8660647103085026, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8529672276350753, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8464977645305514, '((1, 0, 0, 1), (0, 1, 0, 1))|0': 0.13004484304932734, '((1, 0, 0, 1), (0, 1, 0, 1))|1': 0.14317180616740088, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8511560693641619, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8709386281588448, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.16025641025641027, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.14414414414414414, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.1478770131771596, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.1611459265890779, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8492159227985525, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8524590163934426, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.15625, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.1340909090909091, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8645990922844176, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8451882845188284, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8070175438596491, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8415178571428571, '((0, 0, 0, 0), (1, 0, 1, 0))|0': 0.8527827648114902, '((0, 0, 0, 0), (1, 0, 1, 0))|1': 0.8542274052478134, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.15870786516853932, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.12867647058823528, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8316546762589928, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8552036199095022}
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

**Observed (real) value:** 0.2529 (var=0.0004)
**Predicted under pi_5:** 0.0459 (var=0.0010)
**Predicted under pi_3:** 0.0017 (var=0.0001)

### Experiment 6
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.8734177215189873, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.869815668202765, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8942891859052248, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8751279426816786, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.1459227467811159, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.1407811080835604, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.16621004566210046, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.16453900709219857, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.875, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8569819819819819, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8639028475711893, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8432343234323433, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.1772853185595568, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.14007421150278293, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.1342434584755404, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.11834961997828447, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.24841915085817526, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.2784992784992785, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.1471264367816092, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.16767676767676767, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8655172413793103, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.867741935483871, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.15876777251184834, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.12481857764876633, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.7990919409761634, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.779107725788901, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.14925373134328357, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.14274981217129978, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8751369112814896, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.859075535512965}
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

**Observed (real) value:** 0.2659 (var=0.0000)
**Predicted under pi_5:** 0.0415 (var=0.0017)
**Predicted under pi_3:** 0.0008 (var=0.0001)

### Experiment 7
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.49345549738219896, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.5125482625482626, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8589473684210527, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8867924528301887, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.47058823529411764, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.4901785714285714, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8779620853080569, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8535564853556485, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8336980306345733, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.8734177215189873, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8532763532763533, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8770491803278688, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.4674329501915709, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.4762833008447043, '((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.8590785907859079, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.8573446327683616, '((1, 1, 0, 0), (0, 0, 1, 0))|0': 0.18487394957983194, '((1, 1, 0, 0), (0, 0, 1, 0))|1': 0.12462235649546828, '((1, 0, 0, 0), (0, 0, 1, 0))|0': 0.4690265486725664, '((1, 0, 0, 0), (0, 0, 1, 0))|1': 0.4940652818991098, '((1, 0, 1, 1), (1, 1, 1, 1))|0': 0.8663366336633663, '((1, 0, 1, 1), (1, 1, 1, 1))|1': 0.8520408163265306, '((1, 0, 0, 0), (0, 1, 0, 1))|0': 0.7866666666666666, '((1, 0, 0, 0), (0, 1, 0, 1))|1': 0.878095238095238, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.17759562841530055, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.13528591352859135, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.4640151515151515, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.49794238683127573, '((0, 0, 1, 0), (0, 1, 1, 0))|0': 0.829683698296837, '((0, 0, 1, 0), (0, 1, 1, 0))|1': 0.885480572597137}
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

**Observed (real) value:** 0.2714 (var=0.0075)
**Predicted under pi_5:** 0.0224 (var=0.0025)
**Predicted under pi_3:** 0.0432 (var=0.0003)

### Experiment 8
**Design**
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8294360385144429, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.863932898415657, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8645833333333334, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8803418803418803, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.13836477987421383, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.1292517006802721, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.7344173441734417, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.7662337662337663, '((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8619547211485368, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8624930128563443, '((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.7281553398058253, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.7813229571984436, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.7661971830985915, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.7853211009174312, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.18662952646239556, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.18207024029574861, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.2286652078774617, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.2110609480812641, '((0, 1, 1, 1), (0, 0, 1, 1))|0': 0.19558823529411765, '((0, 1, 1, 1), (0, 0, 1, 1))|1': 0.15803571428571428, '((0, 0, 0, 1), (1, 0, 0, 1))|0': 0.8539176626826029, '((0, 0, 0, 1), (1, 0, 0, 1))|1': 0.8510028653295129, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.15886524822695036, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.13789954337899543, '((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.8763005780346821, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.8330708661417323, '((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.18181818181818182, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.13421052631578947, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8483606557377049, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8727134146341463}
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

**Observed (real) value:** 0.2597 (var=0.0020)
**Predicted under pi_5:** 0.0244 (var=0.0026)
**Predicted under pi_3:** 0.0014 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments, both Theory 1 (Tallying with Choice Inertia) and Theory 2 (Weighted Additive Model) fail to accurately predict human behavior. The JSD metric measures the divergence between the real data's conditional choice profile and the proposing theory's reference profile. In every experiment, the observed JSD is consistently high (0.22 - 0.27), whereas both models predict values near zero for their own proposals and relatively low values for the opponent's proposals. This massive underestimation of the divergence indicates that neither unweighted tallying (even with motor inertia) nor fully compensatory weighted integration captures the true cognitive mechanism subjects are using. Given that subjects are provided with explicit feature validities, they are likely employing a non-compensatory heuristic rather than exhaustive integration or naive tallying.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Theory 1 with the 'Take-The-Best' (TTB) heuristic. In this lexicographic theory, decision-makers do not integrate all features or simply tally them. Instead, they rank the features in descending order of their provided validities. They inspect features one by one and stop search at the first feature that discriminates between the two options (i.e., one option has a 1 and the other has a 0), choosing the option with the positive value for that cue. If no cues discriminate, they guess randomly. Incorporate a lapse rate (epsilon) to account for noise or attention lapses. This introduces a fundamentally different, non-compensatory mechanism that strongly contrasts with the exhaustive WADD model of Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 21678,
  "candidates_token_count": 361,
  "total_token_count": 22837
}
```
