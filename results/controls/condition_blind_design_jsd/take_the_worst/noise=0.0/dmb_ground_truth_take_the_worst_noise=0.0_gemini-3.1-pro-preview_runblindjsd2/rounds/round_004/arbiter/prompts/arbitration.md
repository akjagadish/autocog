# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People use a Weighted Additive (WADD) strategy to compare options. They compute a subjective value for each option by summing its feature values weighted by their explicit validities. Decisions are made probabilistically using a softmax function over these values, with occasional random lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

## THEORY 2 — pi_6
Decisions are driven by a Weighted Additive (WADD) evaluation of the options, combined with a sequential 'choice inertia' mechanism. Individuals compute the subjective value of each option by weighting its features by their respective validities. However, when translating these values into choices, people exhibit a tendency to repeat their physical response from the immediately preceding trial. This inertia acts as a baseline shift in the evidence required to choose the previously selected option, capturing sequential dependencies in decision-making.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive value computation
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    beta = float(parameters["beta"])
    rho = float(parameters["rho"])
    epsilon = float(parameters["epsilon"])
    
    # Choice inertia: boost the logit of the previously chosen option
    bias_a = 0.0
    bias_b = 0.0
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = history["response"][-1]
        if last_resp == 0:
            bias_a = rho
        elif last_resp == 1:
            bias_b = rho
            
    logits = np.array([beta * score_a + bias_a, beta * score_b + bias_b])
    
    # Numerically stable softmax
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    p_core = exp_logits / np.sum(exp_logits)
    
    # Trembling hand lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.64, 0.73, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 5: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 8: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 13: A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 15: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 16: A=[0, 0, 1, 1]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 1, 1), (0, 1, 0, 1))|0': 0.37590711175616837, '((0, 0, 1, 1), (0, 1, 0, 1))|1': 0.3627362736273627, '((1, 1, 1, 0), (1, 1, 0, 1))|0': 0.28975265017667845, '((1, 1, 1, 0), (1, 1, 0, 1))|1': 0.29232386961093587, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.13234200743494423, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.1934065934065934, '((1, 1, 0, 0), (0, 0, 0, 0))|0': 0.15348288075560804, '((1, 1, 0, 0), (0, 0, 0, 0))|1': 0.14270724029380902, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.18040089086859687, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.17849223946784923, '((0, 0, 0, 0), (1, 0, 1, 0))|0': 0.8732708612226685, '((0, 0, 0, 0), (1, 0, 1, 0))|1': 0.8470254957507082, '((1, 0, 1, 1), (0, 1, 0, 0))|0': 0.13951310861423222, '((1, 0, 1, 1), (0, 1, 0, 0))|1': 0.16256830601092895, '((0, 1, 1, 1), (1, 0, 0, 1))|0': 0.18993135011441648, '((0, 1, 1, 1), (1, 0, 0, 1))|1': 0.18142548596112312, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8463667820069204, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.7774647887323943, '((0, 1, 0, 0), (1, 0, 1, 0))|0': 0.8362068965517241, '((0, 1, 0, 0), (1, 0, 1, 0))|1': 0.865036231884058, '((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8448883666274971, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.8472075869336143, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.7600554785020804, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.7442075996292864, '((0, 0, 1, 1), (0, 0, 1, 0))|0': 0.18647764449291168, '((0, 0, 1, 1), (0, 0, 1, 0))|1': 0.18346545866364666, '((1, 1, 0, 1), (0, 0, 1, 0))|0': 0.13375130616509928, '((1, 1, 0, 1), (0, 0, 1, 0))|1': 0.14472123368920523, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.7503337783711616, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.7602283539486203}
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
- Predicted under pi_3 (simulated): 0.0014 (var=0.0001)
- Predicted under pi_6 (simulated): 0.0039 (var=0.0008)
- Observed on real data: 0.0658 (var=0.0007)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.73, 0.56, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 9: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 15: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.14511494252873564, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.22282608695652173, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.778852798894264, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8750580585229911, '((1, 0, 1, 0), (1, 1, 1, 0))|0': 0.7762237762237763, '((1, 0, 1, 0), (1, 1, 1, 0))|1': 0.8608294930875576, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.6937269372693727, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8328267477203647, '((1, 1, 0, 1), (0, 0, 1, 0))|0': 0.11751412429378531, '((1, 1, 0, 1), (0, 0, 1, 0))|1': 0.13989071038251366, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.2342857142857143, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4832, '((1, 1, 0, 1), (1, 0, 1, 0))|0': 0.15412186379928317, '((1, 1, 0, 1), (1, 0, 1, 0))|1': 0.205607476635514, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.6766169154228856, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8176100628930818, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.7962138084632516, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.8647450110864745, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.14637146371463713, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.21580547112462006, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.15041572184429328, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.2976939203354298, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.7439024390243902, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.8396436525612472, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.12920738327904452, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.16609783845278725, '((0, 1, 1, 0), (1, 0, 1, 1))|0': 0.7693370165745856, '((0, 1, 1, 0), (1, 0, 1, 1))|1': 0.8698884758364313, '((1, 1, 1, 0), (1, 1, 0, 1))|0': 0.3270321361058601, '((1, 1, 1, 0), (1, 1, 0, 1))|1': 0.5970350404312669}
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
- Predicted under pi_3 (simulated): 0.0049 (var=0.0001)
- Predicted under pi_6 (simulated): 0.0019 (var=0.0007)
- Observed on real data: 0.0667 (var=0.0005)

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
P_REF = {'((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8544303797468354, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8515850144092219, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.8507135016465422, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.8616422947131609, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8434684684684685, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8585526315789473, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.18723404255319148, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.13909774436090225, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8309278350515464, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.84106463878327, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8507042253521127, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8477064220183487, '((0, 1, 0, 0), (1, 1, 1, 0))|0': 0.8584070796460177, '((0, 1, 0, 0), (1, 1, 1, 0))|1': 0.8328358208955224, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.1354625550660793, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.13452914798206278, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.1414048059149723, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.15550239234449761, '((0, 1, 1, 1), (1, 0, 1, 1))|0': 0.8586309523809523, '((0, 1, 1, 1), (1, 0, 1, 1))|1': 0.8572695035460993, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.8174442190669371, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.8569242540168325, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.1520935960591133, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.1417004048582996, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.1292817679558011, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.12960893854748604, '((1, 0, 0, 1), (1, 1, 0, 1))|0': 0.8486547085201793, '((1, 0, 0, 1), (1, 1, 0, 1))|1': 0.8458149779735683, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.17146974063400577, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.13471971066907776}
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

**Observed (real) value:** 0.1278 (var=0.0018)
**Predicted under pi_3:** 0.0545 (var=0.0004)
**Predicted under pi_6:** 0.0516 (var=0.0014)

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
P_REF = {'((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.15658362989323843, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.15976331360946747, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.85766092475068, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8407460545193687, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.14026602176541716, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.14285714285714285, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.1414496833216045, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.16358839050131926, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8337531486146096, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.852882703777336, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8533834586466166, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8027522935779816, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.13660179640718562, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1810344827586207, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.12374042724707779, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.13941018766756033, '((0, 0, 1, 1), (0, 1, 1, 0))|0': 0.5021645021645021, '((0, 0, 1, 1), (0, 1, 1, 0))|1': 0.4897260273972603, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8671428571428571, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.835, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.12324324324324325, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.136, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.1130820399113082, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.13870246085011187, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.1437837837837838, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.1382857142857143, '((1, 0, 1, 0), (0, 0, 1, 1))|0': 0.512396694214876, '((1, 0, 1, 0), (0, 0, 1, 1))|1': 0.5260196905766527}
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

**Observed (real) value:** 0.0115 (var=0.0003)
**Predicted under pi_3:** 0.0086 (var=0.0001)
**Predicted under pi_6:** 0.0070 (var=0.0017)

### Experiment 5
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
P_REF = {'((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.14212152420185376, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.1640530759951749, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.3063973063973064, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.2948294829482948, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12403100775193798, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.14327485380116958, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.7960526315789473, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.78125, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8595764272559853, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8515406162464986, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.8306451612903226, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8304721030042919, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.18655967903711135, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.22042341220423411, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.171875, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.18017241379310345, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.18838992332968238, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.16347237880496054, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8280542986425339, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.832014072119613, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8572727272727273, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8328571428571429, '((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.7511664074650077, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.7865168539325843, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.16783216783216784, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.19839679358717435, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.26578073089701, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.3070469798657718, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.2703984819734345, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.28820375335120646}
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

**Observed (real) value:** 0.0631 (var=0.0005)
**Predicted under pi_3:** 0.0013 (var=0.0001)
**Predicted under pi_6:** 0.0053 (var=0.0012)

### Experiment 6
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]

**Metric**
```python
P_REF = {'((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.5033185840707964, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.5044642857142857, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.5038335158817087, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4791431792559188, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.834625322997416, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8469785575048733, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8716773602199817, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.846262341325811, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.16143497757847533, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.1461100569259962, '((1, 1, 0, 0), (1, 0, 0, 1))|0': 0.5206991720331187, '((1, 1, 0, 0), (1, 0, 0, 1))|1': 0.4950911640953717, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5170842824601367, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.5032537960954447, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.5125, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.5021739130434782, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.16395939086294417, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1619631901840491, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.5163265306122449, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.5109756097560976, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8736383442265795, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8642803877703207, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8541666666666666, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8308080808080808, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.88470066518847, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8641425389755011, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.1368653421633554, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.13870246085011187, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.1446700507614213, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.1541501976284585}
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

**Observed (real) value:** 0.0422 (var=0.0006)
**Predicted under pi_3:** 0.0191 (var=0.0003)
**Predicted under pi_6:** 0.0143 (var=0.0012)

### Experiment 7
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
P_REF = {'((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8243512974051896, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8306389530408006, '((0, 0, 0, 1), (1, 0, 0, 1))|0': 0.75, '((0, 0, 0, 1), (1, 0, 0, 1))|1': 0.8394308943089431, '((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.20378457059679767, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.20035938903863432, '((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.831081081081081, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8326086956521739, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.8106508875739645, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8414539829853055, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8162650602409639, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.8257042253521126, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7980769230769231, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.864247311827957, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.15469613259668508, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.16089385474860335, '((0, 1, 1, 1), (0, 0, 1, 1))|0': 0.1354764638346728, '((0, 1, 1, 1), (0, 0, 1, 1))|1': 0.18945102260495156, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.16691068814055637, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.1378692927484333, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8472527472527472, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.848314606741573, '((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.863013698630137, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.8474025974025974, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.22109826589595374, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.18501805054151624, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.1837037037037037, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.16444444444444445, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.837573385518591, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.805439330543933}
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

**Observed (real) value:** 0.0322 (var=0.0002)
**Predicted under pi_3:** 0.0011 (var=0.0001)
**Predicted under pi_6:** 0.0022 (var=0.0004)

### Experiment 8
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
P_REF = {'((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.8792834890965732, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.8695436507936508, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8768115942028986, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.8729729729729729, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8824175824175824, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8910112359550562, '((1, 0, 0, 0), (0, 1, 0, 1))|0': 0.19534883720930232, '((1, 0, 0, 0), (0, 1, 0, 1))|1': 0.11798107255520504, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8701923076923077, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8706030150753769, '((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.1404833836858006, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.10896309314586995, '((1, 1, 0, 0), (0, 0, 1, 0))|0': 0.13004484304932734, '((1, 1, 0, 0), (0, 0, 1, 0))|1': 0.10635155096011817, '((1, 0, 1, 1), (1, 1, 1, 1))|0': 0.8691176470588236, '((1, 0, 1, 1), (1, 1, 1, 1))|1': 0.8901785714285714, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.8641425389755011, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.876940133037694, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.8669623059866962, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.8917716827279466, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.884828349944629, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8717948717948718, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.11346444780635401, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.12554872695346794, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8833151581243184, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8935447338618346, '((0, 0, 1, 0), (0, 1, 1, 0))|0': 0.8888888888888888, '((0, 0, 1, 0), (0, 1, 1, 0))|1': 0.8853333333333333, '((1, 0, 0, 0), (0, 0, 1, 0))|0': 0.1291759465478842, '((1, 0, 0, 0), (0, 0, 1, 0))|1': 0.12065136935603257}
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

**Observed (real) value:** 0.1498 (var=0.0008)
**Predicted under pi_3:** 0.0258 (var=0.0002)
**Predicted under pi_6:** 0.0364 (var=0.0009)

### Experiment 9
**Design**
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.14130434782608695, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.14798850574712644, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1374223602484472, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.16796875, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.1437389770723104, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.14114114114114115, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.7966101694915254, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8589440504334122, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.16150442477876106, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.16517857142857142, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.14174107142857142, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.1592920353982301, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8240223463687151, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8311808118081181, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.2047670639219935, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.20410490307867732, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8211382113821138, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8311688311688312, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8488888888888889, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8511111111111112, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.1305767138193689, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.13847900113507378, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.858440575321726, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8288100208768268, '((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.25467625899280577, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.23710407239819004, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.7866909753874203, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.7823613086770982, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.1379638439581351, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.17623497997329773, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8618290258449304, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8117408906882592}
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

**Observed (real) value:** 0.0797 (var=0.0006)
**Predicted under pi_3:** 0.0011 (var=0.0003)
**Predicted under pi_6:** 0.0072 (var=0.0014)

### Experiment 10
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.7179144385026738, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.7899239543726235, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.3987012987012987, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.525242718446602, '((1, 0, 1, 1), (0, 1, 1, 0))|0': 0.4, '((1, 0, 1, 1), (0, 1, 1, 0))|1': 0.4097826086956522, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.5949656750572082, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.58207343412527, '((0, 0, 1, 1), (0, 0, 1, 0))|0': 0.4528301886792453, '((0, 0, 1, 1), (0, 0, 1, 0))|1': 0.500945179584121, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.4343163538873995, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.4487666034155598, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.7379310344827587, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.7369565217391304, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.6849865951742627, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.6641366223908919, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.49841772151898733, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.4803082191780822, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.6206467661691543, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.6556224899598394, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.6007853403141361, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.6013513513513513, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.5683192261185006, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.5724563206577595, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.4384949348769899, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.37962128043282234, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.6468571428571429, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.6443243243243243, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.7374517374517374, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.7478005865102639, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.4662857142857143, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.46594594594594596}
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

**Observed (real) value:** 0.0803 (var=0.0011)
**Predicted under pi_3:** 0.0264 (var=0.0006)
**Predicted under pi_6:** 0.0196 (var=0.0012)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both Theory 1 (WADD) and Theory 2 (WADD + choice inertia) severely underpredict the observed Jensen-Shannon Divergence (JSD) across almost all experiments. When computing the sequence-aware JSD between the real human data and the model's reference probabilities, the observed values are consistently much higher (e.g., 0.06-0.15) than the models' self-predicted values (which hover near 0.001-0.03). This indicates that human choices systematically deviate from the Weighted Additive framework entirely. The failure of both models to capture this divergence suggests that simply adding a sequential bias to WADD is insufficient, and the core compensatory assumption of WADD itself is likely incorrect.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory based on a non-compensatory heuristic, such as the Take-The-Best (lexicographic) strategy. Instead of computing a weighted sum of all features, individuals might rank features by their explicit validities and compare options sequentially. They stop at the first feature that discriminates between the two options and choose the one with the positive feature. This would produce step-like, more deterministic choice patterns that could explain the large divergence from WADD-based probabilities seen in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 26173,
  "candidates_token_count": 300,
  "total_token_count": 27465
}
```
