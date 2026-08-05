# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Decision-makers evaluate options using a Weighted Additive (WADD) strategy, but their choices are subject to sequential dependencies, specifically choice inertia. The value of each option is computed as a weighted sum of its features, where weights correspond to cue validities. Additionally, a baseline bias (inertia) is added to the score of the option that was chosen in the immediately preceding trial. This reflects a psychological tendency to repeat previous actions (or avoid them, if inertia is negative). Choice probabilities are generated via a softmax function with an inverse temperature, along with a lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    # Incorporate choice inertia from the previous trial
    inertia = float(parameters["inertia"])
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
        if prev_resp == 0:
            score_a += inertia
        elif prev_resp == 1:
            score_b += inertia
            
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
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_6
Decision-makers use a boundedly rational strategy called Tallying with Validity-Ranked Tie-Breaking to minimize cognitive load while still making informed choices. They first evaluate options by simply counting the number of positive features for each (Tallying) and prefer the option with the highest tally. If the tallies are tied, they fall back to a non-compensatory tie-breaker, sequentially inspecting cues in descending order of their validity until one discriminates between the options. The resulting scores (tallies plus a potential tie-breaking bonus) are passed through a softmax function to generate choice probabilities, allowing for stochasticity, and a lapse rate is included to account for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    # Tally positive features
    score_a = np.sum(a)
    score_b = np.sum(b)
    
    # Validity-Ranked Tie-Breaking
    if score_a == score_b:
        val = np.asarray(parameters["validities"], dtype=float)
        cue_order = np.argsort(-val, kind="stable").tolist()
        tie_bonus = float(parameters["tie_bonus"])
        
        for j in cue_order:
            if a[j] > b[j]:
                score_a += tie_bonus
                break
            elif b[j] > a[j]:
                score_b += tie_bonus
                break
                
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
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.6, 0.76, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0]  B=[1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 11: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 13: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 14: A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 16: A=[0, 0, 1, 1]  B=[0, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 1, 0), (0, 0, 0, 0))|0': 0.42424242424242425, '((0, 1, 1, 0), (0, 0, 0, 0))|1': 0.4749034749034749, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.43899591360186807, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.444234404536862, '((1, 1, 1, 0), (1, 1, 0, 0))|0': 0.4432882414151925, '((1, 1, 1, 0), (1, 1, 0, 0))|1': 0.43146603098927294, '((1, 0, 1, 1), (1, 1, 0, 1))|0': 0.5069667738478028, '((1, 0, 1, 1), (1, 1, 0, 1))|1': 0.4925028835063437, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.44123711340206184, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.5036144578313253, '((0, 1, 0, 1), (1, 0, 0, 0))|0': 0.44263959390862945, '((0, 1, 0, 1), (1, 0, 0, 0))|1': 0.47607361963190187, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.4747368421052632, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.47411764705882353, '((1, 0, 0, 0), (0, 1, 1, 0))|0': 0.48083242059145676, '((1, 0, 0, 0), (0, 1, 1, 0))|1': 0.4971815107102593, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.5044943820224719, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.5274725274725275, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.48688352570828963, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.500590318772137, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.4408033826638478, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.4894613583138173, '((1, 1, 1, 1), (0, 1, 1, 1))|0': 0.47489539748953974, '((1, 1, 1, 1), (0, 1, 1, 1))|1': 0.4976303317535545, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.5227397260273973, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.5352112676056338, '((0, 1, 1, 1), (0, 1, 0, 1))|0': 0.4944690265486726, '((0, 1, 1, 1), (0, 1, 0, 1))|1': 0.49776785714285715}
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
- Predicted under pi_4 (simulated): 0.0025 (var=0.0016)
- Predicted under pi_6 (simulated): 0.0672 (var=0.0018)
- Observed on real data: 0.2152 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.81, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 2: A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 7: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 8: A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  trial 9: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 10: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 11: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 12: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 13: A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 14: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 15: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 16: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.8413391557496361, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8607367475292004, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.13802816901408452, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.12935779816513762, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.1646153846153846, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.15058823529411763, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8609865470852018, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8678414096916299, '((1, 0, 1, 0), (1, 1, 1, 0))|0': 0.7777777777777778, '((1, 0, 1, 0), (1, 1, 1, 0))|1': 0.8520671834625323, '((0, 0, 1, 0), (1, 0, 0, 1))|0': 0.8752244165170556, '((0, 0, 1, 0), (1, 0, 0, 1))|1': 0.8688046647230321, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.13402061855670103, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.1462979482604817, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.15709969788519637, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.17223198594024605, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8460388639760837, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.8390804597701149, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.8686131386861314, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.8547085201793722, '((0, 0, 0, 0), (1, 0, 1, 0))|0': 0.8700696055684455, '((0, 0, 0, 0), (1, 0, 1, 0))|1': 0.8685171658144631, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.8577777777777778, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8718518518518519, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.14989293361884368, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.13953488372093023, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.8082788671023965, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.8598061148396718, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8808743169398907, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8711864406779661, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.8562753036437247, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.8483920367534457}
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
- Predicted under pi_4 (simulated): 0.0771 (var=0.0022)
- Predicted under pi_6 (simulated): 0.0010 (var=0.0002)
- Observed on real data: 0.2652 (var=0.0067)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.126044039483675, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.15734989648033126, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8735119047619048, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8728070175438597, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.16129032258064516, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.1453940066592675, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.85997171145686, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.8664226898444648, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8376068376068376, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8716216216216216, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.8649262202043133, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8324265505984766, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.16248153618906944, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.12911843276936777, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.1632047477744807, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1483126110124334, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.13676731793960922, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.13501483679525222, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.12455516014234876, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.15680473372781065, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8619676945668135, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8543342269883825, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8610855565777369, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8401122019635343, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8388520971302428, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8893838158871566, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.14516129032258066, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.1252236135957066, '((1, 1, 0, 1), (0, 1, 1, 0))|0': 0.13047445255474452, '((1, 1, 0, 1), (0, 1, 1, 0))|1': 0.15346534653465346, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8712328767123287, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8567375886524823}
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

**Observed (real) value:** 0.2658 (var=0.0000)
**Predicted under pi_4:** 0.0793 (var=0.0023)
**Predicted under pi_6:** 0.0013 (var=0.0001)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 1), (0, 1, 1, 1))|0': 0.862453531598513, '((0, 0, 0, 1), (0, 1, 1, 1))|1': 0.8751258811681772, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.47289156626506024, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.4876760563380282, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.13242574257425743, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.14818548387096775, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8419282511210763, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8601321585903083, '((0, 1, 1, 0), (0, 0, 0, 0))|0': 0.14113785557986872, '((0, 1, 1, 0), (0, 0, 0, 0))|1': 0.14672686230248308, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.15040183696900114, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.1496232508073197, '((0, 0, 0, 0), (0, 1, 1, 0))|0': 0.8631346578366446, '((0, 0, 0, 0), (0, 1, 1, 0))|1': 0.8534675615212528, '((0, 1, 0, 1), (1, 1, 1, 1))|0': 0.8433179723502304, '((0, 1, 0, 1), (1, 1, 1, 1))|1': 0.8630393996247655, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.4905814905814906, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.5077720207253886, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.13493377483443708, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.13682432432432431, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1644325290437891, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.17033773861967694, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.14156626506024098, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.14925373134328357, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8377483443708609, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8534675615212528, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8529048207663782, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8425832492431887, '((0, 0, 0, 0), (1, 0, 1, 1))|0': 0.8533627342888643, '((0, 0, 0, 0), (1, 0, 1, 1))|1': 0.8779395296752519, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.14985163204747776, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.15008880994671403}
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

**Observed (real) value:** 0.2554 (var=0.0000)
**Predicted under pi_4:** 0.0562 (var=0.0022)
**Predicted under pi_6:** 0.0141 (var=0.0002)

### Experiment 5
**Design**
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8241525423728814, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.8621987951807228, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.1432408236347359, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.1683748169838946, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.1524024024024024, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.17094017094017094, '((0, 1, 0, 0), (0, 0, 1, 0))|0': 0.13938053097345132, '((0, 1, 0, 0), (0, 0, 1, 0))|1': 0.14397321428571427, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.1352154531946508, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.16519823788546256, '((0, 1, 1, 0), (0, 1, 1, 1))|0': 0.831140350877193, '((0, 1, 1, 0), (0, 1, 1, 1))|1': 0.847972972972973, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8285302593659942, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8318264014466547, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8353863381858903, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8412348401323043, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.18046709129511676, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.13393528969149737, '((0, 1, 1, 1), (1, 1, 1, 1))|0': 0.8490909090909091, '((0, 1, 1, 1), (1, 1, 1, 1))|1': 0.8514285714285714, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.14977973568281938, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.16704035874439463, '((0, 0, 0, 0), (0, 1, 1, 1))|0': 0.8558382257012394, '((0, 0, 0, 0), (0, 1, 1, 1))|1': 0.8239700374531835, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.14167294649585532, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.17758985200845667, '((0, 0, 1, 0), (0, 0, 0, 0))|0': 0.16057585825027684, '((0, 0, 1, 0), (0, 0, 0, 0))|1': 0.13154960981047936, '((0, 1, 1, 1), (0, 1, 0, 0))|0': 0.16817359855334538, '((0, 1, 1, 1), (0, 1, 0, 0))|1': 0.16714697406340057, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8523076923076923, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8458823529411764}
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

**Observed (real) value:** 0.2601 (var=0.0008)
**Predicted under pi_4:** 0.0660 (var=0.0021)
**Predicted under pi_6:** 0.0174 (var=0.0001)

### Experiment 6
**Design**
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.13957759412304868, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.180028129395218, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8734723220704529, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8337408312958435, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.14118629908103592, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.15754560530679934, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8639705882352942, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8455056179775281, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8453038674033149, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8277310924369747, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.16892502258355918, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.19336219336219337, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.14307342922028765, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.18997912317327767, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.1967020023557126, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.23809523809523808, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.130348913759052, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.2491103202846975, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1337099811676083, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.14769647696476965, '((1, 1, 1, 1), (1, 0, 1, 0))|0': 0.13020833333333334, '((1, 1, 1, 1), (1, 0, 1, 0))|1': 0.15432098765432098, '((0, 1, 1, 0), (1, 1, 0, 0))|0': 0.6285452881976212, '((0, 1, 1, 0), (1, 1, 0, 0))|1': 0.6025459688826026, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8790072388831437, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.885954381752701, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.1439312567132116, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.11392405063291139, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.1753902662993572, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.19127988748241911, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.13533284564740308, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.18475750577367206}
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

**Observed (real) value:** 0.2320 (var=0.0050)
**Predicted under pi_4:** 0.0609 (var=0.0026)
**Predicted under pi_6:** 0.0035 (var=0.0002)

### Experiment 7
**Design**
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0), (0, 1, 0, 0))|0': 0.5198836081474297, '((1, 0, 1, 0), (0, 1, 0, 0))|1': 0.47074122236671, '((1, 0, 1, 1), (0, 1, 0, 0))|0': 0.467966573816156, '((1, 0, 1, 1), (0, 1, 0, 0))|1': 0.4332409972299169, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.4904522613065327, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.4360248447204969, '((1, 1, 0, 1), (0, 1, 0, 0))|0': 0.495850622406639, '((1, 1, 0, 1), (0, 1, 0, 0))|1': 0.45095693779904306, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.49638802889576883, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.42719614921780985, '((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.5505154639175258, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.4614457831325301, '((1, 1, 1, 1), (0, 1, 0, 0))|0': 0.47426067907995617, '((1, 1, 1, 1), (0, 1, 0, 0))|1': 0.42728297632469, '((0, 0, 1, 0), (1, 1, 0, 0))|0': 0.4968421052631579, '((0, 0, 1, 0), (1, 1, 0, 0))|1': 0.4623529411764706, '((0, 1, 0, 0), (1, 0, 1, 0))|0': 0.5115529285330468, '((0, 1, 0, 0), (1, 0, 1, 0))|1': 0.4473835537665325, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.5056746532156369, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.4512022630834512, '((1, 1, 1, 1), (1, 0, 1, 1))|0': 0.5224839400428265, '((1, 1, 1, 1), (1, 0, 1, 1))|1': 0.45958429561200925, '((0, 0, 1, 0), (0, 1, 0, 0))|0': 0.5144385026737968, '((0, 0, 1, 0), (0, 1, 0, 0))|1': 0.46936416184971097, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.5223068552774756, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.4699205448354143, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.49892008639308855, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.4473684210526316}
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

**Observed (real) value:** 0.2336 (var=0.0002)
**Predicted under pi_4:** 0.0007 (var=0.0016)
**Predicted under pi_6:** 0.0795 (var=0.0011)

### Experiment 8
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.17132867132867133, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.15309446254071662, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.6379928315412187, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.6723027375201288, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.7880794701986755, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.81438127090301, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.7609427609427609, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.7769485903814262, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.7936210131332082, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8484609313338595, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.797427652733119, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8488918737407656, '((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.6291666666666667, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.6820512820512821, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8179775280898877, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8435424354243543, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.19402985074626866, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.16729559748427672, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7983193277310925, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8370165745856354, '((1, 1, 1, 1), (0, 1, 1, 1))|0': 0.1960352422907489, '((1, 1, 1, 1), (0, 1, 1, 1))|1': 0.14635958395245172, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.6612377850162866, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.6757679180887372, '((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.7945544554455446, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.8359598853868195, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.6296296296296297, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.6730158730158731, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.834920634920635, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8589743589743589, '((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8554913294797688, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8501805054151624}
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

**Observed (real) value:** 0.2239 (var=0.0074)
**Predicted under pi_4:** 0.0494 (var=0.0023)
**Predicted under pi_6:** 0.0122 (var=0.0001)

### Experiment 9
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.5142296368989205, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.5096030729833547, '((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.4219977553310887, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.41804180418041803, '((1, 1, 0, 0), (1, 0, 1, 0))|0': 0.5243781094527363, '((1, 1, 0, 0), (1, 0, 1, 0))|1': 0.5006289308176101, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.5032894736842105, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.5168918918918919, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.5138461538461538, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.5357575757575758, '((0, 0, 0, 0), (1, 1, 1, 1))|0': 0.502262443438914, '((0, 0, 0, 0), (1, 1, 1, 1))|1': 0.5436681222707423, '((1, 1, 0, 0), (1, 0, 0, 0))|0': 0.4811529933481153, '((1, 1, 0, 0), (1, 0, 0, 0))|1': 0.47216035634743875, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.5050847457627119, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.5049180327868853, '((0, 0, 0, 1), (1, 0, 0, 1))|0': 0.5043196544276458, '((0, 0, 0, 1), (1, 0, 0, 1))|1': 0.5137299771167048, '((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.48259860788863107, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.47761194029850745, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.43991416309012876, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.47119815668202764, '((1, 1, 0, 0), (0, 1, 0, 1))|0': 0.4946107784431138, '((1, 1, 0, 0), (0, 1, 0, 1))|1': 0.4736196319018405, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.5406852248394004, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.5311778290993071, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.4575892857142857, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.47013274336283184, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.486784140969163, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.4663677130044843}
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

**Observed (real) value:** 0.2150 (var=0.0000)
**Predicted under pi_4:** 0.0015 (var=0.0011)
**Predicted under pi_6:** 0.0660 (var=0.0015)

### Experiment 10
**Design**
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.4529991047448523, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.4480234260614934, '((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.4255555555555556, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.35777777777777775, '((0, 0, 1, 0), (1, 1, 1, 1))|0': 0.5586538461538462, '((0, 0, 1, 0), (1, 1, 1, 1))|1': 0.4789473684210526, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.42639040348964014, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.37712344280860705, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.4421768707482993, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.4254215304798962, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.43743536711478803, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.42016806722689076, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.44195953141640043, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.3879210220673635, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.578544061302682, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.4894179894179894, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.464746772591857, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.42370744010088274, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.5597548518896833, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.4774665042630938, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.5431034482758621, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.5045871559633027, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.44678609062170704, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.44418331374853115, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.5573604060913706, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.47116564417177914, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.4754601226993865, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.4209245742092457, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.4399164054336468, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.40569395017793597, '((1, 1, 1, 0), (1, 0, 1, 1))|0': 0.4461942257217848, '((1, 1, 1, 0), (1, 0, 1, 1))|1': 0.3902439024390244}
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

**Observed (real) value:** 0.2221 (var=0.0005)
**Predicted under pi_4:** 0.0012 (var=0.0015)
**Predicted under pi_6:** 0.0654 (var=0.0009)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both Theory 1 (Weighted Additive with inertia) and Theory 2 (Tallying with Validity-Ranked Tie-Breaking) completely fail to capture the observed data. Across all 10 experiments, the observed JSD metrics on real data consistently fall in the 0.21 to 0.26 range. In contrast, when simulating predictions under either theory, the metrics are near zero for their own proposed experiments and below 0.08 for the other's, indicating that the simulated data from these models looks fundamentally different from human data. The real subjects are demonstrating choice profiles that neither the compensatory WADD model nor the boundedly rational Tallying model can reproduce, suggesting both theories are degenerate and a fundamentally different cognitive mechanism is at play.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a brand-new theory based on a purely non-compensatory Lexicographic strategy, such as Take-The-Best (TTB). In this theory, decision-makers do not integrate all cues (unlike WADD) nor do they tally them (unlike Theory 2). Instead, they search through the features strictly in descending order of cue validity. They stop and make a choice at the very first cue that discriminates between the two options. If no cues discriminate, they guess randomly. This introduces a different kind of bounded rationality that might better match the stochasticity and systematic biases seen in the human data."
}
```

## Usage

```json
{
  "prompt_token_count": 26926,
  "candidates_token_count": 325,
  "total_token_count": 28143
}
```
