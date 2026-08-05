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
Decision makers exhibit strategy variability, probabilistic alternating between a simple heuristic and a compensatory strategy. Specifically, subjects use a Strategy Mixture of Tallying (counting the number of positive cues for each option, ignoring cue validities) and Weighted Additive (WADD, computing a compensatory score based on log-odds of cue validities). This mixture allows the model to capture both fast, unweighted evidence accumulation and more deliberative, validity-weighted integration on different trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    # WADD component
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_mix = float(parameters["w_mix"])
    
    # Tallying probabilities
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # WADD probabilities
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture
    p_core = w_mix * p_tally + (1.0 - w_mix) * p_wadd
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_6
Take-The-Best with Validity-Scaled Confidence and Choice Inertia: Decision makers use a fast-and-frugal lexicographic heuristic, comparing options sequentially on features ordered by their stated validities. They stop at the first feature that discriminates between the options. However, the confidence (and therefore determinism) of their choice scales with the validity of the cue that resolved the decision. The final choice is also subject to an autoregressive motor/spatial bias (inertia) from the immediately preceding trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    scores = np.zeros(2)
    
    # Take-The-Best heuristic: find the first discriminating cue
    for j in cue_order:
        if a[j] > b[j]:
            scores[0] += val[j]
            break
        if b[j] > a[j]:
            scores[1] += val[j]
            break
            
    # Add choice inertia from the previous trial
    if history and "response" in history and len(history["response"]) > 0:
        last_choice = int(history["response"][-1])
        inertia = float(parameters["inertia"])
        scores[last_choice] += inertia
        
    # Softmax and lapse
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.63, 0.6, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 9: A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 10: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 11: A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 13: A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 15: A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 0, 0), (1, 1, 1, 1))|0': 0.8704453441295547, '((0, 0, 0, 0), (1, 1, 1, 1))|1': 0.8706326723323891, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.15, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.13137254901960785, '((0, 1, 0, 0), (0, 0, 1, 0))|0': 0.39369158878504673, '((0, 1, 0, 0), (0, 0, 1, 0))|1': 0.4110169491525424, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.4532304725168756, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.528178243774574, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.43133802816901406, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.5016233766233766, '((1, 0, 1, 1), (1, 1, 0, 1))|0': 0.5971638655462185, '((1, 0, 1, 1), (1, 1, 0, 1))|1': 0.5996462264150944, '((1, 1, 0, 1), (0, 0, 0, 1))|0': 0.13682432432432431, '((1, 1, 0, 1), (0, 0, 0, 1))|1': 0.12665562913907286, '((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8488745980707395, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.8624787775891342, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.16997792494481237, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.10888252148997135, '((0, 0, 1, 0), (0, 1, 0, 1))|0': 0.8286252354048964, '((0, 0, 1, 0), (0, 1, 0, 1))|1': 0.8238482384823849, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8407445708376422, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8523409363745498, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.8274760383386581, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8574310692669805, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.6139705882352942, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.6199186991869918, '((0, 0, 1, 0), (1, 0, 0, 1))|0': 0.8719298245614036, '((0, 0, 1, 0), (1, 0, 0, 1))|1': 0.8585365853658536, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.7010135135135135, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.6788079470198676}
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
- Predicted under pi_4 (simulated): 0.0013 (var=0.0002)
- Predicted under pi_6 (simulated): 0.0263 (var=0.0031)
- Observed on real data: 0.2411 (var=0.0027)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.74, 0.64, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 9: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 10: A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 11: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 12: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 14: A=[1, 0, 0, 1]  B=[1, 1, 1, 1]
  trial 15: A=[0, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.5417085427135678, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.5167701863354037, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.5420765027322404, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.48926553672316386, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.4376899696048632, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.4034440344403444, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.5974534769833496, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.47753530166880614, '((1, 1, 0, 0), (1, 0, 0, 0))|0': 0.4877750611246944, '((1, 1, 0, 0), (1, 0, 0, 0))|1': 0.44281524926686217, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.38907103825136613, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.40790960451977404, '((1, 0, 0, 1), (1, 1, 1, 1))|0': 0.581140350877193, '((1, 0, 0, 1), (1, 1, 1, 1))|1': 0.49324324324324326, '((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.5845824411134903, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.5484988452655889, '((1, 0, 0, 0), (0, 0, 1, 1))|0': 0.4152637485970819, '((1, 0, 0, 0), (0, 0, 1, 1))|1': 0.3872387238723872, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.4424686192468619, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.42298578199052134, '((1, 1, 1, 1), (0, 1, 0, 0))|0': 0.4067622950819672, '((1, 1, 1, 1), (0, 1, 0, 0))|1': 0.39805825242718446, '((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.5773874862788145, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.5331833520809899, '((0, 1, 1, 1), (0, 0, 0, 1))|0': 0.4966216216216216, '((0, 1, 1, 1), (0, 0, 0, 1))|1': 0.4166666666666667, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.4881720430107527, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.4264367816091954, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.5672645739910314, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.5, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.4263565891472868, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.4427083333333333}
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
- Predicted under pi_4 (simulated): 0.0598 (var=0.0007)
- Predicted under pi_6 (simulated): 0.0007 (var=0.0015)
- Observed on real data: 0.2314 (var=0.0001)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.14532374100719425, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.14027149321266968, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.14891304347826087, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.14204545454545456, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8696969696969697, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8291666666666667, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8757575757575757, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8614035087719298, '((1, 0, 0, 1), (1, 1, 1, 0))|0': 0.8546255506607929, '((1, 0, 0, 1), (1, 1, 1, 0))|1': 0.8430493273542601, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8666666666666667, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8621621621621621, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.17159763313609466, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.12529002320185614, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.1332547169811321, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.1441717791411043, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.7886710239651417, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8806860551826995, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.854389721627409, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.873218304576144, '((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8542825361512792, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.855715871254162, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8677007299270073, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8480113636363636, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.14301929625425652, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.1588683351468988, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.1291053227633069, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.15485278080697928, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.856043956043956, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8382022471910112, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.16374269005847952, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.12724014336917563}
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

**Observed (real) value:** 0.2689 (var=0.0008)
**Predicted under pi_4:** 0.0198 (var=0.0002)
**Predicted under pi_6:** 0.0711 (var=0.0029)

### Experiment 4
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.8162162162162162, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8586345381526105, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.832, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8754838709677419, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.1724137931034483, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.14285714285714285, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.8335483870967741, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.855609756097561, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.8704883227176221, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.871331828442438, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8712871287128713, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8636779505946935, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.842031029619182, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.846929422548121, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.18151815181518152, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.1507537688442211, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8575539568345324, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8597285067873304, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.18433179723502305, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.1575984990619137, '((0, 0, 0, 0), (1, 0, 0, 0))|0': 0.8415094339622642, '((0, 0, 0, 0), (1, 0, 0, 0))|1': 0.8497512437810946, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.84, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.8457142857142858, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.48918640576725025, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.5271411338962606, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.5010482180293501, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4837490551776266, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.8520710059171598, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.8567615658362989, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7663230240549829, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8436050364479788}
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

**Observed (real) value:** 0.2318 (var=0.0107)
**Predicted under pi_4:** 0.0024 (var=0.0001)
**Predicted under pi_6:** 0.0533 (var=0.0042)

### Experiment 5
**Design**
  A=[0, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.14613778705636743, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.1445874337623013, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.14887794198139026, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.1314156796390299, '((0, 1, 0, 1), (0, 1, 1, 1))|0': 0.8539944903581267, '((0, 1, 0, 1), (0, 1, 1, 1))|1': 0.8579465541490858, '((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.8591117917304747, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.8559622195985832, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.847084708470847, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.8653198653198653, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8435277382645804, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8505013673655424, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.16152716593245228, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.13047363717605004, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.8445901639344262, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.7781818181818182, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8793103448275862, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.865036231884058, '((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8561253561253561, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8715846994535519, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.15529753265602322, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.1422142214221422, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.15214180206794684, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.15672306322350846, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.8633257403189066, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.8322981366459627, '((0, 0, 1, 0), (1, 0, 0, 1))|0': 0.8514705882352941, '((0, 0, 1, 0), (1, 0, 0, 1))|1': 0.8321428571428572, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.16770186335403728, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.1404707668944571}
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

**Observed (real) value:** 0.2723 (var=0.0005)
**Predicted under pi_4:** 0.0210 (var=0.0006)
**Predicted under pi_6:** 0.0578 (var=0.0041)

### Experiment 6
**Design**
  A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8877937831690674, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.8253638253638254, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.11549295774647887, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.12018348623853212, '((1, 1, 0, 0), (1, 1, 0, 1))|0': 0.763235294117647, '((1, 1, 0, 0), (1, 1, 0, 1))|1': 0.7227272727272728, '((1, 0, 1, 0), (1, 0, 1, 1))|0': 0.7636761487964989, '((1, 0, 1, 0), (1, 0, 1, 1))|1': 0.7787810383747178, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.13499480789200416, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.14814814814814814, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.28645383951682485, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.31045241809672386, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8513853904282116, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8697813121272365, '((1, 1, 0, 1), (0, 0, 0, 1))|0': 0.10682226211849193, '((1, 1, 0, 1), (0, 0, 0, 1))|1': 0.13994169096209913, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.1196319018404908, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.15328467153284672, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.13957446808510637, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.1376, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.6856780735107731, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.675568743818002, '((0, 0, 0, 1), (0, 1, 0, 1))|0': 0.8723897911832946, '((0, 0, 0, 1), (0, 1, 0, 1))|1': 0.8284023668639053, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.12698412698412698, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.11929824561403508, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.1417142857142857, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.1408, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8236559139784946, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8091954022988506, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.14899328859060404, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.12985781990521328}
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

**Observed (real) value:** 0.2460 (var=0.0012)
**Predicted under pi_4:** 0.0093 (var=0.0004)
**Predicted under pi_6:** 0.0549 (var=0.0036)

### Experiment 7
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0), (1, 1, 1, 1))|0': 0.848851269649335, '((1, 1, 0, 0), (1, 1, 1, 1))|1': 0.8756423432682425, '((1, 1, 0, 1), (1, 0, 0, 1))|0': 0.1488933601609658, '((1, 1, 0, 1), (1, 0, 0, 1))|1': 0.15272448196469685, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.2138728323699422, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.20306859205776173, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.14391143911439114, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.15363128491620112, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.14601769911504425, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.19402985074626866, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8621908127208481, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8532934131736527, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.6896551724137931, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.6927860696517413, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.6811797752808989, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.6865808823529411, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.13060179257362356, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.13542688910696762, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.8680811808118081, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.8533519553072626, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.12791702679343128, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.16174183514774496, '((0, 1, 0, 0), (1, 1, 0, 0))|0': 0.8446215139442231, '((0, 1, 0, 0), (1, 1, 0, 0))|1': 0.8615090735434575, '((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.8611599297012302, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.8610271903323263, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8228346456692913, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8421733505821475, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.15481171548117154, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.132375189107413, '((1, 0, 0, 0), (1, 0, 1, 0))|0': 0.8642086330935251, '((1, 0, 0, 0), (1, 0, 1, 0))|1': 0.8473837209302325}
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

**Observed (real) value:** 0.2611 (var=0.0002)
**Predicted under pi_4:** 0.0013 (var=0.0001)
**Predicted under pi_6:** 0.0580 (var=0.0029)

### Experiment 8
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8738738738738738, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.882494004796163, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.8158347676419966, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.8072100313479624, '((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.15524475524475526, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.1631336405529954, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.8671988388969522, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.8364928909952607, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.10869565217391304, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.13793103448275862, '((1, 1, 0, 0), (0, 0, 0, 0))|0': 0.12197686645636173, '((1, 1, 0, 0), (0, 0, 0, 0))|1': 0.127208480565371, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.11345454545454546, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.13176470588235295, '((1, 1, 1, 0), (1, 0, 1, 1))|0': 0.14825174825174825, '((1, 1, 1, 0), (1, 0, 1, 1))|1': 0.12442396313364056, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.12267657992565056, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.12747252747252746, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.11588921282798834, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.14719626168224298, '((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.7353448275862069, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.7328125, '((0, 0, 1, 1), (0, 1, 0, 1))|0': 0.8229461756373938, '((0, 0, 1, 1), (0, 1, 0, 1))|1': 0.8071297989031079, '((1, 0, 1, 0), (1, 1, 0, 1))|0': 0.8168044077134986, '((1, 0, 1, 0), (1, 1, 0, 1))|1': 0.8417132216014898, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.13426423200859292, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.13003452243958574, '((1, 0, 0, 1), (1, 1, 0, 0))|0': 0.8900343642611683, '((1, 0, 0, 1), (1, 1, 0, 0))|1': 0.8805031446540881, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12517193947730398, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.10624417520969245}
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

**Observed (real) value:** 0.2658 (var=0.0017)
**Predicted under pi_4:** 0.0123 (var=0.0003)
**Predicted under pi_6:** 0.0568 (var=0.0056)

### Experiment 9
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.5161637931034483, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.5240825688073395, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.36378205128205127, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.31037414965986393, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.6975023126734505, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.7009735744089013, '((0, 1, 0, 1), (0, 1, 1, 0))|0': 0.6290516206482593, '((0, 1, 0, 1), (0, 1, 1, 0))|1': 0.6380558428128231, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.14984709480122324, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.14184397163120568, '((0, 0, 1, 0), (0, 1, 0, 1))|0': 0.838405036726128, '((0, 0, 1, 0), (0, 1, 0, 1))|1': 0.8642266824085005, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.6720085470085471, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.6712962962962963, '((1, 1, 1, 1), (0, 1, 1, 0))|0': 0.13793103448275862, '((1, 1, 1, 1), (0, 1, 1, 0))|1': 0.1504297994269341, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.1532258064516129, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.16559485530546625, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8964757709251101, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8688340807174888, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.6379310344827587, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.687793427230047, '((1, 1, 0, 1), (1, 0, 0, 1))|0': 0.13604378420641125, '((1, 1, 0, 1), (1, 0, 0, 1))|1': 0.16314779270633398, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.12582014666152064, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.14469772051536176, '((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.8667239896818573, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.8445839874411303}
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

**Observed (real) value:** 0.2391 (var=0.0008)
**Predicted under pi_4:** 0.0010 (var=0.0001)
**Predicted under pi_6:** 0.0431 (var=0.0030)

### Experiment 10
**Design**
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.3399103139013453, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.37664233576642336, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.45098039215686275, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.5226757369614512, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.5025536261491318, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.6321559074299634, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.4888211382113821, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.571078431372549, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.484548825710754, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.627648839556004, '((1, 0, 0, 0), (0, 1, 1, 0))|0': 0.3731527093596059, '((1, 0, 0, 0), (0, 1, 1, 0))|1': 0.4524291497975709, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.4384525205158265, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.5068637803590285, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.5042462845010616, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.5641025641025641, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.4496487119437939, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.48731501057082455, '((0, 1, 0, 1), (1, 1, 0, 1))|0': 0.662020905923345, '((0, 1, 0, 1), (1, 1, 0, 1))|1': 0.7273695420660277, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.4547770700636943, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.47783251231527096, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.63125, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.6452380952380953, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.5013368983957219, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.6387832699619772, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.30851063829787234, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.3018867924528302, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.4416167664670659, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.4807692307692308, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.5912653975363942, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.659316427783903}
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

**Observed (real) value:** 0.2027 (var=0.0000)
**Predicted under pi_4:** 0.0294 (var=0.0006)
**Predicted under pi_6:** 0.0089 (var=0.0014)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both pi_4 (Strategy Mixture) and pi_6 (Take-The-Best with Inertia) fail to adequately capture the observed real data across all experiments. This is evidenced by the high Jensen-Shannon Divergence (JSD) values (ranging from ~0.20 to ~0.27) when comparing the real data to the reference distributions generated by the models. The simulated predictions for the proposing models are consistently near zero (as expected), but the real data diverges significantly from both. This indicates that neither the WADD/Tallying mixture nor the deterministic lexicographic heuristic with inertia accurately reflects human decision-making in this task.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory based on an Additive Difference Model with Probabilistic Feature Comparison. Instead of calculating absolute values for each option (as in WADD) or strictly applying a lexicographic stopping rule (as in TTB), assume decision-makers compare options attribute-by-attribute. The differences on each attribute are weighted non-linearly by their stated validities and accumulated into a relative evidence signal. Introduce a stochastic element where attention to features is proportional to their validities, and use a softmax function over the accumulated differences with a lapse rate to account for baseline noise."
}
```

## Usage

```json
{
  "prompt_token_count": 27008,
  "candidates_token_count": 290,
  "total_token_count": 28802
}
```
