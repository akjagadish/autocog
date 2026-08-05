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
Take-The-Best (Lexicographic) Heuristic: Decision-makers evaluate features sequentially in order of decreasing cue validity. The first feature that discriminates between the two options strictly determines the preference, and all remaining lower-validity features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory strategy is formalized by assigning a binary score to the winning option based on the first discriminating cue, which is then mapped to choice probabilities via a softmax function (to capture decision noise) and a uniform lapse rate (to capture random errors).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Take-The-Best expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    # stable sort ensures consistent tie-breaking for equal validities
    cue_order = np.argsort(-validities, kind="stable")
    
    score_a = 0.0
    score_b = 0.0
    
    # Evaluate cues sequentially
    for idx in cue_order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break

    scores = np.array([score_a, score_b])
    
    # If no cues discriminate, guess randomly
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
People integrate information across all features by weighting each feature by its subjective validity. The Weighted Additive (WADD) model computes an overall value for each option by summing the products of the feature values and their corresponding cue validities. This mechanism allows for compensatory decision-making, where multiple weak cues can collectively override a single strong cue, unlike non-compensatory heuristics (e.g., Take The Best) or unweighted integration (e.g., Tallying). The resulting option values are transformed into choice probabilities via a softmax function, with an independent lapse rate to capture random guessing or attentional errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate WADD scores by weighting each feature by its validity
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.95, 0.72, 0.67, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 2: A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 12: A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 13: A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 15: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 16: A=[1, 0, 0, 0]  B=[1, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.16690442225392296, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.1492265696087352, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8384728340675477, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8489722966934763, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.10912343470483005, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.1495601173020528, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.8663101604278075, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8672566371681416, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.18376550169109357, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.14895947426067907, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8370786516853933, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8252747252747252, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.14014598540145987, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.14977578475336323, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.15948275862068967, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.13677536231884058, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8485401459854015, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.84375, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.8449612403100775, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.8315789473684211, '((1, 0, 0, 0), (0, 1, 1, 0))|0': 0.1412894375857339, '((1, 0, 0, 0), (0, 1, 1, 0))|1': 0.15779645191409897, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8469601677148847, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8669690098261527, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8352941176470589, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8660714285714286, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.8468271334792122, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.8419864559819413, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.17982456140350878, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.1388888888888889, '((0, 1, 0, 1), (1, 1, 0, 1))|0': 0.8676337262012692, '((0, 1, 0, 1), (1, 1, 0, 1))|1': 0.8436154949784792}
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
- Predicted under pi_5 (simulated): 0.0011 (var=0.0002)
- Predicted under pi_3 (simulated): 0.0471 (var=0.0003)
- Observed on real data: 0.0640 (var=0.0007)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.84, 0.8, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1]  B=[0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 8: A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  trial 9: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 10: A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 12: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 13: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 14: A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 15: A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 16: A=[0, 0, 1, 1]  B=[1, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.3313343328335832, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.2965578111209179, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.4525462962962963, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.3856837606837607, '((1, 0, 1, 1), (1, 1, 1, 1))|0': 0.8444444444444444, '((1, 0, 1, 1), (1, 1, 1, 1))|1': 0.8493827160493828, '((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8880208333333334, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8488372093023255, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8564417177914111, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8700507614213198, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.15994236311239193, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.1518987341772152, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8163841807909604, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.8498168498168498, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8389154704944178, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8499573742540495, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8468606431852986, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8500435919790759, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8858773181169758, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8771610555050046, '((0, 0, 0, 0), (1, 1, 1, 1))|0': 0.8677517802644964, '((0, 0, 0, 0), (1, 1, 1, 1))|1': 0.8494492044063647, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8380952380952381, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8542635658914729, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.12440944881889764, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.1339622641509434, '((0, 1, 1, 1), (0, 1, 0, 0))|0': 0.11645569620253164, '((0, 1, 1, 1), (0, 1, 0, 0))|1': 0.14796747967479676, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.13561190738699008, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.12541993281075028, '((1, 0, 0, 1), (0, 0, 0, 1))|0': 0.16666666666666666, '((1, 0, 0, 1), (0, 0, 0, 1))|1': 0.1431818181818182}
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
- Predicted under pi_5 (simulated): 0.0053 (var=0.0004)
- Predicted under pi_3 (simulated): 0.0014 (var=0.0001)
- Observed on real data: 0.0352 (var=0.0003)

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
P_REF = {'((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8566591422121896, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8610503282275711, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8595555555555555, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8562962962962963, '((1, 0, 0, 1), (1, 1, 1, 0))|0': 0.8435277382645804, '((1, 0, 0, 1), (1, 1, 1, 0))|1': 0.8395624430264357, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.1537313432835821, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.1407079646017699, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8465703971119134, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8424855491329479, '((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8467261904761905, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8794326241134752, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8613861386138614, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8447024673439768, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.14742857142857144, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.1783783783783784, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8619528619528619, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8514851485148515, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.1471861471861472, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.16517189835575485, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.1467065868263473, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.14752650176678445, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.14094775212636695, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.19202363367799113, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8496993987975952, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8708685626441199, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8471810089020771, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8534635879218473, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.13849509269356597, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.16081540203850508, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.1365149833518313, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.13793103448275862}
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

**Observed (real) value:** 0.0648 (var=0.0004)
**Predicted under pi_5:** 0.0014 (var=0.0002)
**Predicted under pi_3:** 0.0206 (var=0.0002)

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
P_REF = {'((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.8082788671023965, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.8680089485458613, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.16129032258064516, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.13183520599250936, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.4983089064261556, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.5169769989047097, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.8590604026845637, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.8344370860927153, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.7848101265822784, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8643634037108126, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.844950213371266, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.8568824065633546, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.835820895522388, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8622715404699739, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.138996138996139, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.12512218963831867, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.8567454798331016, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.877890841813136, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.5089974293059126, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8239700374531835, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8773646444879322, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.14710252600297177, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.1251109139307897, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8580786026200873, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8687782805429864, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.861995753715499, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.8585402558314522, '((0, 0, 0, 0), (1, 0, 0, 0))|0': 0.8573248407643312, '((0, 0, 0, 0), (1, 0, 0, 0))|1': 0.8551724137931035, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8632326820603907, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8771220695230396}
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

**Observed (real) value:** 0.0885 (var=0.0006)
**Predicted under pi_5:** 0.0140 (var=0.0001)
**Predicted under pi_3:** 0.0051 (var=0.0002)

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
P_REF = {'((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.8103448275862069, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.875, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.15528634361233482, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.14573991031390135, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.8577857785778578, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.8359941944847605, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.8294243070362474, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.8444778362133734, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.15181058495821728, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.13493530499075784, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.1558538404175988, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.1655773420479303, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.1476147614761476, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.14804063860667635, '((0, 1, 0, 1), (0, 1, 1, 1))|0': 0.8198847262247838, '((0, 1, 0, 1), (0, 1, 1, 1))|1': 0.8670886075949367, '((0, 0, 1, 0), (1, 0, 0, 1))|0': 0.8489525909592062, '((0, 0, 1, 0), (1, 0, 0, 1))|1': 0.8689809630459127, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.14231974921630094, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.15261845386533665, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.8493449781659389, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.8506787330316742, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8577680525164114, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8295711060948081, '((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8246013667425968, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8708765315739868, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8229166666666666, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.851063829787234, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.12705530642750373, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.13793103448275862}
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

**Observed (real) value:** 0.0796 (var=0.0006)
**Predicted under pi_5:** 0.0010 (var=0.0002)
**Predicted under pi_3:** 0.0493 (var=0.0003)

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
P_REF = {'((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.1539039039039039, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.1517094017094017, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.8102815177478581, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.8331637843336724, '((0, 0, 0, 1), (0, 1, 0, 1))|0': 0.8107569721115537, '((0, 0, 0, 1), (0, 1, 0, 1))|1': 0.847457627118644, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.14677103718199608, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.15938303341902313, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.1515837104072398, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.1462882096069869, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.478494623655914, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4578313253012048, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.15146579804560262, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.1888111888111888, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.1648706896551724, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.16628440366972477, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.14054600606673406, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1467324290998767, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8410404624277457, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8582677165354331, '((1, 0, 1, 0), (1, 0, 1, 1))|0': 0.8095890410958904, '((1, 0, 1, 0), (1, 0, 1, 1))|1': 0.8411214953271028, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.15300546448087432, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.11797752808988764, '((1, 1, 0, 1), (0, 0, 0, 1))|0': 0.14821591948764867, '((1, 1, 0, 1), (0, 0, 0, 1))|1': 0.14742014742014742, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8242640499553969, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.801178203240059, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.5322878228782287, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.5111731843575419, '((1, 1, 0, 0), (1, 1, 0, 1))|0': 0.8487903225806451, '((1, 1, 0, 0), (1, 1, 0, 1))|1': 0.8514851485148515}
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

**Observed (real) value:** 0.0508 (var=0.0002)
**Predicted under pi_5:** 0.0402 (var=0.0004)
**Predicted under pi_3:** 0.0008 (var=0.0001)

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
P_REF = {'((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.8725314183123878, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.8367346938775511, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5176151761517616, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.4996505939902166, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.13953488372093023, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.18456375838926176, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.14543404735062007, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.15553121577217963, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.16082659478885894, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.17321688500727803, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8301886792452831, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8435374149659864, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.15270935960591134, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.14394904458598726, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.15707620528771385, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.12718786464410736, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.4788732394366197, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.5240963855421686, '((1, 0, 0, 0), (1, 0, 1, 0))|0': 0.860236220472441, '((1, 0, 0, 0), (1, 0, 1, 0))|1': 0.8507653061224489, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8608597285067874, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8438864628820961, '((0, 1, 0, 0), (1, 1, 0, 0))|0': 0.8354285714285714, '((0, 1, 0, 0), (1, 1, 0, 0))|1': 0.8475675675675676, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1678463094034378, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.16522811344019728, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.856425702811245, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.8383084577114428, '((1, 1, 0, 0), (1, 1, 1, 1))|0': 0.8713450292397661, '((1, 1, 0, 0), (1, 1, 1, 1))|1': 0.8817204301075269, '((1, 1, 0, 1), (1, 0, 0, 1))|0': 0.13852813852813853, '((1, 1, 0, 1), (1, 0, 0, 1))|1': 0.13911472448057813}
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

**Observed (real) value:** 0.0463 (var=0.0003)
**Predicted under pi_5:** 0.0301 (var=0.0001)
**Predicted under pi_3:** 0.0063 (var=0.0001)

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
P_REF = {'((1, 0, 1, 0), (1, 1, 0, 1))|0': 0.8346379647749511, '((1, 0, 1, 0), (1, 1, 0, 1))|1': 0.8277634961439588, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.12027833001988071, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.14231738035264482, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.8267477203647416, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.7954545454545454, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.1488862837045721, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.12777191129883844, '((1, 0, 0, 1), (1, 1, 0, 0))|0': 0.8195020746887967, '((1, 0, 0, 1), (1, 1, 0, 0))|1': 0.8163841807909604, '((1, 1, 1, 0), (1, 0, 1, 1))|0': 0.1988388969521045, '((1, 1, 1, 0), (1, 0, 1, 1))|1': 0.1485148514851485, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.11472448057813911, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.13852813852813853, '((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.8232104121475055, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.8428246013667426, '((1, 1, 0, 0), (0, 0, 0, 0))|0': 0.11304347826086956, '((1, 1, 0, 0), (0, 0, 0, 0))|1': 0.15327102803738318, '((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.4293286219081272, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.42065868263473055, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12097407698350353, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.12144212523719165, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.1331521739130435, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.12593984962406016, '((0, 0, 1, 1), (0, 1, 0, 1))|0': 0.572405929304447, '((0, 0, 1, 1), (0, 1, 0, 1))|1': 0.6099518459069021, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1233140655105973, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.13910761154855644, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.14508138711960367, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.1834625322997416, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8810810810810811, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8742857142857143}
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

**Observed (real) value:** 0.1035 (var=0.0008)
**Predicted under pi_5:** 0.0251 (var=0.0002)
**Predicted under pi_3:** 0.0008 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the JSD metric is conditioned on the previous trial's response, capturing sequential dependence in decision-making. Both Theory 1 (Take-The-Best) and Theory 2 (Weighted Additive) are memoryless models that predict choices based solely on the current trial's features. Because of this, both theories fail to capture the observed JSD values in the real data (which range from 0.03 to 0.10), indicating that humans exhibit sequential effects (such as choice inertia or alternation) that neither pure TTB nor pure WADD accounts for. While WADD generally aligns better with the compensatory nature of the choices, the consistent gap between predicted and observed metrics suggests that a purely memoryless WADD is insufficient.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Weighted Additive with Choice Inertia' theory. This theory should maintain the compensatory feature integration of the WADD model (weighting each feature by its validity) but explicitly incorporate a sequential dependence mechanism. Specifically, it should add a 'choice inertia' parameter that provides a utility bonus to the option chosen on the immediately preceding trial, thereby naturally explaining the conditional response distributions that the current memoryless models fail to capture."
}
```

## Usage

```json
{
  "prompt_token_count": 22208,
  "candidates_token_count": 282,
  "total_token_count": 23795
}
```
