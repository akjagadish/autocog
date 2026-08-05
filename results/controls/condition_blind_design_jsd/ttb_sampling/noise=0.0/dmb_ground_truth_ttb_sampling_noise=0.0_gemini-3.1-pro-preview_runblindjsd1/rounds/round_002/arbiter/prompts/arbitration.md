# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take-The-Best (TTB) is a non-compensatory fast-and-frugal heuristic. People rank features by their subjective validity and compare options sequentially. The decision is based entirely on the first feature that discriminates between the options, ignoring all remaining features. If all features tie, the decision maker guesses randomly. Response noise is captured by a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_4
Two-Stage Heuristic (Hybrid TTB-Tallying over Remaining Cues): Decision makers use a bounded sequential search, initially relying on the most valid cues to discriminate between options (Take-The-Best). If the top 'k' cues fail to discriminate (i.e., they are tied), the decision maker falls back to a computationally simpler 'Tallying' strategy. However, instead of tallying all cues, they only tally the remaining unexamined cues, avoiding double-counting the cues that already tied. This provides a psychologically plausible sequential search process that blends lexicographic and tallying strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    k = int(parameters["max_cues"])
    
    decided = False
    p_core = np.array([0.5, 0.5])
    
    # Stage 1: Take-The-Best on the top k cues
    for idx in order[:k]:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            decided = True
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            decided = True
            break
            
    # Stage 2: Fallback to Tallying on remaining unexamined cues if top cues tie
    if not decided:
        remaining_order = order[k:]
        if len(remaining_order) > 0:
            a_rem = a[remaining_order]
            b_rem = b[remaining_order]
            a_wins = float(np.sum(a_rem > b_rem))
            b_wins = float(np.sum(b_rem > a_rem))
            if a_wins > b_wins:
                p_core = np.array([1.0, 0.0])
            elif b_wins > a_wins:
                p_core = np.array([0.0, 1.0])
            else:
                p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([0.5, 0.5])
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.55, 0.65, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 9: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 11: A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 12: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 13: A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 14: A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 1, 1), (0, 1, 1, 1))|0': 0.167420814479638, '((1, 1, 1, 1), (0, 1, 1, 1))|1': 0.11561119293078057, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.8738938053097345, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8560267857142857, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8552631578947368, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8891369047619048, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8465011286681715, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8732498157700811, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8842105263157894, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8854961832061069, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.13548387096774195, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.12434456928838951, '((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8744343891402715, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8700873362445415, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.868995633187773, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.8688230008984726, '((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.8370044052863436, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.8785759694850604, '((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.8617511520737328, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.8740849194729137, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.8440366972477065, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.8868520859671302, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8442265795206971, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8877551020408163, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.8174603174603174, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.8869509043927648, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8436213991769548, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.875951293759513, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.12236286919831224, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.11068458093410109, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8761261261261262, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8606194690265486}
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
- Predicted under pi_4 (simulated): 0.0029 (var=0.0002)
- Observed on real data: 0.0021 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.87, 0.89, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 4: A=[0, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1]  B=[1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 0, 1]  B=[0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  trial 11: A=[0, 1, 1, 0]  B=[0, 1, 0, 0]
  trial 12: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 13: A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  trial 14: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 15: A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 16: A=[0, 1, 0, 1]  B=[0, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8851744186046512, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.85431654676259, '((1, 1, 1, 1), (0, 1, 0, 0))|0': 0.138815207780725, '((1, 1, 1, 1), (0, 1, 0, 0))|1': 0.13303437967115098, '((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.7742331288343558, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.7372262773722628, '((1, 0, 1, 1), (0, 1, 0, 0))|0': 0.11290959336754836, '((1, 0, 1, 1), (0, 1, 0, 0))|1': 0.14995313964386128, '((1, 1, 1, 1), (1, 0, 1, 1))|0': 0.13828238719068414, '((1, 1, 1, 1), (1, 0, 1, 1))|1': 0.12488769092542677, '((0, 1, 0, 0), (1, 0, 1, 0))|0': 0.8749523446435379, '((0, 1, 0, 0), (1, 0, 1, 0))|1': 0.8515864892528148, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.11936339522546419, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.14499252615844543, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.12627551020408162, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.15517241379310345, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.13167259786476868, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.13609467455621302, '((1, 1, 0, 1), (0, 1, 0, 0))|0': 0.12406417112299466, '((1, 1, 0, 1), (0, 1, 0, 0))|1': 0.13641618497109825, '((1, 0, 1, 0), (0, 1, 0, 0))|0': 0.13288288288288289, '((1, 0, 1, 0), (0, 1, 0, 0))|1': 0.12938596491228072, '((0, 0, 1, 0), (0, 1, 0, 0))|0': 0.24436363636363637, '((0, 0, 1, 0), (0, 1, 0, 0))|1': 0.2447058823529412, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8819255222524978, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8669527896995708, '((0, 0, 1, 0), (1, 1, 0, 0))|0': 0.8461538461538461, '((0, 0, 1, 0), (1, 1, 0, 0))|1': 0.8728323699421965}
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
- Predicted under pi_3 (simulated): 0.0028 (var=0.0001)
- Predicted under pi_4 (simulated): 0.0014 (var=0.0001)
- Observed on real data: 0.0031 (var=0.0001)

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
P_REF = {'((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8471337579617835, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8368298368298368, '((1, 1, 0, 1), (0, 1, 1, 0))|0': 0.17772215269086358, '((1, 1, 0, 1), (0, 1, 1, 0))|1': 0.16583416583416583, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.15309734513274337, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.1537313432835821, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8552522746071133, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8324873096446701, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.16329479768786126, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1299638989169675, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8578052550231839, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8640093786635404, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8636779505946935, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8458274398868458, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.11648568608094768, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.16010165184243966, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.49140271493212667, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.5179856115107914, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.5100502512562815, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.49302788844621515, '((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.10619469026548672, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.11892963330029732, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.512781954887218, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.474468085106383, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12003311258278146, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.13175675675675674, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.14905814905814907, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.15889464594127806, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.508296943231441, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.4900763358778626, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8048780487804879, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8489451476793249}
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

**Observed (real) value:** 0.0205 (var=0.0002)
**Predicted under pi_3:** 0.0249 (var=0.0002)
**Predicted under pi_4:** 0.0158 (var=0.0002)

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
P_REF = {'((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.7913950456323338, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.7831558567279767, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.2125, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.17314814814814813, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.3384201077199282, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.35276967930029157, '((0, 0, 0, 0), (0, 1, 1, 0))|0': 0.8560700876095119, '((0, 0, 0, 0), (0, 1, 1, 0))|1': 0.8271728271728271, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.22364217252396165, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.206984667802385, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.2425249169435216, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.21962095875139354, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7389858686616791, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.6549413735343383, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.2998805256869773, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.3052959501557632, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.45555555555555555, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.45308641975308644, '((0, 0, 0, 0), (1, 0, 1, 1))|0': 0.8474387527839644, '((0, 0, 0, 0), (1, 0, 1, 1))|1': 0.835920177383592, '((0, 1, 0, 1), (1, 1, 1, 1))|0': 0.8377125193199382, '((0, 1, 0, 1), (1, 1, 1, 1))|1': 0.8475967174677609, '((0, 1, 1, 0), (0, 0, 0, 0))|0': 0.16120906801007556, '((0, 1, 1, 0), (0, 0, 0, 0))|1': 0.17693836978131214, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.7583444592790387, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.7916270218839201, '((0, 0, 0, 1), (0, 1, 1, 1))|0': 0.8501026694045175, '((0, 0, 0, 1), (0, 1, 1, 1))|1': 0.8317191283292978, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.2862669245647969, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.24551831644583008, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.4927675988428158, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.583224115334207}
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

**Observed (real) value:** 0.0172 (var=0.0002)
**Predicted under pi_3:** 0.0258 (var=0.0002)
**Predicted under pi_4:** 0.0175 (var=0.0002)

### Experiment 5
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
P_REF = {'((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8840579710144928, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8748615725359912, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.11559139784946236, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.12134502923976608, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.1378692927484333, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.14641288433382138, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.10933333333333334, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.1362962962962963, '((0, 1, 1, 0), (1, 1, 0, 0))|0': 0.8860648553900088, '((0, 1, 1, 0), (1, 1, 0, 0))|1': 0.874051593323217, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.11531531531531532, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.14202898550724638, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1309823677581864, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1650943396226415, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.1084070796460177, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.11160714285714286, '((1, 1, 1, 1), (1, 0, 1, 0))|0': 0.10942441492726122, '((1, 1, 1, 1), (1, 0, 1, 0))|1': 0.2146118721461187, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.860832137733142, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8712601994560291, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.11829134720700986, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.11161217587373168, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.14411764705882352, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.12142857142857143, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8772378516624041, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8177966101694916, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.10836501901140684, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.1891891891891892, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.11185682326621924, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.16993464052287582, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8882733148661126, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8729016786570744}
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

**Observed (real) value:** 0.0038 (var=0.0001)
**Predicted under pi_3:** 0.0011 (var=0.0001)
**Predicted under pi_4:** 0.0020 (var=0.0001)

### Experiment 6
**Design**
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.6834677419354839, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.75, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.2972972972972973, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.3210930828351836, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.5884861407249466, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.525522041763341, '((0, 0, 1, 0), (1, 1, 1, 1))|0': 0.8384074941451991, '((0, 0, 1, 0), (1, 1, 1, 1))|1': 0.8456659619450317, '((1, 1, 0, 0), (0, 0, 1, 0))|0': 0.23440453686200377, '((1, 1, 0, 0), (0, 0, 1, 0))|1': 0.2749326145552561, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.15806451612903225, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.1875, '((0, 1, 1, 0), (0, 0, 1, 1))|0': 0.4133489461358314, '((0, 1, 1, 0), (0, 0, 1, 1))|1': 0.5232558139534884, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.6568627450980392, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.7034990791896869, '((1, 0, 0, 1), (1, 1, 0, 0))|0': 0.48372781065088755, '((1, 0, 0, 1), (1, 1, 0, 0))|1': 0.5204626334519573, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.3069544364508393, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.3115942028985507, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8237082066869301, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8081180811808119, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.823943661971831, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8312236286919831, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.376425855513308, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.44919786096256686, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.721120186697783, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.7592788971367974, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.5577156743620899, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.646878198567042, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.26953748006379585, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.23870417732310314}
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

**Observed (real) value:** 0.0335 (var=0.0005)
**Predicted under pi_3:** 0.0278 (var=0.0004)
**Predicted under pi_4:** 0.0304 (var=0.0003)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, the Two-Stage Heuristic (pi_4) generally outperforms pure Take-The-Best (pi_3). In experiments 3, 4, 5, and 6, pi_4's predictions are consistently closer to the observed values than pi_3's. While pi_3 performs slightly better in experiments 1 and 2, pi_4 is much more robust overall. Pure Take-The-Best seems too rigid, as subjects appear to use more than just the single most valid discriminating cue. However, pi_4's hybrid approach of TTB followed by tallying might still be an approximation of a fully compensatory strategy.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the pure Take-The-Best theory with a Weighted Additive (WADD) theory. In WADD, decision makers integrate all available cues by weighting each feature by its subjective validity (or a function of it) and summing them up to form an overall value for each option. This provides a fully compensatory baseline that contrasts with the bounded rationality heuristics (like the hybrid TTB-Tallying model) and can help determine if subjects are truly using sequential heuristics or if they are integrating all information."
}
```

## Usage

```json
{
  "prompt_token_count": 17281,
  "candidates_token_count": 281,
  "total_token_count": 18379
}
```
