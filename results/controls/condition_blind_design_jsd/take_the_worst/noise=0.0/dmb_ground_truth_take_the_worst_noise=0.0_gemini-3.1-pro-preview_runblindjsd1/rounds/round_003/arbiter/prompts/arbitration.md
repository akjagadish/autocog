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
Weighted Additive (WADD) with Choice Inertia: Decision-makers evaluate options by computing a weighted sum of their features, where the weights correspond to the given cue validities. However, choices are not independent across trials. Individuals exhibit a sequential dependency (choice inertia or alternation) where the probability of selecting an option is biased by whether it was chosen on the immediately preceding trial. This is modeled by adding an inertia parameter to the logit of the previously chosen option before applying the softmax response rule.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted additive evaluation
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    inertia = float(parameters["inertia"])
    
    logits = beta * np.array([score_a, score_b])
    
    # Apply choice inertia based on the previous trial's response
    if history and "response" in history and len(history["response"]) > 0:
        last_choice = int(history["response"][-1])
        if 0 <= last_choice < 2:
            logits[last_choice] += inertia
            
    # Numerically stable softmax
    z = logits - np.max(logits)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_3
People make decisions using a Weighted Additive (WADD) strategy. Instead of relying on a single discriminating cue (Take The Best) or treating all cues equally (Tallying), decision makers compute a comprehensive value for each option by summing its feature values weighted by their respective cue validities. The option with the higher total weighted value is favored. This compensatory approach allows multiple weak cues to outweigh a single strong cue, reflecting a more exhaustive integration of available information. Choice stochasticity is modeled via a softmax over the weighted sums with an inverse temperature parameter (beta), along with an independent lapse rate (epsilon) for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute the weighted sum of features for each option
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Convert scores to probabilities using a numerically stable softmax
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
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.95, 0.59, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 12: A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 13: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 14: A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 15: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 16: A=[1, 1, 1, 0]  B=[1, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.08614232209737828, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.5587431693989071, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.14430379746835442, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.7040650406504065, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.15143929912390489, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.7042957042957043, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.39809296781883197, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.882414151925078, '((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.1911021233569262, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.7632552404438965, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.1393188854489164, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.6919374247894103, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.09440993788819876, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.4894472361809045, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.1265164644714038, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.6811145510835913, '((1, 1, 1, 0), (1, 0, 1, 1))|0': 0.21995708154506438, '((1, 1, 1, 0), (1, 0, 1, 1))|1': 0.7880184331797235, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.12876427829698858, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.7347670250896058, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.4838709677419355, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8748114630467572, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.38969072164948454, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8614457831325302, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.2949117341640706, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.8375149342891278, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.13854166666666667, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.6607142857142857, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.09968847352024922, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.5567502986857825, '((0, 0, 1, 0), (1, 1, 1, 1))|0': 0.48509485094850946, '((0, 0, 1, 0), (1, 1, 1, 1))|1': 0.8903318903318903}
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
- Predicted under pi_3 (simulated): 0.0631 (var=0.0005)
- Observed on real data: 0.1007 (var=0.0008)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.71, 0.84, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 12: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 13: A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 14: A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  trial 15: A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 16: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8555555555555555, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8633333333333333, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8577586206896551, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8638663967611336, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.12944162436548223, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.13764044943820225, '((1, 1, 1, 1), (0, 1, 1, 0))|0': 0.13256227758007116, '((1, 1, 1, 1), (0, 1, 1, 0))|1': 0.1301775147928994, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.134180790960452, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.1575091575091575, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.12055974165769645, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.1285878300803674, '((0, 1, 1, 0), (0, 0, 1, 1))|0': 0.2547914317925592, '((0, 1, 1, 0), (0, 0, 1, 1))|1': 0.26396495071193865, '((0, 0, 0, 0), (1, 0, 0, 0))|0': 0.8759328358208955, '((0, 0, 0, 0), (1, 0, 0, 0))|1': 0.8392857142857143, '((0, 0, 0, 0), (1, 0, 1, 1))|0': 0.879862700228833, '((0, 0, 0, 0), (1, 0, 1, 1))|1': 0.8585313174946004, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.12701829924650163, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.17795637198622274, '((0, 1, 0, 0), (0, 0, 1, 0))|0': 0.7338345864661654, '((0, 1, 0, 0), (0, 0, 1, 0))|1': 0.7030837004405286, '((0, 1, 1, 1), (0, 1, 0, 0))|0': 0.1387900355871886, '((0, 1, 1, 1), (0, 1, 0, 0))|1': 0.1390532544378698, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.13925729442970822, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.13193116634799235, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8413566739606126, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8047404063205418, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8732840549102429, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.8640093786635404}
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
- Predicted under pi_5 (simulated): 0.0768 (var=0.0034)
- Predicted under pi_3 (simulated): 0.0008 (var=0.0002)
- Observed on real data: 0.0537 (var=0.0004)

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
P_REF = {'((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8423383525243578, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8643815201192251, '((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.24615384615384617, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.14675324675324675, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.14125412541254126, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.20350877192982456, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.850965250965251, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8415841584158416, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.16739446870451238, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.14195867026055706, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8648401826484018, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.849645390070922, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8343685300207039, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8580106302201974, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8283752860411899, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.851063829787234, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.16551724137931034, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.14084507042253522, '((1, 1, 0, 1), (0, 1, 1, 0))|0': 0.16923076923076924, '((1, 1, 0, 1), (0, 1, 1, 0))|1': 0.1449438202247191, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.14798206278026907, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.17372262773722627, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12308868501529052, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.2073170731707317, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.14730447987851178, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.18426501035196688, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8387755102040816, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8725190839694656, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.8347953216374269, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.8530465949820788, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.8445040214477212, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8458149779735683}
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

**Observed (real) value:** 0.0960 (var=0.0007)
**Predicted under pi_5:** 0.0684 (var=0.0027)
**Predicted under pi_3:** 0.0025 (var=0.0002)

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
P_REF = {'((0, 1, 1, 0), (0, 0, 0, 0))|0': 0.1609403254972875, '((0, 1, 1, 0), (0, 0, 0, 0))|1': 0.13953488372093023, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.11976744186046512, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.128125, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.13706140350877194, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.12387387387387387, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8507795100222717, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8669623059866962, '((0, 0, 0, 0), (1, 0, 1, 1))|0': 0.8205128205128205, '((0, 0, 0, 0), (1, 0, 1, 1))|1': 0.8743961352657005, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8588120740019474, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8771021992238034, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1534344335414808, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.1561119293078056, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.13930348258706468, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.14339622641509434, '((0, 0, 0, 1), (0, 1, 1, 1))|0': 0.8685376661742984, '((0, 0, 0, 1), (0, 1, 1, 1))|1': 0.8717720391807658, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.13359920239282153, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.13927227101631118, '((0, 1, 0, 1), (1, 1, 1, 1))|0': 0.8641425389755011, '((0, 1, 0, 1), (1, 1, 1, 1))|1': 0.8813747228381374, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8737373737373737, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8765432098765432, '((0, 0, 0, 0), (0, 1, 1, 0))|0': 0.872617853560682, '((0, 0, 0, 0), (0, 1, 1, 0))|1': 0.8268991282689913, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.14428857715430862, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.15211970074812967, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.49504950495049505, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.49056603773584906, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.4828101644245142, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.5057471264367817}
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

**Observed (real) value:** 0.0325 (var=0.0004)
**Predicted under pi_5:** 0.0914 (var=0.0024)
**Predicted under pi_3:** 0.0028 (var=0.0001)

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
P_REF = {'((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.14257684761281883, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.2029520295202952, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.12597547380156077, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.1406423034330011, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8563049853372434, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8667262969588551, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8413173652694611, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8763250883392226, '((0, 0, 0, 0), (0, 1, 1, 1))|0': 0.809322033898305, '((0, 0, 0, 0), (0, 1, 1, 1))|1': 0.8689759036144579, '((0, 1, 1, 1), (1, 1, 1, 1))|0': 0.8490566037735849, '((0, 1, 1, 1), (1, 1, 1, 1))|1': 0.8420256991685563, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.15807174887892378, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.1211453744493392, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.13644524236983843, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.16034985422740525, '((0, 0, 1, 0), (0, 0, 0, 0))|0': 0.1548154815481548, '((0, 0, 1, 0), (0, 0, 0, 0))|1': 0.15384615384615385, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.13353338334583645, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.17130620985010706, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8703427719821163, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8680248007085917, '((0, 1, 1, 0), (0, 1, 1, 1))|0': 0.8742469879518072, '((0, 1, 1, 0), (0, 1, 1, 1))|1': 0.8347457627118644, '((0, 1, 0, 0), (0, 0, 1, 0))|0': 0.1336405529953917, '((0, 1, 0, 0), (0, 0, 1, 0))|1': 0.13898704358068315, '((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8688969258589512, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.861671469740634, '((0, 1, 1, 1), (0, 1, 0, 0))|0': 0.1352154531946508, '((0, 1, 1, 1), (0, 1, 0, 0))|1': 0.19383259911894274, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.13435114503816795, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.15918367346938775}
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

**Observed (real) value:** 0.0940 (var=0.0007)
**Predicted under pi_5:** 0.0743 (var=0.0031)
**Predicted under pi_3:** 0.0211 (var=0.0001)

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
P_REF = {'((1, 1, 1, 1), (1, 0, 1, 0))|0': 0.15768930523028885, '((1, 1, 1, 1), (1, 0, 1, 0))|1': 0.18882466281310212, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.12979683972911965, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.15207877461706784, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8172645739910314, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8414096916299559, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.14609375, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.19423076923076923, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.17772692601067888, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.18609406952965235, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.14332514332514332, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.2114695340501792, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.157725321888412, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.1313364055299539, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8474051123160341, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8172888015717092, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.18388429752066116, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.17427884615384615, '((0, 1, 1, 0), (1, 1, 0, 0))|0': 0.6583333333333333, '((0, 1, 1, 0), (1, 1, 0, 0))|1': 0.625, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8367521367521368, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8634920634920635, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8360030511060259, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8118609406952966, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.14745011086474502, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.15812917594654788, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.19033457249070632, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.21978021978021978, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.20466321243523317, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.2087227414330218, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.16091954022988506, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.17556346381969157}
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

**Observed (real) value:** 0.0961 (var=0.0006)
**Predicted under pi_5:** 0.0737 (var=0.0031)
**Predicted under pi_3:** 0.0010 (var=0.0002)

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
P_REF = {'((0, 0, 1, 0), (0, 1, 0, 0))|0': 0.5139882888744307, '((0, 0, 1, 0), (0, 1, 0, 0))|1': 0.5171102661596958, '((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.5025188916876574, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.5079522862823062, '((0, 1, 0, 0), (1, 0, 1, 0))|0': 0.8493392070484581, '((0, 1, 0, 0), (1, 0, 1, 0))|1': 0.8263157894736842, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.15172413793103448, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.14344262295081966, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.49163346613545816, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.5064220183486239, '((1, 0, 1, 1), (0, 1, 0, 0))|0': 0.12944983818770225, '((1, 0, 1, 1), (0, 1, 0, 0))|1': 0.1374113475177305, '((1, 1, 1, 1), (0, 1, 0, 0))|0': 0.12340036563071298, '((1, 1, 1, 1), (0, 1, 0, 0))|1': 0.1643059490084986, '((1, 1, 1, 1), (1, 0, 1, 1))|0': 0.13463098134630982, '((1, 1, 1, 1), (1, 0, 1, 1))|1': 0.14991181657848324, '((1, 1, 0, 1), (0, 1, 0, 0))|0': 0.1153250773993808, '((1, 1, 0, 1), (0, 1, 0, 0))|1': 0.1594488188976378, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.5108267716535433, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.48596938775510207, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.14106019766397124, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.16885007278020378, '((0, 0, 1, 0), (1, 1, 0, 0))|0': 0.8499506416584403, '((0, 0, 1, 0), (1, 1, 0, 0))|1': 0.8729351969504447, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.4948571428571429, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.52, '((1, 0, 1, 0), (0, 1, 0, 0))|0': 0.14845360824742268, '((1, 0, 1, 0), (0, 1, 0, 0))|1': 0.15283018867924528}
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

**Observed (real) value:** 0.0783 (var=0.0007)
**Predicted under pi_5:** 0.0589 (var=0.0019)
**Predicted under pi_3:** 0.0087 (var=0.0002)

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
P_REF = {'((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.6329411764705882, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.6567272727272727, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.7513089005235603, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8596614950634697, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8232931726907631, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8448540706605223, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8129496402877698, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8442796610169492, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.6349206349206349, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.6848659003831418, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8050541516245487, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8290529695024077, '((1, 1, 1, 1), (0, 1, 1, 1))|0': 0.153125, '((1, 1, 1, 1), (0, 1, 1, 1))|1': 0.15344827586206897, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.7430167597765364, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.8016643550624133, '((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.7675675675675676, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.8232931726907631, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.7801980198019802, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.8061776061776061, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.19974874371859297, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.17430278884462153, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.6336898395721925, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.655893536121673, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.1887905604719764, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.15151515151515152, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.6541935483870968, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.6419512195121951, '((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8364864864864865, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8358490566037736, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8495887191539365, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8314014752370916}
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

**Observed (real) value:** 0.1349 (var=0.0011)
**Predicted under pi_5:** 0.0597 (var=0.0030)
**Predicted under pi_3:** 0.0011 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The evaluation relies on a sequence-aware metric (JSD conditional on the previous response). THEORY 2 (pi_3) is a purely independent-trial WADD model, and it consistently predicts near-zero values for this metric, drastically underestimating the observed sequential dependencies in the real data across almost all experiments. THEORY 1 (pi_5) incorporates simple choice inertia (repeating the previous response index) and gets much closer to the observed values in Experiments 2, 3, 5, 6, 7, and 8. However, THEORY 1 still shows some discrepancies, suggesting that while sequential dependence is present, pure index-based choice inertia might not be the complete story. Since THEORY 2 completely fails to account for the sequential dynamics captured by the metric, it should be replaced with a stronger competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that incorporates sequential dependence, but moves beyond simple index-based choice inertia. For instance, propose 'WADD with Feature-based Inertia', where subjects are biased toward options that share features with the option they chose on the previous trial. This accounts for choices being driven by the previously attended attributes rather than just repeating a spatial or index-based response, potentially capturing the conditional choice profile better than THEORY 1."
}
```

## Usage

```json
{
  "prompt_token_count": 21989,
  "candidates_token_count": 297,
  "total_token_count": 23494
}
```
