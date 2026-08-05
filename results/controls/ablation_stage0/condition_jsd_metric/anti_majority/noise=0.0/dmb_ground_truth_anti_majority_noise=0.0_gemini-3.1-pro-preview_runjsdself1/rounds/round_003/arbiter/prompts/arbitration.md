# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_5
Decision-makers use a bounded evidence accumulation process that integrates sequential dependencies. They default to a non-compensatory heuristic (Take The Best) but smoothly transition to a compensatory strategy (Weighted Additive) when opposing evidence exceeds a soft threshold. Furthermore, choices are subject to sequential dependencies: the baseline preference for an option is dynamically adjusted based on the previous trial's choice, capturing choice inertia or auto-correlation. This moving baseline influences the final decision probabilities alongside the strategy-specific evidence.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    import scipy.special
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate log-odds weights for WADD and opposing evidence
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # --- Take The Best (TTB) Phase ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is not None:
        # Calculate opposing evidence: sum of weights where the TTB loser beats the TTB winner
        if winner_ttb == 0:
            opposing_evidence = np.sum(weights * (b > a))
        else:
            opposing_evidence = np.sum(weights * (a > b))
            
        k = float(parameters["k"])
        threshold = float(parameters["threshold"])
        # Soft transition: probability of using WADD is a logistic function of opposing evidence
        p_wadd = float(scipy.special.expit(k * (opposing_evidence - threshold)))
    else:
        # If no cue discriminates, default to WADD (which will tie)
        p_wadd = 1.0
        
    # --- Sequential Dependency (Inertia) ---
    inertia = float(parameters["inertia"])
    z_inertia = np.array([0.0, 0.0])
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = int(history["response"][-1])
        z_inertia[last_resp] = inertia
        
    # --- Decision Formulation ---
    # WADD probabilities with inertia
    scores_wadd = np.array([np.sum(weights * a), np.sum(weights * b)])
    beta_wadd = float(parameters["beta_wadd"])
    z_w = beta_wadd * scores_wadd + z_inertia
    z_w = z_w - np.max(z_w)
    e_w = np.exp(z_w)
    p_core_wadd = e_w / np.sum(e_w)
    
    # TTB probabilities with inertia
    if winner_ttb is not None:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
    beta_ttb = float(parameters["beta_ttb"])
    z_t = beta_ttb * scores_ttb + z_inertia
    z_t = z_t - np.max(z_t)
    e_t = np.exp(z_t)
    p_core_ttb = e_t / np.sum(e_t)
    
    # Mix strategies based on soft threshold
    p_core = p_wadd * p_core_wadd + (1.0 - p_wadd) * p_core_ttb
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Rationale:** To cleanly dissociate the advocated strict Take The Best (TTB) theory from the competing Soft Compensatory Check model, we parametrically increase the amount of 'opposing evidence' (lower-validity cues favoring the TTB loser) while holding the top discriminating cue constant. In this design, Option A is always favored by the highest-validity cue. Across the main trials, the number of lower-validity cues favoring Option B increases, which increases the total log-odds weight of opposing evidence. Strict TTB ignores all cues beyond the first discriminating one and therefore predicts a constant, high probability of choosing Option A across all these trials. The competing theory, however, evaluates the opposing evidence; as it increases, the model's probability of switching from TTB to the Weighted Additive (WADD) strategy increases. Because WADD strongly favors Option B in trials with high opposing evidence, the competing model predicts a graded, sharp decrease in the probability of choosing Option A. This provides a clear, quantitative dissociation: a flat line (TTB) versus a decreasing curve (Competing Model).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|0': 0.8468834688346883, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|1': 0.8427128427128427, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.1435793731041456, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.15844636251541308, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.8491164476665156, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.8557071069633884, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.8375763747454175, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.8398203592814372, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.15498357064622126, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.15783540022547915, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.13622011727559766, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.16702819956616052, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.1452265372168285, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.16400709219858156, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.14634146341463414, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.14193548387096774}
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
- Predicted under pi_1 (simulated): 0.0005 (var=0.0002)
- Predicted under pi_5 (simulated): 0.0219 (var=0.0068)
- Observed on real data: 0.0475 (var=0.0018)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 9: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 10: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 11: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 12: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]

**Rationale:** To cleanly dissociate the advocated Soft Compensatory Check model from strict Take The Best (TTB), we exploit the sequential dependency (choice inertia) unique to the advocated theory. We construct an unbalanced experimental design where 83% of the unique trial pairs strongly favor Option A under both models (Option A dominates on the highest validity cues). This induces a strong global base-rate of 'A' responses. We randomly intersperse critical trials where Option A and Option B have identical feature vectors (perfect ties). Strict TTB has no mechanism for history dependence and must guess exactly 50/50 on these tied trials. In contrast, the advocated theory updates its baseline preference using the previous trial's choice. Because the preceding trial is highly likely to have resulted in an 'A' choice, the advocated model's inertia mechanism predicts a systematic deviation from 50/50 (typically a strong bias toward Option A) on the perfectly tied critical trials. This provides a pure, quantitative dissociation based on sequential dynamics rather than static feature weights.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 1, 1), (0, 0, 1, 0, 0))|0': 0.1690223792697291, '((1, 1, 0, 1, 1), (0, 0, 1, 0, 0))|1': 0.18518518518518517, '((1, 1, 1, 0, 1), (0, 0, 0, 1, 0))|0': 0.16879659211927583, '((1, 1, 1, 0, 1), (0, 0, 0, 1, 0))|1': 0.18582375478927204, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|0': 0.15929675013319125, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|1': 0.18164435946462715, '((1, 1, 1, 1, 1), (0, 0, 0, 0, 0))|0': 0.15846994535519127, '((1, 1, 1, 1, 1), (0, 0, 0, 0, 0))|1': 0.19896640826873385, '((1, 1, 1, 1, 1), (1, 1, 1, 1, 1))|0': 0.45639187574671447, '((1, 1, 1, 1, 1), (1, 1, 1, 1, 1))|1': 0.39944903581267216, '((1, 1, 0, 0, 1), (0, 0, 1, 1, 0))|0': 0.15211267605633802, '((1, 1, 0, 0, 1), (0, 0, 1, 1, 0))|1': 0.168, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.15720081135902636, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.17523364485981308, '((0, 0, 0, 0, 0), (0, 0, 0, 0, 0))|0': 0.525532969757065, '((0, 0, 0, 0, 0), (0, 0, 0, 0, 0))|1': 0.44386422976501305, '((1, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.15070921985815602, '((1, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.18382352941176472, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|0': 0.18339276617422312, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|1': 0.20137299771167047, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|0': 0.14422057264050903, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|1': 0.17898832684824903, '((1, 1, 0, 1, 0), (0, 0, 1, 0, 1))|0': 0.18096809680968096, '((1, 1, 0, 1, 0), (0, 0, 1, 0, 1))|1': 0.17010309278350516}
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
- Predicted under pi_1 (simulated): 0.0014 (var=0.0001)
- Predicted under pi_5 (simulated): 0.0007 (var=0.0003)
- Observed on real data: 0.2264 (var=0.0034)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.15307653826913456, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.15302935665209244, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.16191155492154066, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.15241128298453138, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.850808122668877, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.8399326032013479, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.15374921235034655, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.14406358668653751, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.8381001021450459, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.8368107302533532, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.1472196900638104, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.17354196301564723, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.8401142041399001, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.8585720782173716, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.8504335260115607, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.8528880866425993}
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

**Observed (real) value:** 0.0213 (var=0.0038)
**Predicted under pi_1:** 0.0008 (var=0.0002)
**Predicted under pi_5:** 0.0118 (var=0.0023)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8595400870105656, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8663987945755901, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.499, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.505625, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.8404878048780487, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.8509677419354839, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.1471861471861472, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.14616755793226383, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.11522048364153627, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.15090543259557343, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 0))|0': 0.1424713031735314, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 0))|1': 0.14771118452100046, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.5124013528748591, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.4769989047097481, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8614576493762311, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.8724121328839672}
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

**Observed (real) value:** 0.1430 (var=0.0097)
**Predicted under pi_1:** 0.2353 (var=0.0052)
**Predicted under pi_5:** 0.1426 (var=0.0082)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13673548889754578, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.1582537517053206, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.14908637873754152, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.16526845637583892, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.15455512229705778, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.18485237483953787, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.13743218806509946, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.18587896253602307, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|0': 0.14599609375, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|1': 0.15077319587628865, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.15265017667844524, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.18831168831168832, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.8477350590026647, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.8314491264131552, '((0, 1, 0, 1, 0), (1, 0, 0, 0, 0))|0': 0.8527131782945736, '((0, 1, 0, 1, 0), (1, 0, 0, 0, 0))|1': 0.8517279821627648}
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

**Observed (real) value:** 0.0281 (var=0.0057)
**Predicted under pi_1:** 0.0007 (var=0.0001)
**Predicted under pi_5:** 0.0183 (var=0.0042)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.8362631843294827, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.8415164698570541, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8264248704663213, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8333333333333334, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8243243243243243, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8450635386119257, '((0, 0, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.14682139253279516, '((0, 0, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.1588380716934487, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.18855218855218855, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.1716171617161716, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.14464882943143811, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.1771523178807947, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|0': 0.8571428571428571, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|1': 0.8512396694214877, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.20281509916826615, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.178468624064479}
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

**Observed (real) value:** 0.2270 (var=0.0071)
**Predicted under pi_1:** 0.2373 (var=0.0045)
**Predicted under pi_5:** 0.1320 (var=0.0101)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1))|0': 0.13764654798089448, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1))|1': 0.21914357682619648, '((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.14113997673516868, '((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.21852731591448932, '((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1))|0': 0.13422291993720564, '((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1))|1': 0.19911504424778761, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.14207005116096025, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.18518518518518517, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.13920566260322453, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.19037199124726478, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0))|0': 0.14324746291959406, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0))|1': 0.18493150684931506, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|0': 0.13613380007779075, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|1': 0.21445221445221446, '((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|0': 0.1334115805946792, '((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|1': 0.21621621621621623, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 1))|0': 0.12776699029126212, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 1))|1': 0.2235294117647059}
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

**Observed (real) value:** 0.0139 (var=0.0045)
**Predicted under pi_1:** 0.0009 (var=0.0002)
**Predicted under pi_5:** 0.0023 (var=0.0014)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 0, 1, 0, 1), (1, 0, 0, 1, 0, 1, 0))|0': 0.5921273031825796, '((0, 1, 1, 0, 1, 0, 1), (1, 0, 0, 1, 0, 1, 0))|1': 0.5928689883913765, '((1, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0))|0': 0.1679144385026738, '((1, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0))|1': 0.1658703071672355, '((0, 1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0))|0': 0.6255792400370713, '((0, 1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0))|1': 0.7161241483724451, '((0, 0, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0))|0': 0.8542955326460481, '((0, 0, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0))|1': 0.8359788359788359, '((1, 0, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0, 0))|0': 0.46107178968655205, '((1, 0, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0, 0))|1': 0.32459248759744863, '((0, 0, 1, 0, 1, 0, 1), (1, 1, 0, 1, 0, 1, 0))|0': 0.8420208500400962, '((0, 0, 1, 0, 1, 0, 1), (1, 1, 0, 1, 0, 1, 0))|1': 0.8586296617519514, '((1, 0, 0, 1, 0, 1, 0), (0, 1, 1, 0, 1, 0, 1))|0': 0.4446022727272727, '((1, 0, 0, 1, 0, 1, 0), (0, 1, 1, 0, 1, 0, 1))|1': 0.3286290322580645, '((0, 1, 1, 0, 1, 0, 0), (1, 0, 0, 1, 0, 0, 0))|0': 0.6172300981461287, '((0, 1, 1, 0, 1, 0, 0), (1, 0, 0, 1, 0, 0, 0))|1': 0.6095751854349292, '((1, 0, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0, 0))|0': 0.31062271062271063, '((1, 0, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0, 0))|1': 0.34782608695652173, '((1, 1, 0, 1, 0, 0, 0), (0, 0, 1, 0, 1, 0, 0))|0': 0.17225747960108795, '((1, 1, 0, 1, 0, 0, 0), (0, 0, 1, 0, 1, 0, 0))|1': 0.16268311488049345, '((1, 1, 0, 1, 0, 1, 0), (0, 0, 1, 0, 1, 0, 1))|0': 0.14965397923875431, '((1, 1, 0, 1, 0, 1, 0), (0, 0, 1, 0, 1, 0, 1))|1': 0.14067524115755628, '((0, 0, 1, 0, 1, 0, 0), (1, 1, 0, 1, 0, 0, 0))|0': 0.8399412628487518, '((0, 0, 1, 0, 1, 0, 0), (1, 1, 0, 1, 0, 0, 0))|1': 0.8360433604336044}
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

**Observed (real) value:** 0.1472 (var=0.0017)
**Predicted under pi_1:** 0.0162 (var=0.0004)
**Predicted under pi_5:** 0.0017 (var=0.0011)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, strict Take The Best (THEORY 1) consistently fails to capture choices when opposing evidence is high (e.g., Exps 4, 6, 8), as seen by large deviations in JSD. THEORY 2 (Soft Compensatory Check with Inertia) performs better on some of these (e.g., Exp 4) but completely fails on Exp 2 and Exp 8, where its specific sequential dependency (inertia) mechanism predicts behavior that heavily deviates from the observed data (predicting ~0 JSD to itself while the data shows >0.14). This suggests that while a purely non-compensatory model (THEORY 1) is too strict, the specific hybrid and inertia mechanisms of THEORY 2 are also flawed. A brand-new theory is needed to replace the overly rigid THEORY 1, incorporating a more psychologically plausible mechanism for cue integration that doesn't rely on the failed inertia assumption.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Probabilistic Evidence Accumulation' or 'Noisy Tallying' theory. Instead of a deterministic cascade (TTB) or a hard-coded threshold for WADD + inertia, assume decision-makers sample cues probabilistically based on their validities. The sampled cues cast votes (tallying), and the decision is made once a fixed number of samples is drawn or a relative evidence threshold is reached. This naturally allows for compensatory behavior when many low-validity cues oppose a high-validity cue, without requiring an explicit strategy-switching threshold or trial-to-trial choice inertia."
}
```

## Usage

```json
{
  "prompt_token_count": 18690,
  "candidates_token_count": 359,
  "total_token_count": 20231
}
```
