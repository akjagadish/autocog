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
Probabilistic Evidence Accumulation with Baseline Evidence: Decision-makers sample cues probabilistically proportional to their log-odds validities. Each sampled cue casts a vote for the option it favors, and decision-makers also possess a baseline level of prior evidence for both options. The process continues until a relative evidence threshold is reached, producing compensatory behavior and graded choice probabilities based on the relative accumulated evidence. The inclusion of baseline evidence smooths out extreme probability predictions and naturally handles cases where no cues favor an option.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Log-odds weights for proportional sampling probabilities
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # Total sampling weight favoring each option
    v_a = np.sum(weights * (a > b))
    v_b = np.sum(weights * (b > a))
    
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    c = float(parameters["c"])
    
    # Add baseline constant to smooth out extreme ratios and avoid zero-evidence edge cases
    v_a_eff = v_a + c
    v_b_eff = v_b + c
    
    # Probability of reaching the relative threshold theta first via power-law rule
    log_p_a_unnorm = theta * np.log(v_a_eff)
    log_p_b_unnorm = theta * np.log(v_b_eff)
    max_log = max(log_p_a_unnorm, log_p_b_unnorm)
    exp_a = np.exp(log_p_a_unnorm - max_log)
    exp_b = np.exp(log_p_b_unnorm - max_log)
    p_a = exp_a / (exp_a + exp_b)
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


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


## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=8):** [0.95, 0.8, 0.9, 0.8, 0.85, 0.85, 0.75, 0.75]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0]

**Rationale:** To cleanly dissociate the advocated Probabilistic Evidence Accumulation theory from the competing Soft Compensatory Check model, we parametrically add matched pairs of cues favoring Option A and Option B equally, keeping the absolute WADD score difference strictly constant. In all trials, Option A is favored by the highest-validity cue (winning TTB) and also has a higher total WADD score. Across the trial sets, we add lower-validity cues that favor A and B equally. 

For the competing theory, the absolute WADD score difference remains constant, and TTB consistently favors Option A. Thus, it predicts a high, relatively constant probability of choosing A (only slightly mixing between two stable, high-probability strategies as opposing evidence increases). 

In contrast, the advocated theory evaluates the *ratio* of accumulated evidence (plus a baseline constant). As equal amounts of evidence are added to both options, the ratio of total evidence for A vs. B compresses towards 1. Therefore, the advocated theory uniquely predicts a graded decrease in the probability of choosing Option A, moving steadily towards 0.5 as total evidence increases, providing a sharp quantitative dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 1, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1, 0))|0': 0.6046712802768166, '((0, 0, 1, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1, 0))|1': 0.625, '((1, 1, 0, 0, 1, 0, 1, 0), (0, 0, 1, 1, 0, 1, 0, 1))|0': 0.3926796204247628, '((1, 1, 0, 0, 1, 0, 1, 0), (0, 0, 1, 1, 0, 1, 0, 1))|1': 0.3869698294709226, '((0, 0, 1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0))|0': 0.6421791167973586, '((0, 0, 1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0))|1': 0.6419856962557846, '((1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0, 0))|0': 0.330575256107171, '((1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0, 0))|1': 0.3717948717948718, '((0, 0, 1, 1, 0, 1, 0, 0), (1, 1, 0, 0, 1, 0, 0, 0))|0': 0.6174645390070922, '((0, 0, 1, 1, 0, 1, 0, 0), (1, 1, 0, 0, 1, 0, 0, 0))|1': 0.6226415094339622, '((1, 1, 0, 0, 1, 0, 0, 0), (0, 0, 1, 1, 0, 1, 0, 0))|0': 0.3659018987341772, '((1, 1, 0, 0, 1, 0, 0, 0), (0, 0, 1, 1, 0, 1, 0, 0))|1': 0.3772007042253521}
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
- Predicted under pi_6 (simulated): 0.0005 (var=0.0002)
- Predicted under pi_5 (simulated): 0.0260 (var=0.0014)
- Observed on real data: 0.0759 (var=0.0023)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=10):** [0.881, 0.818, 0.731, 0.731, 0.679, 0.679, 0.622, 0.622, 0.562, 0.562]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Rationale:** To cleanly dissociate the advocated Soft Compensatory Check model from the competing Probabilistic Evidence Accumulation theory, we construct a series of 'WADD-tied' trials where the total log-odds evidence for Option A and Option B are virtually identical. Across four critical levels of trials, we parametrically increase this tied total evidence (W). For the competing theory, because the evidence for A and B remains balanced (v_a ≈ v_b = W) in every trial, the ratio of evidence is always ~1, and the model predicts a flat line of exactly 50% choice probability across all levels of W. For the advocated theory, however, Option A is always the Take The Best (TTB) winner (since it wins the single highest validity cue to match the sum of B's multiple lower cues). As W increases, the 'opposing evidence' (the evidence favoring B) parametrically increases. This drives a smooth transition from strict TTB (which strongly favors A) to WADD (which predicts 50% due to the tie). Thus, the advocated theory uniquely predicts a graded decrease in the probability of choosing Option A towards 50% as W increases, providing a striking quantitative dissociation against the competing theory's flat 50% prediction.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 1))|0': 0.19366450683945285, '((0, 0, 0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 1))|1': 0.22551928783382788, '((0, 0, 0, 0, 0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0, 0, 1, 0, 0))|0': 0.8411274823830878, '((0, 0, 0, 0, 0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0, 0, 1, 0, 0))|1': 0.8057210965435042, '((0, 0, 1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1, 1, 0, 0))|0': 0.19487179487179487, '((0, 0, 1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1, 1, 0, 0))|1': 0.23414634146341465, '((0, 1, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1, 0, 0, 0, 0))|0': 0.20534550195567144, '((0, 1, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1, 0, 0, 0, 0))|1': 0.2586605080831409, '((0, 0, 1, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0))|0': 0.7243460764587525, '((0, 0, 1, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0))|1': 0.7837837837837838, '((0, 0, 0, 0, 0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 0, 0, 1, 0, 0, 0))|0': 0.8179530201342282, '((0, 0, 0, 0, 0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 0, 0, 1, 0, 0, 0))|1': 0.8228476821192053, '((0, 1, 0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 0))|0': 0.16074600355239788, '((0, 1, 0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 0))|1': 0.2135523613963039, '((0, 0, 0, 0, 0, 0, 1, 1, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0, 0, 0))|0': 0.8093023255813954, '((0, 0, 0, 0, 0, 0, 1, 1, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0, 0, 0))|1': 0.7810055865921788, '((0, 0, 0, 0, 1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0, 0))|0': 0.7798076923076923, '((0, 0, 0, 0, 1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0, 0))|1': 0.7955882352941176, '((1, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0, 0, 0, 0))|0': 0.30064308681672025, '((1, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0, 0, 0, 0))|1': 0.22890888638920134, '((0, 0, 0, 0, 0, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1, 0, 0, 0, 0))|0': 0.8041112454655381, '((0, 0, 0, 0, 0, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1, 0, 0, 0, 0))|1': 0.8423394787031151, '((1, 0, 1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 1))|0': 0.15270935960591134, '((1, 0, 1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 1))|1': 0.19373942470389172}
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
- Predicted under pi_6 (simulated): 0.0323 (var=0.0004)
- Predicted under pi_5 (simulated): 0.0010 (var=0.0005)
- Observed on real data: 0.0860 (var=0.0016)

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
**Predicted under pi_6:** 0.0756 (var=0.0009)
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
**Predicted under pi_6:** 0.0459 (var=0.0003)
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
**Predicted under pi_6:** 0.0944 (var=0.0012)
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
**Predicted under pi_6:** 0.0154 (var=0.0006)
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
**Predicted under pi_6:** 0.0315 (var=0.0004)
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
**Predicted under pi_6:** 0.0246 (var=0.0006)
**Predicted under pi_5:** 0.0017 (var=0.0011)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.0475 (var=0.0018)
**Predicted under pi_6:** 0.1100 (var=0.0015)
**Predicted under pi_5:** 0.0219 (var=0.0068)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]

**Metric**
```python
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

```

**Observed (real) value:** 0.2264 (var=0.0034)
**Predicted under pi_6:** 0.0027 (var=0.0003)
**Predicted under pi_5:** 0.0007 (var=0.0003)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 2 (pi_5) captures the observed data much better than Theory 1 (pi_6) across a majority of the experiments (e.g., Exps 3, 4, 5, and 7), as indicated by its predicted JSD closely matching the observed JSD. However, both models fail to fully capture the behavior in Experiments 8 and 10, where the observed JSD is quite high. Theory 1 consistently diverges from the observed data across most experiments. Therefore, Theory 1 should be replaced with a brand new theory that can capture both the sequential dependencies (which Theory 2 models well) and the specific feature interactions that both models currently miss.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory based on Decision Field Theory (DFT) or a Leaky Competing Accumulator (LCA) framework, incorporating sequential dependencies and dynamic attention switching between features. Unlike Theory 1's static proportional sampling and Theory 2's deterministic TTB-to-WADD threshold, this new theory should assume that decision-makers dynamically shift their attention between cues over time. The preference state is updated via a leaky accumulator with lateral inhibition, and the starting point of the accumulation is biased by the previous trial's choice (inertia). This naturally captures both sequential effects and graded compensatory behavior while providing a fundamentally different mechanism than soft strategy switching."
}
```

## Usage

```json
{
  "prompt_token_count": 23488,
  "candidates_token_count": 315,
  "total_token_count": 25378
}
```
