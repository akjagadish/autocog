# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5_1
People compare options through a conflict-gated dual-lexicographic/tallying system. A baseline mixture combines a weak forward Take-The-Best route, a moderate backward lexicographic route, and feature-win tallying. When the highest-validity and lowest-validity cues discriminate in opposite directions and the tally does not oppose the highest cue, the decision maker enters a conflict regime: tallying is suspended and the choice is made with high probability by a strong backward lexicographic scan over ascending validities. Subjects differ substantially in how strongly and deterministically they reverse on such conflict trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    def set_direction(indices):
        signs = []
        for j in indices:
            if a[j] > b[j]:
                signs.append(1)
            elif b[j] > a[j]:
                signs.append(-1)
        if not signs:
            return 0
        if all(s == signs[0] for s in signs):
            return signs[0]
        return 0

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_conflict_backward = float(parameters['beta_conflict_back'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward_base = lex_probabilities(ascending, beta_backward)
    p_backward_conflict = lex_probabilities(ascending, beta_conflict_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally_base = 1.0 - w_forward - w_backward
    if w_tally_base < 0.0:
        w_tally_base = 0.0
        denom = w_forward + w_backward
        w_forward /= denom
        w_backward /= denom

    v_max = np.max(validities)
    T = np.where(validities == v_max)[0]
    v_min = np.min(validities)
    L = np.where(validities == v_min)[0]

    top_dir = set_direction(T)
    low_dir = set_direction(L)
    tally_d = int(a_wins - b_wins)

    live_conflict = (
        top_dir != 0 and
        low_dir != 0 and
        top_dir == -low_dir and
        tally_d * top_dir >= 0
    )

    if live_conflict:
        w_forward_eff = 1.0 - float(parameters['w_conflict_back'])
        w_backward_eff = float(parameters['w_conflict_back'])
        w_tally_eff = 0.0
        p_backward = p_backward_conflict
    else:
        w_forward_eff = w_forward
        w_backward_eff = w_backward
        w_tally_eff = w_tally_base
        p_backward = p_backward_base

    weights = np.array([w_forward_eff, w_backward_eff, w_tally_eff], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5_1)

### DESIGN
**Validities (n_features=6):** [0.9, 0.78, 0.66, 0.58, 0.52, 0.52]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 6: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 10: A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 12: A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 13: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 14: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 15: A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  trial 16: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 17: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 20: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 21: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 22: A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 23: A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 24: A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  trial 25: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 26: A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 27: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 28: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 29: A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 30: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 31: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 32: A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Rationale:** Six experts with validities [0.90, 0.78, 0.66, 0.58, 0.52, 0.52]. Features 4 and 5 are tied for lowest validity; because both theories use stable argsort, feature 4 is the first cue in ascending/backward scans. Every trial sets feature 0 (highest) and feature 4 (first-lowest) to discriminate in opposite directions. The middle features fix the tally difference at +1 for top-A trials and -1 for top-B trials. The critical manipulation is feature 5, the second tied-lowest cue: in gate-active trials it is tied, so the advocated model's low_dir is nonzero and its conflict gate is live (tally does not oppose the top cue); in matched gate-inactive trials feature 5 discriminates in the same direction as the top cue, making low_dir mixed and switching the gate off. The competing stable-mixture model is blind to this manipulation: forward lex stops at feature 0, backward lex stops at feature 4 before feature 5, and tally sees the same d, so it predicts identical P(A) across gate-active and gate-inactive pairs within each top direction. With representative mid-range parameters (advocated: w_forward=.25, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2, w_conflict_back=.80, beta_conflict_back=2.5; competing: w_forward=.27, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2), top-A gate-active P(A)=.168 versus gate-inactive P(A)=.464 for the advocated model, while the competing model gives .457 for both; top-B gate-active P(A)=.832 versus gate-inactive P(A)=.536 for the advocated model, while the competing model gives .543 for both. This yields a gate-condition by top-direction interaction unique to the advocated conflict-gated architecture.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
This metric estimates the signed trial-type effect created by the tied low-validity cue manipulation. For each subject and each top-cue direction, it classifies trials by whether the advocated conflict gate is live using the highest-validity cue, the tied lowest-validity cues, and the tally direction. It then computes signed choose-A proportions, oriented by top-cue direction, and takes inactive minus active within each top direction before averaging across the two top directions. The advocated conflict-gated model predicts a large positive value because the gate reverses the choice away from the top cue when active, while the stable-mixture model predicts approximately zero because its forward scan stops before the tied low cue and its backward scan stops at the first tied low cue, making the two trial types indistinguishable.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.90, 0.78, 0.66, 0.58, 0.52, 0.52])
    top_idx = int(np.argmax(validities))
    low_idx = np.where(validities == np.min(validities))[0]

    subject_effects = []
    for sid, sub in data.groupby('subject_id'):
        groups = {1: {'active': [], 'inactive': []},
                  -1: {'active': [], 'inactive': []}}
        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            top_dir = int(np.sign(a[top_idx] - b[top_idx]))
            if top_dir == 0:
                continue
            lows = [int(np.sign(a[j] - b[j])) for j in low_idx]
            nonzero = [d for d in lows if d != 0]
            if len(nonzero) > 0 and all(d == nonzero[0] for d in nonzero):
                low_dir = nonzero[0]
            else:
                low_dir = 0
            tally_d = int(np.sum(a - b))
            gate = (low_dir != 0 and top_dir == -low_dir and tally_d * top_dir >= 0)
            y_a = 1.0 - float(row['response'])
            signed_y = float(top_dir) * y_a
            groups[top_dir]['active' if gate else 'inactive'].append(signed_y)

        dir_effects = []
        for d in (1, -1):
            active = groups[d]['active']
            inactive = groups[d]['inactive']
            if len(active) > 0 and len(inactive) > 0:
                dir_effects.append(float(np.mean(inactive) - np.mean(active)))
        if dir_effects:
            subject_effects.append(float(np.mean(dir_effects)))

    if not subject_effects:
        return 0.0
    return float(np.mean(subject_effects))

### RESULTS
- Predicted under pi_5_1 (simulated): 0.2417 (var=0.0120)
- Predicted under pi_4 (simulated): -0.0012 (var=0.0120)
- Observed on real data: -0.0283 (var=0.0032)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=8):** [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  trial 4: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 10: A=[1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0]
  trial 12: A=[1, 1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  trial 14: A=[1, 1, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  trial 15: A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 16: A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  trial 20: A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  trial 22: A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 1]
  trial 23: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  trial 24: A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  trial 25: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 26: A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1]
  trial 27: A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  trial 28: A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  trial 29: A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  trial 30: A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  trial 31: A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  trial 32: A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 1]

**Rationale:** This design isolates a route-invariant conflict-gate toggle using three tied lowest-validity cues. Features 5-7 all have validity 0.50. Feature 5 is always tied, so the backward lexicographic scan skips it and stops at feature 6, which always discriminates opposite the highest-validity cue (feature 0). Feature 7 is the critical gate trigger: in gate-active trials it is tied, so the lowest-validity set has a uniform direction and the competing model's conflict gate is live; in gate-inactive trials feature 7 discriminates in the same direction as the top cue, making the lowest-validity set mixed and switching the gate off. A middle feature is adjusted to hold the feature-win tally difference constant (d=+1 for top-A, d=-1 for top-B) across the gate manipulation. The advocated stable-mixture model is blind to feature 7 because forward lex stops at feature 0, backward lex stops at feature 6, and tally d is unchanged; it predicts identical P(A) for matched gate-active and gate-inactive pairs. The competing conflict-gated model predicts a large gate-condition gap: with mid-range parameters, top-A gate-active P(A)=0.168 versus gate-inactive P(A)=0.457; top-B gate-active P(A)=0.832 versus gate-inactive P(A)=0.543, while the advocated model gives 0.464 and 0.536 in both gate conditions. The 32 unique pairs are balanced over top direction, gate condition, and middle-cue patterns, repeated K=3 for 96 trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the predicted conflict-gate effect. Feature 0 is the unique highest-validity cue, so every response is recoded as choose-top: 1 if the subject chose the option favored by feature 0, otherwise 0. Gate-active trials are those where feature 7 is tied; in this design that is exactly when the lowest-validity cue set is uniform and the competing theory's conflict gate switches on, producing a strong backward-lexicographic shift away from the top cue. Gate-inactive trials are those where feature 7 discriminates in the same direction as feature 0, so the competing theory remains in its baseline mixture. The metric is the choose-top rate on gate-inactive trials minus the choose-top rate on gate-active trials. Under the advocated stable-mixture theory, forward lexicographic scanning stops at feature 0, backward lexicographic scanning stops at feature 6, and tally differences are balanced across the gate manipulation, so feature 7 is irrelevant and the expected metric is zero. Under the competing conflict-gated theory, tied feature 7 sharply lowers choose-top responding on gate-active trials, so the metric is expected to be large and positive. The contrast is within-subject and uses roughly 48 trials per condition, keeping per-subject variance manageable.

Source:
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a0 = data['option_a_ratings'].apply(lambda x: x[0]).astype(int)
    b0 = data['option_b_ratings'].apply(lambda x: x[0]).astype(int)
    a7 = data['option_a_ratings'].apply(lambda x: x[7]).astype(int)
    b7 = data['option_b_ratings'].apply(lambda x: x[7]).astype(int)

    top_is_a = a0 > b0
    gate_active = a7 == b7

    choose_top = ((top_is_a & (data['response'] == 0)) |
                  (~top_is_a & (data['response'] == 1))).astype(float)

    if gate_active.sum() == 0 or (~gate_active).sum() == 0:
        return 0.0

    return float(choose_top[~gate_active].mean() - choose_top[gate_active].mean())


### RESULTS
- Predicted under pi_5_1 (simulated): 0.2717 (var=0.0167)
- Predicted under pi_4 (simulated): 0.0100 (var=0.0111)
- Observed on real data: -0.0008 (var=0.0091)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        # Cues are in descending-validity order, so the first discriminating
        # feature is the Take The Best cue.
        for j in range(len(a)):
            if a[j] > b[j]:
                pred = 0
                break
            if b[j] > a[j]:
                pred = 1
                break
        if pred is not None:
            matches.append(1.0 if r == pred else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2487 (var=0.0065)
**Predicted under pi_5_1:** 0.2360 (var=0.0031)
**Predicted under pi_4:** 0.2306 (var=0.0024)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        response_a = 1.0 if int(row["response"]) == 0 else 0.0

        # Take The Best style second-cue prediction.
        # By design, feature 0 is tied on every trial, so feature 1 is
        # the first discriminating cue.
        if a[1] > b[1]:
            second_wins_a = True
        elif b[1] > a[1]:
            second_wins_a = False
        else:
            second_wins_a = True

        second_score = response_a if second_wins_a else (1.0 - response_a)

        # Tallying total score. Feature-specific ties contribute zero.
        wins_a = float(np.sum(a > b))
        wins_b = float(np.sum(b > a))
        tally_d = wins_a - wins_b

        if tally_d > 0.0:
            tally_score = response_a
        elif tally_d < 0.0:
            tally_score = 1.0 - response_a
        else:
            # Tallying has no preference on a tied tally.
            tally_score = 0.5

        scores.append(tally_score - second_score)

    return float(np.mean(scores)) if scores else 0.0
```

**Observed (real) value:** 0.4196 (var=0.0128)
**Predicted under pi_5_1:** 0.4519 (var=0.0082)
**Predicted under pi_4:** 0.4335 (var=0.0093)

### Experiment 5
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    a1 = data["option_a_ratings"].apply(lambda x: x[1])
    b1 = data["option_b_ratings"].apply(lambda x: x[1])
    f1_a = a1 > b1
    f1_b = b1 > a1
    chose_a = data["response"].astype(int) == 0
    if f1_a.sum() == 0 or f1_b.sum() == 0:
        return 0.0
    return float(chose_a[f1_a].mean() - chose_a[f1_b].mean())
```

**Observed (real) value:** -0.1300 (var=0.0159)
**Predicted under pi_5_1:** -0.1329 (var=0.0102)
**Predicted under pi_4:** -0.1325 (var=0.0113)

### Experiment 6
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.88, 0.80, 0.72, 0.65, 0.60, 0.55, 0.52])
    order = np.argsort(-validities)
    pos = []
    neg = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        first = 'A'
        found = False
        for j in order:
            if a[j] > b[j]:
                first = 'A'
                found = True
                break
            if b[j] > a[j]:
                first = 'B'
                found = True
                break
        if not found:
            continue
        chose_a = 1.0 if row.response == 0 else 0.0
        if first == 'A':
            pos.append(chose_a)
        else:
            neg.append(chose_a)
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))

```

**Observed (real) value:** -0.2742 (var=0.0069)
**Predicted under pi_5_1:** -0.3025 (var=0.0121)
**Predicted under pi_4:** -0.2879 (var=0.0081)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    a0 = np.array([int(r[0]) for r in data['option_a_ratings']])
    b0 = np.array([int(r[0]) for r in data['option_b_ratings']])
    choose_a = (np.asarray(data['response'], dtype=int) == 0).astype(float)
    top_a = a0 > b0
    top_b = ~top_a
    if not np.any(top_a) or not np.any(top_b):
        return 0.0
    return float(choose_a[top_a].mean() - choose_a[top_b].mean())
```

**Observed (real) value:** -0.6583 (var=0.0598)
**Predicted under pi_5_1:** -0.5975 (var=0.0258)
**Predicted under pi_4:** -0.2833 (var=0.0124)

### Experiment 8
**Design**
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    if len(data) == 0:
        return 0.0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=int)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=int)
    diff = a - b

    top_cue = diff[:, 0]
    tally_diff = (diff > 0).sum(axis=1) - (diff < 0).sum(axis=1)

    chose_a = (data['response'].to_numpy() == 0).astype(float)

    top_a_plus = (top_cue > 0) & (tally_diff >= 1)
    top_b_minus = (top_cue < 0) & (tally_diff <= -1)

    mean_top_a = np.mean(chose_a[top_a_plus]) if np.any(top_a_plus) else 0.0
    mean_top_b = np.mean(chose_a[top_b_minus]) if np.any(top_b_minus) else 0.0

    return float(mean_top_a - mean_top_b)
```

**Observed (real) value:** -0.6711 (var=0.0489)
**Predicted under pi_5_1:** -0.6233 (var=0.0227)
**Predicted under pi_4:** -0.0722 (var=0.0187)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    choices = []
    for _, row in data.iterrows():
        a = [int(x) for x in row['option_a_ratings']]
        b = [int(x) for x in row['option_b_ratings']]
        if a[0] == b[0]:
            continue
        top = 0 if a[0] > b[0] else 1
        top_vec = a if top == 0 else b
        other_vec = b if top == 0 else a
        d_top = 0
        for i in range(len(top_vec)):
            if top_vec[i] > other_vec[i]:
                d_top += 1
            elif other_vec[i] > top_vec[i]:
                d_top -= 1
        if d_top >= 0:
            choices.append(1.0 if int(row['response']) == top else 0.0)
    if not choices:
        return 0.5
    return float(np.mean(choices))
```

**Observed (real) value:** 0.1553 (var=0.0159)
**Predicted under pi_5_1:** 0.1960 (var=0.0071)
**Predicted under pi_4:** 0.4160 (var=0.0041)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    validities = np.array([0.9, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5], dtype=float)
    j_hi = int(np.argmax(validities))
    j_lo = int(np.argmin(validities))

    a_arr = np.array([np.asarray(x, dtype=float) for x in data['option_a_ratings']])
    b_arr = np.array([np.asarray(x, dtype=float) for x in data['option_b_ratings']])
    resp = np.asarray(data['response'], dtype=float)

    hi_diff = a_arr[:, j_hi] - b_arr[:, j_hi]
    lo_diff = a_arr[:, j_lo] - b_arr[:, j_lo]
    d = (a_arr > b_arr).sum(axis=1) - (b_arr > a_arr).sum(axis=1)

    hi_dir = np.where(hi_diff > 0, 1, np.where(hi_diff < 0, -1, 0))
    lo_dir = np.where(lo_diff > 0, 1, np.where(lo_diff < 0, -1, 0))
    live = (hi_dir != 0) & (lo_dir != 0) & (hi_dir == -lo_dir) & (d * hi_dir >= 0)

    y = np.where(lo_dir > 0, (resp == 0).astype(float),
                 np.where(lo_dir < 0, (resp == 1).astype(float), 0.5))

    if np.sum(live) == 0 or np.sum(~live) == 0:
        return 0.0
    return float(np.mean(y[live]) - np.mean(y[~live]))
```

**Observed (real) value:** 0.0150 (var=0.0027)
**Predicted under pi_5_1:** -0.0021 (var=0.0064)
**Predicted under pi_4:** -0.0162 (var=0.0110)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Exp1 and Exp2 were designed to toggle pi_5_1's conflict gate through a second tied lowest-validity cue. The real data show no gate effect: pi_5_1 predicted 0.2417 and 0.2717, while observed values were -0.0283 and -0.0008, very close to pi_4's near-zero predictions. That is a strong refutation of pi_5_1's current tied-cue gate implementation, not necessarily of gating altogether. In the older experiments, the opposite is true: Exp7, Exp8, and Exp9 show strong anti-top-cue or anti-tally-aligned responding with observed values of -0.6583, -0.6711, and 0.1553. pi_5_1 is much closer on these (-0.5975, -0.6233, 0.1960) than pi_4 (-0.2833, -0.0722, 0.4160). Exp3 through Exp6 are fit about equally well by both theories, and Exp10 is near zero for both. Thus pi_5_1's conflict shift is needed to explain Exp7-9, but its current aggregation over all tied lowest-validity cues creates a spurious effect in Exp1-2. The best next step is a model-level repair of pi_5_1: keep the gated architecture but make conflict detection require unique extreme cues, so tied low-validity cues no longer toggle the gate.",
  "verdict": "new_model",
  "target_theory_idx": 1,
  "recommendation": "Regenerate pi_5_1's predict function so conflict gating requires a unique highest-validity cue and a unique lowest-validity cue. If either extreme is tied, set live_conflict=False and use the baseline mixture; do not aggregate all tied extrema into top_dir or low_dir. This makes the extra tied low cue in Exp1 and Exp2 irrelevant, predicting gate metrics near zero, while keeping the conflict shift active in Exp7-9 and avoiding gate activation in Exp10 where the top validity is tied. Then retune parameter ranges: keep the baseline forward lexicographic route weak but allow moderate backward and tally routes; strengthen the conflict regime with w_conflict_back around 0.75 to 0.95 and beta_conflict_back around 1.8 to 4.0 to close the remaining magnitude gaps in Exp7-9. Also increase between-subject spread in w_conflict_back and beta_conflict_back because observed variance in Exp7 and Exp8 is larger than current pi_5_1 predictions. Keep beta_forward low from 0 to 0.6, beta_backward moderate from about 1.0 to 2.5, and beta_tally from about 1.5 to 3.0. Validate that Exp1 and Exp2 gate metrics stay near zero, Exp3-6 preserve their current close fit, and Exp7-9 move closer to the observed values."
}
```

## Usage

```json
{
  "input_tokens": 22844,
  "output_tokens": 26425,
  "reasoning_tokens": 25776,
  "provider": "Baidu"
}
```
