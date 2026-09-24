# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
People choose by a confidence-gated unweighted signed tally. On clear tallies, choice follows the tally through a moderate softmax plus a small lapse. When the tally is tied or nearly tied, people may switch to a restricted sub-tally only if the advertised cue validities themselves flag a small coherent elite (two to three cues separated from the rest, or a large validity gap with at most six top cues). Critically, that restricted block is not defined by validity alone: cues are re-ranked by a mixture of advertised validity and serial display position, where later cues receive exponentially greater recency weight. This allows late relevant cues to dominate early equally-valid cue blocks, producing reversals like Experiment 10 while preserving validity use in Experiments 3 and 4. Large-feature environments with small tallies are treated as low confidence and heavily regularized with near-chance lapse, preventing deterministic regression blow-ups.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))
    positions = np.arange(n_features, dtype=float)

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    def lapse_mix(core, eps):
        probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
        probs = np.clip(probs, 0.005, 0.995)
        total = float(np.sum(probs))
        if total <= 1e-12:
            return np.array([0.5, 0.5])
        return probs / total

    vmax = float(np.max(validities))
    top_tol = float(parameters['top_tol'])
    top_mask = validities >= (vmax - top_tol)
    top_size = int(np.sum(top_mask))

    if top_size < n_features:
        second_max = float(np.max(validities[~top_mask]))
    else:
        second_max = vmax
    validity_gap = vmax - second_max

    gap_hi = float(parameters['gap_hi'])
    gap_lo = float(parameters['gap_lo'])
    size_min = int(parameters['small_size_min'])
    size_max = int(parameters['small_size_max'])
    block_max = int(parameters['block_max'])

    small_elite = (size_min <= top_size <= size_max) and (validity_gap >= gap_lo)
    block_clear = (validity_gap >= gap_hi) and (top_size <= block_max)
    salience = 1.0 if (small_elite or block_clear) else 0.0

    tau_tie = float(parameters['tau_tie'])

    if salience > 0.5 and abs(tally) <= tau_tie:
        rho = float(parameters['rho'])

        # Stable recency-weighted cue relevance. Subtracting the max position
        # prevents overflow while preserving the ranking and normalized weights.
        combined = np.asarray(validities, dtype=float) * np.exp(
            rho * (positions - positions.max())
        )

        order = np.argsort(-combined)
        selected = order[:top_size]
        if selected.size == 0:
            return np.array([0.5, 0.5])

        sel_weights = combined[selected]
        denom = float(np.sum(sel_weights))
        if denom <= 1e-12:
            weights = np.ones_like(sel_weights)
        else:
            weights = sel_weights / denom * float(selected.size)

        sub_tally = float(np.dot(weights, diff[selected]))

        beta_sub = float(parameters['beta_sub'])
        eps_sub = float(parameters['epsilon_sub'])

        if abs(sub_tally) > 1e-12:
            core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
            return lapse_mix(core, eps_sub)

        return np.array([0.5, 0.5])

    near_bound = float(parameters['near_tally_bound'])
    n_low_conf = int(parameters['n_low_conf'])

    if abs(tally) <= near_bound and n_features >= n_low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = float(parameters['beta_tally'])
        eps = float(parameters['epsilon_tally'])

    core = stable_softmax(np.array([tally, -tally]), beta)
    return lapse_mix(core, eps)


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))


## THEORY 2 — pi_6
Tally-first choice with a salience-gated and size-regularized validity tie-break. On every trial the unweighted signed tally of the feature differences is the default choice signal. When the tally is clear, choice follows it through a moderate softmax plus an explicit lapse. When the tally is tied or near-tied, validity is allowed to matter only if the task's validity distribution itself advertises a clear elite: either a small elite block of two to three maximally valid cues separated from the remaining cues, or a large validity gap combined with a bounded elite-block size. In those gated cases people compute a restricted sub-tally over the elite cues and choose its favored option stochastically. Larger elite blocks receive an additional lapse, preventing overconfident validity-following for wide top tiers. In large stimulus arrays, near-zero tallies are treated as low-confidence and are heavily regularized with extra choice noise, so deterministic regression blows-up do not occur.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))

    vmax = float(np.max(validities))
    top_tol = float(parameters['top_tol'])
    top_mask = validities >= (vmax - top_tol)
    top_size = int(np.sum(top_mask))

    if top_size < n_features:
        second_max = float(np.max(validities[~top_mask]))
    else:
        second_max = vmax
    validity_gap = vmax - second_max

    gap_hi = float(parameters['salience_gap_hi'])
    gap_lo = float(parameters['salience_gap_lo'])
    size_min = int(parameters['top_size_small_min'])
    size_max = int(parameters['top_size_small_max'])
    block_max = int(parameters['top_size_block_max'])

    block_clear = (validity_gap >= gap_hi) and (top_size <= block_max)
    small_elite = (size_min <= top_size <= size_max) and (validity_gap >= gap_lo)
    salience = 1.0 if (block_clear or small_elite) else 0.0

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    tau_tie = float(parameters['tau_tie'])

    if salience > 0.5 and abs(tally) <= tau_tie:
        top_diff = diff[top_mask]
        sub_a = float(np.sum(top_diff > 0.0))
        sub_b = float(np.sum(top_diff < 0.0))
        sub_tally = sub_a - sub_b

        beta_sub = float(parameters['beta_sub'])
        eps_sub = float(parameters['epsilon_sub'])
        eps_sub_size = float(parameters['epsilon_sub_size'])
        eps_sub_eff = eps_sub + eps_sub_size * max(0.0, float(top_size) - 3.0)
        eps_sub_eff = float(np.clip(eps_sub_eff, 0.0, 0.95))

        if abs(sub_tally) > 1e-12:
            core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
            probs = (1.0 - eps_sub_eff) * core + eps_sub_eff * np.array([0.5, 0.5])
        else:
            probs = np.array([0.5, 0.5])
        return probs

    near_bound = float(parameters['near_tally_bound'])
    n_low_conf = int(parameters['n_low_conf'])

    if abs(tally) <= near_bound and n_features >= n_low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = float(parameters['beta_tally'])
        eps = float(parameters['epsilon_tally'])

    core = stable_softmax(np.array([tally, -tally]), beta)
    probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.array([0.5, 0.5])
    return probs / total


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))


## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=12):** [1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 11: A=[1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Rationale:** All 12 unique pairs are unweighted tally ties, d = sum(A - B) = 0, so both fallback gates open. The validities define a three-cue elite: the first three experts have validity 1.0 and the remaining nine have 0.8, giving a gap of 0.20 and top_size = 3, which meets both theories' small-elite conditions. The first three features define T = #top A wins - #top B wins among positions 0-2; the last three define L = #late A wins - #late B wins among positions 9-11. Middle features 3-8 are tally-balancing fillers. For the advocated model, combined relevance = validity * exp(rho * (position - max_position)) makes the final three positions dominate every earlier 0.8 cue for rho in [0.60, 1.00], so its restricted block is positions 9-11 and its weighted sub-tally is exactly +3 on L=+3 trials and -3 on L=-3 trials. The competing model ignores position and uses the three 1.0 experts, so its sub-tally equals T. The design is a 4 x 2 factorial over T in {+3,+1,-1,-3} and L in {+3,-3}; trials 1-8 are sign-reversal cells. Critical predictions: in T=+3,L=-3 cells, competing p(A) is roughly 0.85-0.91 while advocated p(A) is roughly 0.03-0.07; in T=-3,L=+3 cells these reverse. Same-sign cells provide magnitude controls. A validity-only fallback cannot produce an L effect within a fixed T, and a recency-weighted fallback cannot produce a T effect within a fixed L, giving a quantitative double dissociation. The 12 unique pairs are repeated 8 times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric measures the double-dissociation reversal contrast using the two critical sign-reversal cells: top tally T=+3 with late tally L=-3, and top tally T=-3 with late tally L=+3. All trials are unweighted tally ties, so both theories enter their gated sub-tally path rather than using the full tally. The validity-only theory follows the three high-validity early cues, producing a high A-choice rate when T=+3 and a low A-choice rate when T=-3. The advocated recency-weighted theory follows the late three cues, producing the opposite pattern. Subtracting the A-choice rate in the T=+3,L=-3 cell from the A-choice rate in the T=-3,L=+3 cell therefore yields approximately +0.9 under the advocated theory and approximately -0.8 under the competing theory. Each subject provides 16 trials per critical cell, so the binomial variability of this contrast is modest, giving a large and stable between-theory separation.

Source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    chose_a_c1 = []
    chose_a_c2 = []

    for _, row in data.iterrows():
        a = [int(v) for v in row['option_a_ratings']]
        b = [int(v) for v in row['option_b_ratings']]
        diff = [a[j] - b[j] for j in range(len(a))]

        t_top = int(sum(diff[:3]))
        l_late = int(sum(diff[-3:]))
        chose_a = 1.0 if int(row['response']) == 0 else 0.0

        if t_top == 3 and l_late == -3:
            chose_a_c1.append(chose_a)
        elif t_top == -3 and l_late == 3:
            chose_a_c2.append(chose_a)

    if len(chose_a_c1) == 0 or len(chose_a_c2) == 0:
        return 0.0

    return float(np.mean(chose_a_c2) - np.mean(chose_a_c1))

### RESULTS
- Predicted under pi_7 (simulated): 0.8875 (var=0.0059)
- Predicted under pi_6 (simulated): -0.7338 (var=0.0218)
- Observed on real data: 0.0125 (var=0.0153)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=8):** [1.0, 0.996, 0.993, 0.991, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Rationale:** Eight experts are used. The first four validities (1.00, 0.996, 0.993, 0.991) fall within the advocated model's top_tol of 0.01, so it groups them into a top block of size 4. The competing model's top_tol of 0.001 leaves only the 1.00 expert in its top block, so its top_size is 1. With the next highest validity at 0.50, the advocated model sees a large validity gap of 0.50 with a bounded top block of size 4, which activates its block_clear gated tie-break on every trial. The competing model instead sees a validity gap of only 0.004 and no eligible small elite, so its validity gate stays closed. All 12 unique pairs have an exact unweighted tally tie, d = sum(A - B) = 0, so the competing model remains in default tally mode and predicts p(A) = 0.5 for every pair. The advocated model uses the gated top-block sub-tally T over the first four experts. The unique T values are +4, +3, +3, +2, +2, +1, -1, -2, -2, -3, -3, -4. With beta_sub between 0.35 and 0.50 and the top_size=4 extra lapse of about 0.24 to 0.34, advocated p(A) is ordered by T: roughly 0.61 to 0.68 for T = +1, rising to roughly 0.79 to 0.87 for T = +4, with mirrored values below 0.5 for negative T. The competing model is flat at 0.5 for all 12 pairs. Thus the design produces a quantitative T-ordered psychometric function for the advocated theory against a flat chance prediction for the competing theory. The 12 unique pairs are repeated 8 times for 96 trials per subject.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The validities [1.0, 0.996, 0.993, 0.991, 0.5, 0.5, 0.5, 0.5] make the first four cues the top block under the advocated theory's top_tol of 0.01. Every unique pair has an unweighted tally of 0, so the competing theory never opens its validity gate and predicts p(A) = 0.5 for all trials. The advocated theory instead uses the sub_tally over the first four cues, with values +4, +3, +3, +2, +2, +1, -1, -2, -2, -3, -3, -4. This metric is a weighted validity-following score: sum(T * chose_A) / sum(T for T > 0), where T is the first-four-cue sub_tally and chose_A is 1 when A is chosen. Under competing or random choices the positive and negative T contributions cancel in expectation, giving about 0. Under the advocated theory the score is expected to be much larger, roughly 0.48 to 0.65, because A choices are more common for positive T and less common for negative T. With 96 balanced trials per subject, this score has low between-subject variance and should sharply separate the two theories.

Source:
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    def top_sum(ratings):
        ratings = list(ratings)
        return int(ratings[0]) + int(ratings[1]) + int(ratings[2]) + int(ratings[3])

    a_sum = data['option_a_ratings'].apply(top_sum)
    b_sum = data['option_b_ratings'].apply(top_sum)
    sub_tally = a_sum - b_sum

    chose_a = 1 - data['response']

    numerator = float((sub_tally * chose_a).sum())
    denominator = float(sub_tally[sub_tally > 0].sum())

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


### RESULTS
- Predicted under pi_7 (simulated): 0.0103 (var=0.0147)
- Predicted under pi_6 (simulated): 0.5927 (var=0.0131)
- Observed on real data: -0.6113 (var=0.0332)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.84, 0.76, 0.68, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind='stable')
    A = np.array([np.asarray(row, dtype=float) for row in data['option_a_ratings']])
    B = np.array([np.asarray(row, dtype=float) for row in data['option_b_ratings']])
    resp_is_a = (data['response'].to_numpy(dtype=float) == 0.0).astype(float)

    winner = np.full(A.shape[0], -1, dtype=int)
    unresolved = np.ones(A.shape[0], dtype=bool)
    for j in cue_order:
        diff = A[:, j] - B[:, j]
        first = unresolved & (diff != 0)
        winner[first] = np.where(diff[first] > 0, 0, 1)
        unresolved[first] = False

    tally = (A > B).sum(axis=1) - (B > A).sum(axis=1)
    side = np.where(winner == 0, 1.0, np.where(winner == 1, -1.0, 0.0))
    signed_weight = side * (np.abs(tally) + 1.0)

    numerator = float(np.sum(signed_weight * resp_is_a))
    denom = float(np.sum(signed_weight[signed_weight > 0]))
    if denom == 0.0:
        return 0.0
    return numerator / denom

```

**Observed (real) value:** -0.6137 (var=0.0420)
**Predicted under pi_7:** -0.7377 (var=0.0029)
**Predicted under pi_6:** -0.7690 (var=0.0029)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    resp = data['response'].astype(int).tolist()

    # Highest validity is at display index 4: [0.57, 0.7, 0.5, 0.77, 0.92, 0.84, 0.63, 0.52]
    top_idx = 4

    xs = []
    ys = []

    for a, b, r in zip(a_col, b_col, resp):
        if a[top_idx] == b[top_idx]:
            continue

        top_is_a = a[top_idx] > b[top_idx]

        a_wins = 0
        b_wins = 0
        for fa, fb in zip(a, b):
            if fa > fb:
                a_wins += 1
            elif fb > fa:
                b_wins += 1

        d = a_wins - b_wins
        if d == 0:
            continue

        # Signed tally advantage for the option favored by the top cue.
        signed_tally_for_top = d if top_is_a else -d

        # 1 if the subject chose the option favored by the top cue, else 0.
        top_chosen = 1.0 if ((top_is_a and r == 0) or ((not top_is_a) and r == 1)) else 0.0

        xs.append(float(signed_tally_for_top))
        ys.append(top_chosen)

    if len(xs) < 2:
        return 0.0

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    xm = x.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        return 0.0

    slope = float(np.sum((x - xm) * (y - y.mean())) / sxx)
    return slope
```

**Observed (real) value:** 0.0659 (var=0.0002)
**Predicted under pi_7:** 0.0864 (var=0.0000)
**Predicted under pi_6:** 0.0875 (var=0.0000)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    centered = validities - validities.mean()

    cap = 3.0
    subj_vals = []

    for _, sub in data.groupby('subject_id'):
        a = np.asarray(sub['option_a_ratings'].tolist(), dtype=float)
        b = np.asarray(sub['option_b_ratings'].tolist(), dtype=float)
        diff = a - b
        tally = diff.sum(axis=1)
        s = diff @ centered

        mask = np.abs(tally) < 1e-9
        if int(mask.sum()) < 4:
            continue

        s0 = s[mask]
        resp = sub['response'].to_numpy(dtype=float)
        y = (resp[mask] == 0.0).astype(float)

        score = float(np.sum(s0 * (y - 0.5)))
        null_sd = float(np.sqrt(0.25 * np.sum(s0 * s0)))
        if null_sd <= 1e-12:
            continue

        z = score / null_sd
        subj_vals.append(float(min(abs(z), cap)))

    if not subj_vals:
        return 0.0
    return float(np.mean(subj_vals))

```

**Observed (real) value:** 2.6289 (var=0.2854)
**Predicted under pi_7:** 3.0000 (var=0.0000)
**Predicted under pi_6:** 2.2500 (var=0.6567)

### Experiment 6
**Design**
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0])
    centered = validities - validities.mean()

    def one_subject_stat(sub):
        A = np.array(sub['option_a_ratings'].tolist(), dtype=float)
        B = np.array(sub['option_b_ratings'].tolist(), dtype=float)
        y = (sub['response'].to_numpy() == 0).astype(float)
        diff = A - B
        tie_trial_mask = np.isclose(np.sum(diff, axis=1), 0.0)
        s = diff.dot(centered)
        score = np.sum(s[tie_trial_mask] * y[tie_trial_mask])
        null_var = 0.25 * np.sum(s[tie_trial_mask] ** 2)
        if null_var <= 0.0:
            return 0.0
        z = score / np.sqrt(null_var)
        return float(np.abs(z) - 0.7978845608028654)

    if data['subject_id'].nunique() == 1:
        return one_subject_stat(data)

    values = [one_subject_stat(sub) for _, sub in data.groupby('subject_id', sort=False)]
    if not values:
        return 0.0
    return float(np.mean(values))

```

**Observed (real) value:** 4.2015 (var=2.0036)
**Predicted under pi_7:** 5.1889 (var=0.1789)
**Predicted under pi_6:** 4.1985 (var=0.4476)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    diff = a - b
    total_tally = diff.sum(axis=1)
    high_tally = diff[:, 8:].sum(axis=1)
    chose_a = (data['response'].to_numpy() == 0).astype(float)
    subject = data['subject_id'].to_numpy()

    per_subject = []
    for sid in np.unique(subject):
        idx = subject == sid
        tie = total_tally[idx] == 0.0
        high_pos = high_tally[idx] > 0
        high_neg = high_tally[idx] < 0
        pos_trials = tie & high_pos
        neg_trials = tie & high_neg
        if pos_trials.sum() > 0 and neg_trials.sum() > 0:
            diff_rate = float(chose_a[idx][pos_trials].mean() - chose_a[idx][neg_trials].mean())
            per_subject.append(diff_rate)

    if not per_subject:
        return float('nan')

    return float(np.mean(per_subject))
```

**Observed (real) value:** -0.0467 (var=0.0156)
**Predicted under pi_7:** 0.0100 (var=0.0410)
**Predicted under pi_6:** -0.0183 (var=0.0526)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    validities = np.array([0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0], dtype=float)
    weights = (validities - 0.5) ** 6.0

    a = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b = np.array(data['option_b_ratings'].tolist(), dtype=float)
    resp = data['response'].to_numpy(dtype=float)

    diff = a - b
    tally = (a > b).sum(axis=1) - (b > a).sum(axis=1)
    tie_mask = tally == 0

    if not np.any(tie_mask):
        return 0.0

    v = diff[tie_mask] @ weights
    y = resp[tie_mask]
    signed_accuracy = np.sign(v) * (1.0 - 2.0 * y)
    abs_v = np.abs(v)
    total_weight = float(abs_v.sum())

    if total_weight <= 0.0:
        return 0.0

    return float(np.sum(abs_v * signed_accuracy) / total_weight)

```

**Observed (real) value:** 0.0204 (var=0.0176)
**Predicted under pi_7:** -0.0061 (var=0.0427)
**Predicted under pi_6:** -0.0098 (var=0.0375)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df["_order"] = df.groupby("subject_id", sort=False).cumcount()
    df = df[df["_order"] >= 12]
    if df.empty:
        return 0.0

    a = np.asarray(df["option_a_ratings"].tolist(), dtype=float)
    b = np.asarray(df["option_b_ratings"].tolist(), dtype=float)
    d = (a - b).sum(axis=1)
    resp = df["response"].to_numpy().astype(int)

    v = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.85, 0.75, 0.65, 0.95, 0.99])
    w = v - 0.5

    ties = np.abs(d) < 0.5
    non_ties = ~ties

    if ties.any():
        tie_scores = np.dot(a[ties] - b[ties], w)
        validity_side = np.where(tie_scores > 0, 0, 1)
        tie_rate = float(np.mean(resp[ties] == validity_side))
    else:
        tie_rate = 0.5

    if non_ties.any():
        tally_side = np.where(d[non_ties] > 0, 0, 1)
        non_tie_rate = float(np.mean(resp[non_ties] == tally_side))
    else:
        non_tie_rate = 0.5

    return tie_rate - non_tie_rate

```

**Observed (real) value:** -0.5522 (var=0.0900)
**Predicted under pi_7:** -0.3947 (var=0.0078)
**Predicted under pi_6:** -0.3110 (var=0.0242)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    subjects = list(data['subject_id'].unique())
    z_scores = []
    for sid in subjects:
        sub = data[data['subject_id'] == sid]
        d_vals = []
        s_vals = []
        y_vals = []
        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            n = a.size
            if n < 2:
                continue
            half = n // 2
            diff = a - b
            high = float(np.sum(diff[:half]))
            low = float(np.sum(diff[half:]))
            d = high + low
            s = 0.25 * (high - low)
            y = 1.0 - float(row['response'])
            d_vals.append(d)
            s_vals.append(s)
            y_vals.append(y)
        if len(y_vals) < 10:
            continue
        X = np.column_stack([
            np.ones(len(y_vals)),
            np.asarray(d_vals, dtype=float),
            np.asarray(s_vals, dtype=float)
        ])
        y = np.asarray(y_vals, dtype=float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        beta = np.asarray(beta, dtype=float).ravel()
        p_hat = X @ beta
        p_hat = np.clip(p_hat, 0.005, 0.995)
        resid = y - p_hat
        XtX = X.T @ X
        try:
            bread = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(XtX)
        meat = (X * (resid ** 2)[:, None]).T @ X
        cov = bread @ meat @ bread
        se_d = float(np.sqrt(max(cov[1, 1], 1e-12)))
        z_d = float(beta[1]) / se_d if se_d > 0 else 0.0
        z_scores.append(z_d)
    if len(z_scores) == 0:
        return 0.0
    if len(subjects) > 1:
        return float(np.var(np.asarray(z_scores, dtype=float), ddof=0))
    return float(z_scores[0])
```

**Observed (real) value:** 0.9276 (var=0.9276)
**Predicted under pi_7:** 0.7907 (var=0.7907)
**Predicted under pi_6:** 1.5589 (var=1.5589)

### Experiment 11
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0
    a = np.asarray(list(data['option_a_ratings']), dtype=float)
    b = np.asarray(list(data['option_b_ratings']), dtype=float)
    tally = np.sum(a - b, axis=1)
    resp = data['response'].to_numpy(dtype=float)
    pos_mask = tally > 0.0
    neg_mask = tally < 0.0
    pos_mean = float(np.mean(resp[pos_mask])) if np.any(pos_mask) else 0.5
    neg_mean = float(np.mean(resp[neg_mask])) if np.any(neg_mask) else 0.5
    return neg_mean - pos_mean

```

**Observed (real) value:** 0.0300 (var=0.0026)
**Predicted under pi_7:** 0.0492 (var=0.0092)
**Predicted under pi_6:** 0.0750 (var=0.0143)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data):
    # Top three features are the three perfectly valid experts in this design.
    a_top = np.array([float(r[0] + r[1] + r[2]) for r in data['option_a_ratings']])
    b_top = np.array([float(r[0] + r[1] + r[2]) for r in data['option_b_ratings']])
    top_tally = a_top - b_top

    chose_a = 1.0 - np.asarray(data['response'], dtype=float)

    mask_pos3 = top_tally == 3
    mask_neg3 = top_tally == -3

    if mask_pos3.sum() == 0 or mask_neg3.sum() == 0:
        return 0.0

    p_a_given_top_plus3 = chose_a[mask_pos3].mean()
    p_a_given_top_minus3 = chose_a[mask_neg3].mean()
    return float(p_a_given_top_plus3 - p_a_given_top_minus3)

```

**Observed (real) value:** -0.7450 (var=0.0406)
**Predicted under pi_7:** -0.9050 (var=0.0104)
**Predicted under pi_6:** 0.7500 (var=0.0269)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, pi_7 is the better of the two theories but neither is task-invariant. On the critical double-dissociation Experiment 1, the observed value is 0.0125, essentially chance, while pi_7 predicts 0.8875 and pi_6 predicts -0.7338. So pi_7's recency-weighted tie-break overpredicted a late-cue effect and pi_6's validity-only tie-break predicted a strong opposite effect. Experiment 2 is also decisive: observed is -0.6113, pi_6 predicts +0.5927 (wrong direction), and pi_7 predicts 0.0103 (no effect). The observed negative value is consistent with choosing the late low-validity block against the early high-validity block, a serial-position effect that pi_6 cannot represent. On the remaining experiments, pi_7 is closer on Exp3, Exp4, Exp8, Exp9, Exp10, Exp11, and especially Exp12, while pi_6 is closer on Exp6 and the two are roughly tied on Exp5 and Exp7. Exp12 strongly favors pi_7's direction: observed -0.7450, pi_7 -0.9050, pi_6 +0.7500. Thus pi_6's top-validity tie-break is structurally wrong when early and late cues conflict, and pi_7's validity-gated recency is too strong in Exp1 and ineffective in Exp2. A new theory is warranted rather than a parameter repair.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_6 with a tally-first, block-attention tie-break theory. Keep the default mechanism: when the unweighted tally is clearly nonzero, choice follows the tally through a softmax plus lapse. When the tally is tied or near-tied, do not follow top validity directly. Instead, use the advertised validities only to carve the display into perceptually distinct blocks, and then choose according to the sub-tally of the most recent or spatially coherent block. A useful formalization has two routes: a large-boundary route, opened when a validity jump is large, about 0.4 or more, creating an attended block of up to six cues; and a small late-block route, opened when a block of about two to three cues sits at the end of the display even with a small validity jump, about 0.1. This predicts Experiment 1 chance because the early top-three elite is early and the remaining late and middle cues are homogeneous, so no distinct late block is formed. It predicts Experiment 2 negative scores because the first four near-perfect cues and last four 0.5 cues form a sharp boundary and the late block dominates. It predicts Experiment 12 negative anti-top scores for the same reason. It predicts Experiments 5 and 6 positive validity effects because the elite is the late block. It predicts Experiment 8 chance because the late high block has size four and only a small boundary gap, so neither route opens. The model should include parameters for a large boundary gap, a small late-block route with block-size limits of roughly two to three, a recency weight favoring later blocks, sub-tally temperature, block-size-dependent lapse, and a low-confidence near-chance lapse when no block is selected. The new theory differs from pi_6 by allowing serial-position and block effects, and differs from pi_7 by making recency depend on perceptual block segmentation rather than on a validity elite gate; this should retain pi_7's Exp12 success while fixing Exp1 and Exp2."
}
```

## Usage

```json
{
  "input_tokens": 22204,
  "output_tokens": 22016,
  "reasoning_tokens": 21243,
  "provider": "Baidu"
}
```
