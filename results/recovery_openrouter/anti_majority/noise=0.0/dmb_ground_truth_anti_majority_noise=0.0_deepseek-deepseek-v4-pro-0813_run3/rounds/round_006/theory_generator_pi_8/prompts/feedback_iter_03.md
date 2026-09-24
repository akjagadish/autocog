# feedback_iter_03

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Propose a new theory: inverse-valence majority with unique-expert precedence. The theory should retain the inverse-valence assumption, treating rating 0 as a clean/favorable cue and rating 1 as a defect, so every discriminating feature contributes approximately +1 for A or -1 for B to a raw clean-cue tally. The new ingredient is that if exactly one discriminating expert has a strictly highest communicated validity, that unique expert receives a privileged override weight, active within a limited raw-tally window such as abs(D) <= 3, with strength increasing in the validity gap to the next-best expert. If two or more experts are tied at the maximum validity, the unique-expert override is suppressed and choice is driven by a near-equal-weight tally over all discriminating cues. Exact raw-tally ties can be broken by choices among the changed experts, but tied top experts should not be allowed to override a non-zero low-validity majority. The model should include subject-level parameters for the probability of applying unique-expert precedence, the maximum override window, softmax choice gain, and a lapse rate, all on distributions matched to the observed between-subject variance. This theory differs from pi_7 by removing continuous logit-tanh weighted evidence for every feature and from pi_5 by removing probabilistic cue exclusion and by tying the privileged role to uniqueness of the top expert rather than generic validity order. It should be instantiated with parameter ranges that allow near-equal majority weighting and a small but occasionally decisive unique-top bonus.


## CANDIDATE THEORY
Inverse-valence majority with gated unique-expert precedence. A binary rating of 0 is encoded as clean/favorable and a rating of 1 as a defect, so every discriminating expert contributes approximately +1 to a raw clean-cue tally when A is clean and -1 when B is clean. The default choice is a softmax over this raw equal-weight tally with subject-specific gain and lapse. When the discriminating experts contain exactly one expert with strictly highest communicated validity, that unique expert receives a gap-scaled precedence bonus with subject-level application probability, but only within a limited raw-tally window. If two or more experts are tied at the maximum validity, unique-expert precedence is suppressed and choice is driven by the raw equal-weight tally. Exact zero-tally ties invoke the unique top expert only probabilistically through a separate subject-level tie probability, and that tie preference is now deliberately softened. Nonzero small-tally conflicts receive a stronger, gap-scaled unique-expert override, especially for one-cue conflicts, so low-validity majorities are not followed too automatically.

`predict(parameters, state, history) -> np.ndarray`:
import math
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError('validities length mismatch')

    changed = a != b
    disc = np.flatnonzero(changed)
    if disc.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))

    gain = float(parameters['gain'])
    epsilon = float(parameters['epsilon'])
    p_apply = float(parameters['p_override'])
    tie_p = float(parameters['tie_p'])
    window = int(round(float(parameters['window'])))

    E = D
    unique_top = False
    sign_top = 0.0
    gap = 0.0

    if disc.size == 1:
        j_top = int(disc[0])
        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
        gap = 0.04
        unique_top = True
    else:
        vals_disc = val[disc]
        top_val = float(np.max(vals_disc))
        top_hits = disc[np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
        unique_top = top_hits.size == 1
        if unique_top:
            j_top = int(top_hits[0])
            sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
            others = vals_disc[~np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
            if others.size == 0:
                gap = 0.04
            else:
                gap = float(top_val - float(np.max(others)))

    if unique_top:
        if abs(D) <= 1e-12:
            # Softened, subject-level probabilistic tie break.  The unique
            # expert is no longer strongly decisive on exact zero-tally ties.
            if np.random.rand() < tie_p:
                tie_bonus = 0.30 + 0.22 * math.tanh(gap / 0.06)
                E = sign_top * tie_bonus
            else:
                E = 0.0
        elif abs(D) <= window and np.random.rand() < p_apply:
            # Stronger gap-scaled unique-expert override for small nonzero
            # conflicts, with a slight extra boost for one-cue conflicts.
            bonus = 1.30 + 2.10 * math.tanh(gap / 0.10)
            if abs(D) <= 1.0:
                bonus *= 1.12
            E = D + sign_top * bonus
    elif abs(D) <= 1e-12:
        E = 0.0

    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- gain: [1.25, 1.95]
- epsilon: [0.18, 0.38]
- p_override: [0.45, 0.70]
- tie_p: [0.15, 0.55]
- window: {1, 2}
- validities: validities

`rationale`:
This is a minimal in-family recalibration of the running-best candidate. The exact-zero-tally unique-top tie break was still too decisive, so tie_p is lowered and the tie bonus coefficient is reduced substantially. The nonzero unique-expert override is strengthened for small-tally conflicts by raising p_override and increasing the gap-scaled bonus, with a small additional boost for one-cue conflicts to restore the correct signs in Experiments 6 and 7 without reintroducing continuous validity weighting or cue exclusion. Subject-level ranges are widened to increase the simulated between-subject variance. This keeps the inverse-valence majority and unique-expert-precedence family intact.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1038 -> ACCEPTED
- iter 2: loss=0.0661 -> ACCEPTED
- iter 3: loss=0.0660 -> ACCEPTED
- iter 4 (current candidate you are grading): loss=0.0975 -> REJECTED
Running-best (last accepted) base: iter 3 at loss=0.0660.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    A = np.array(data['option_a_ratings'].tolist(), dtype=int)
    B = np.array(data['option_b_ratings'].tolist(), dtype=int)
    y = 1.0 - data['response'].to_numpy(dtype=float)

    top_a = A[:, 0] > B[:, 0]
    top_b = A[:, 0] < B[:, 0]

    lower_a_wins = (A[:, 1:] > B[:, 1:]).sum(axis=1)
    lower_b_wins = (B[:, 1:] > A[:, 1:]).sum(axis=1)
    lower_net = lower_a_wins - lower_b_wins

    high = lower_net >= 1
    low = lower_net <= -1

    def diff(top_mask):
        hi = y[top_mask & high]
        lo = y[top_mask & low]
        if len(hi) == 0 or len(lo) == 0:
            return 0.0
        return float(hi.mean() - lo.mean())

    return (diff(top_a) + diff(top_b)) / 2.0
```

**Observed (real) value:** -0.5117 (var=0.0161)
**Candidate trajectory (this loop):**
  - iter 1: -0.3421 (var=0.0104) (Δ vs real +0.1696)
  - iter 2: -0.4492 (var=0.0055) (Δ vs real +0.0625)
  - iter 3: -0.4729 (var=0.0085) (Δ vs real +0.0387)
  - iter 4 (current): -0.4925 (var=0.0103) (Δ vs real +0.0192)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0092 (var=0.0043)
- pi_2: 0.6258 (var=0.0234)
- pi_3: -0.5379 (var=0.0084)
- pi_4: -0.5413 (var=0.0057)
- pi_5: -0.5308 (var=0.0142)
- pi_4_1: -0.5442 (var=0.0102)
- pi_6: -0.3700 (var=0.2536)
- pi_7: -0.4754 (var=0.0242)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_rows = data['option_a_ratings'].tolist()
    b_rows = data['option_b_ratings'].tolist()
    tally = []
    for ra, rb in zip(a_rows, b_rows):
        a_wins = sum(1 for x, y in zip(ra, rb) if x > y)
        b_wins = sum(1 for x, y in zip(ra, rb) if y > x)
        tally.append(a_wins - b_wins)
    tally = np.asarray(tally, dtype=float)
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    pos_rate = choose_a[tally > 0].mean()
    neg_rate = choose_a[tally < 0].mean()
    return float(pos_rate - neg_rate)
```

**Observed (real) value:** -0.6438 (var=0.0430)
**Candidate trajectory (this loop):**
  - iter 1: -0.3763 (var=0.0214) (Δ vs real +0.2675)
  - iter 2: -0.5050 (var=0.0279) (Δ vs real +0.1388)
  - iter 3: -0.5563 (var=0.0160) (Δ vs real +0.0875)
  - iter 4 (current): -0.5350 (var=0.0382) (Δ vs real +0.1088)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7112 (var=0.0379)
- pi_1: -0.0013 (var=0.0059)
- pi_3: -0.6069 (var=0.0076)
- pi_4: -0.5913 (var=0.0097)
- pi_5: -0.5938 (var=0.0426)
- pi_4_1: -0.6038 (var=0.0238)
- pi_6: -0.6238 (var=0.2467)
- pi_7: -0.6138 (var=0.0287)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.55])
    cue_order = np.argsort(-validities, kind='stable')

    a = np.array([list(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([list(r) for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)

    n = a.shape[0]
    if n == 0:
        return 0.0

    predicted = np.full(n, -1, dtype=int)
    for cue in cue_order:
        undecided = predicted == -1
        disc = np.logical_and(undecided, a[:, cue] != b[:, cue])
        if not np.any(disc):
            continue
        predicted[disc] = np.where(a[disc, cue] > b[disc, cue], 0, 1)

    decided = predicted != -1
    if not np.any(decided):
        return 0.0

    match = (resp[decided] == predicted[decided]).astype(float)
    return float(np.mean(2.0 * match - 1.0))
```

**Observed (real) value:** -0.4300 (var=0.0221)
**Candidate trajectory (this loop):**
  - iter 1: -0.4338 (var=0.0091) (Δ vs real -0.0038)
  - iter 2: -0.4271 (var=0.0110) (Δ vs real +0.0029)
  - iter 3: -0.3887 (var=0.0095) (Δ vs real +0.0413)
  - iter 4 (current): -0.0921 (var=0.0099) (Δ vs real +0.3379)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7113 (var=0.0398)
- pi_3: 0.0175 (var=0.0062)
- pi_2: 0.0117 (var=0.0115)
- pi_4: -0.4575 (var=0.0063)
- pi_5: -0.4596 (var=0.0245)
- pi_4_1: -0.4029 (var=0.0226)
- pi_6: -0.3550 (var=0.1762)
- pi_7: -0.6425 (var=0.0137)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    # inverse-tally advantage for A: #(A=0,B=1) - #(A=1,B=0)
    s = (b > a).sum(axis=1) - (a > b).sum(axis=1)
    mask_pos = s == 2
    mask_neg = s == -2
    pos = data.loc[mask_pos, 'response']
    neg = data.loc[mask_neg, 'response']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    p_pos = (pos == 0).mean()
    p_neg = (neg == 0).mean()
    return float(p_pos - p_neg)
```

**Observed (real) value:** 0.4925 (var=0.0658)
**Candidate trajectory (this loop):**
  - iter 1: 0.3606 (var=0.0172) (Δ vs real -0.1319)
  - iter 2: 0.5025 (var=0.0195) (Δ vs real +0.0100)
  - iter 3: 0.5587 (var=0.0236) (Δ vs real +0.0663)
  - iter 4 (current): 0.5112 (var=0.0298) (Δ vs real +0.0188)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6381 (var=0.0147)
- pi_1: -0.0244 (var=0.0050)
- pi_2: -0.7525 (var=0.0447)
- pi_4: 0.4313 (var=0.0096)
- pi_5: 0.4237 (var=0.0670)
- pi_4_1: 0.5069 (var=0.0237)
- pi_6: 0.4681 (var=0.1627)
- pi_7: 0.5363 (var=0.0483)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

PAIR_SIGNS = {
    ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): 1.0,
    ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 0, 1, 0, 0, 0)): 1.0,
    ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)): -1.0,
    ((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 0, 1, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 0)): 1.0,
}

def metric(data: pd.DataFrame) -> float:
    contributions = []
    for _, row in data.iterrows():
        key = (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))
        sign = PAIR_SIGNS.get(key)
        if sign is None:
            continue
        y = 1.0 if row['response'] == 0 else -1.0
        contributions.append(y * sign)
    return float(np.mean(contributions)) if contributions else 0.0
```

**Observed (real) value:** 0.2658 (var=0.1584)
**Candidate trajectory (this loop):**
  - iter 1: 0.3146 (var=0.0140) (Δ vs real +0.0488)
  - iter 2: 0.3492 (var=0.0164) (Δ vs real +0.0833)
  - iter 3: 0.2296 (var=0.0127) (Δ vs real -0.0362)
  - iter 4 (current): 0.1479 (var=0.0099) (Δ vs real -0.1179)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2971 (var=0.0096)
- pi_3: -0.2708 (var=0.0095)
- pi_1: -0.6967 (var=0.0260)
- pi_2: 0.3704 (var=0.0144)
- pi_5: 0.3858 (var=0.0225)
- pi_4_1: 0.2446 (var=0.0156)
- pi_6: 0.3300 (var=0.1102)
- pi_7: 0.2608 (var=0.0101)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    validities = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55], dtype=float)
    weights = 1.0 + 2.0 * (validities - validities.mean())

    pos = []
    neg = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        clean_a = (b > a).astype(float)
        clean_b = (a > b).astype(float)
        diff = clean_a - clean_b
        d = float(np.sum(diff))
        s = float(np.dot(weights, diff))
        y = 1.0 - float(row['response'])
        if d < 0.0 and s > 0.0:
            pos.append(y)
        elif d > 0.0 and s < 0.0:
            neg.append(y)

    if not pos or not neg:
        return 0.0

    return float(np.mean(pos) - np.mean(neg))
```

**Observed (real) value:** 0.0867 (var=0.4244)
**Candidate trajectory (this loop):**
  - iter 1: 0.0292 (var=0.0283) (Δ vs real -0.0575)
  - iter 2: 0.0267 (var=0.0329) (Δ vs real -0.0600)
  - iter 3: -0.0550 (var=0.0179) (Δ vs real -0.1417)
  - iter 4 (current): 0.1758 (var=0.0326) (Δ vs real +0.0892)
**Other theories' values on this metric (for reference):**
- pi_3: -0.5333 (var=0.0223)
- pi_4: 0.2158 (var=0.0210)
- pi_1: -0.7375 (var=0.0404)
- pi_2: 0.6717 (var=0.0518)
- pi_5: 0.1383 (var=0.0736)
- pi_4_1: 0.1075 (var=0.0264)
- pi_6: 0.0358 (var=0.0678)
- pi_7: -0.2058 (var=0.0270)

### Experiment 7
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    a0 = data['option_a_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    b0 = data['option_b_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    ld = data.apply(lambda row: sum([1 for x, y in zip(row['option_a_ratings'][1:], row['option_b_ratings'][1:]) if x != y]), axis=1).to_numpy(dtype=float)
    choose_a = 1.0 - data['response'].to_numpy(dtype=float)
    level = np.where(ld == 1, 1, np.where(ld == 3, 2, np.where(ld == 5, 3, 0)))
    topA = ((a0 == 0) & (b0 == 1))
    topB = ((a0 == 1) & (b0 == 0))
    m1 = choose_a[topA & (level == 1)].mean() if (topA & (level == 1)).any() else 0.0
    m2 = choose_a[topB & (level == 3)].mean() if (topB & (level == 3)).any() else 0.0
    m3 = choose_a[topA & (level == 3)].mean() if (topA & (level == 3)).any() else 0.0
    m4 = choose_a[topB & (level == 1)].mean() if (topB & (level == 1)).any() else 0.0
    return float(m1 + m2 - m3 - m4)
```

**Observed (real) value:** 0.0725 (var=0.0580)
**Candidate trajectory (this loop):**
  - iter 1: 0.0550 (var=0.0487) (Δ vs real -0.0175)
  - iter 2: 0.0212 (var=0.0393) (Δ vs real -0.0512)
  - iter 3: -0.0250 (var=0.0448) (Δ vs real -0.0975)
  - iter 4 (current): 0.0012 (var=0.0473) (Δ vs real -0.0713)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0025 (var=0.0525)
- pi_5: 0.3325 (var=0.0612)
- pi_1: 0.0288 (var=0.0221)
- pi_2: 0.0137 (var=0.0602)
- pi_3: -0.0025 (var=0.0792)
- pi_4_1: -0.0400 (var=0.0672)
- pi_6: 0.1800 (var=0.1187)
- pi_7: 0.0812 (var=0.0474)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    counts = {}
    totals = {}

    for row in data.itertuples(index=False):
        a = [int(x) for x in row.option_a_ratings]
        b = [int(x) for x in row.option_b_ratings]

        # h = 0 means the three high-validity cues favour B under inverse valence;
        # h = 1 is the mirror where they favour A.
        h = 0 if sum(a[:3]) > sum(b[:3]) else 1

        # Number of opposing low-validity cues.
        n = sum(1 for i in range(3, len(a)) if a[i] != b[i])

        key = (h, n)
        totals[key] = totals.get(key, 0) + 1
        if int(row.response) == 0:
            counts[key] = counts.get(key, 0) + 1

    props = {}
    for h in (0, 1):
        for n in (3, 4, 5):
            tot = totals.get((h, n), 0)
            props[(h, n)] = (counts.get((h, n), 0) / tot) if tot > 0 else 0.5

    d3 = props[(0, 3)] - props[(1, 3)]
    d4 = props[(0, 4)] - props[(1, 4)]
    d5 = props[(0, 5)] - props[(1, 5)]

    return float(-d3 + d4 + 2.0 * d5)
```

**Observed (real) value:** 1.0100 (var=2.8518)
**Candidate trajectory (this loop):**
  - iter 1: 1.3262 (var=0.1793) (Δ vs real +0.3163)
  - iter 2: 1.8413 (var=0.1475) (Δ vs real +0.8313)
  - iter 3: 1.7437 (var=0.1235) (Δ vs real +0.7338)
  - iter 4 (current): 1.7763 (var=0.1986) (Δ vs real +0.7663)
**Other theories' values on this metric (for reference):**
- pi_5: 1.3512 (var=0.8533)
- pi_4: -0.1263 (var=0.1938)
- pi_1: 1.4112 (var=0.1813)
- pi_2: -2.0162 (var=0.4200)
- pi_3: 1.7625 (var=0.1534)
- pi_4_1: 0.5012 (var=0.5143)
- pi_6: 1.1075 (var=2.7433)
- pi_7: 1.6462 (var=0.3404)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

_VALIDITIES = np.array([0.95, 0.90, 0.65, 0.64, 0.63, 0.62, 0.60, 0.60])
_FIRST_CUE_ORDER = np.argsort(-_VALIDITIES, kind='stable')


def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=np.int64)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=np.int64)
    r = np.asarray(data['response'].tolist(), dtype=np.int64)

    n = len(data)
    first_col = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)

    for j in _FIRST_CUE_ORDER:
        differs = a[:, j] != b[:, j]
        newly = differs & ~assigned
        first_col[newly] = j
        assigned |= differs

    valid = first_col >= 0
    if not valid.any():
        return 0.0

    idx_valid = np.nonzero(valid)[0]
    cols_valid = first_col[valid]
    a_first = a[idx_valid, cols_valid]
    b_first = b[idx_valid, cols_valid]
    favors_a = (a_first == 0) & (b_first == 1)

    high_mask = np.zeros(n, dtype=bool)
    low_mask = np.zeros(n, dtype=bool)
    high_mask[idx_valid[favors_a]] = True
    low_mask[idx_valid[~favors_a]] = True

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0

    chose_a = r == 0
    return float(chose_a[high_mask].mean() - chose_a[low_mask].mean())

```

**Observed (real) value:** 0.3050 (var=0.0605)
**Candidate trajectory (this loop):**
  - iter 1: 0.2900 (var=0.0086) (Δ vs real -0.0150)
  - iter 2: 0.3329 (var=0.0068) (Δ vs real +0.0279)
  - iter 3: 0.3079 (var=0.0137) (Δ vs real +0.0029)
  - iter 4 (current): 0.0492 (var=0.0116) (Δ vs real -0.2558)
**Other theories' values on this metric (for reference):**
- pi_4_1: 0.4188 (var=0.0158)
- pi_5: 0.2196 (var=0.0342)
- pi_1: -0.7067 (var=0.0351)
- pi_2: 0.0204 (var=0.0104)
- pi_3: -0.0142 (var=0.0125)
- pi_4: 0.4296 (var=0.0093)
- pi_6: 0.1742 (var=0.1020)
- pi_7: 0.4579 (var=0.0150)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    scores = []
    for sid, sub in data.groupby('subject_id'):
        low = []
        for a, b, resp in zip(sub['option_a_ratings'], sub['option_b_ratings'], sub['response']):
            s = 0
            for x, y in zip(a, b):
                if x == 0 and y == 1:
                    s += 1
                elif x == 1 and y == 0:
                    s -= 1
            if s > 0:
                low.append(1.0 if resp == 0 else 0.0)
            elif s < 0:
                low.append(1.0 if resp == 1 else 0.0)
        if not low:
            scores.append(0.0)
        else:
            q = float(np.mean(low))
            scores.append(q * q - q)
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1667 (var=0.0053)
**Candidate trajectory (this loop):**
  - iter 1: -0.2128 (var=0.0012) (Δ vs real -0.0462)
  - iter 2: -0.1911 (var=0.0024) (Δ vs real -0.0244)
  - iter 3: -0.1705 (var=0.0023) (Δ vs real -0.0038)
  - iter 4 (current): -0.1837 (var=0.0034) (Δ vs real -0.0170)
**Other theories' values on this metric (for reference):**
- pi_5: -0.2057 (var=0.0022)
- pi_4_1: -0.2374 (var=0.0002)
- pi_1: -0.1112 (var=0.0033)
- pi_2: -0.1264 (var=0.0029)
- pi_3: -0.1529 (var=0.0007)
- pi_4: -0.2445 (var=0.0000)
- pi_6: -0.1839 (var=0.0030)
- pi_7: -0.1824 (var=0.0024)

### Experiment 11
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_list = data['option_a_ratings'].tolist()
    b_list = data['option_b_ratings'].tolist()
    k = np.array([sum(1 for x, y in zip(a, b) if x == 0 and y == 1) for a, b in zip(a_list, b_list)], dtype=float)
    resp = np.asarray(data['response'], dtype=float)
    subject_ids = data['subject_id'].to_numpy()
    informative = k != 4
    inv_pred = np.where(k < 4, 1.0, 0.0)
    inv_match = resp == inv_pred

    def subject_consistency(mask):
        valid = informative & mask
        if not valid.any():
            return 0.5
        return float(np.mean(inv_match[valid]))

    unique_ids = np.unique(subject_ids)
    if len(unique_ids) > 1:
        positive_count = []
        for sid in unique_ids:
            s = subject_consistency(subject_ids == sid)
            positive_count.append(1.0 if s < 0.5 else 0.0)
        return float(np.mean(positive_count))
    s = subject_consistency(np.ones(len(data), dtype=bool))
    return float(max(s, 1.0 - s))

```

**Observed (real) value:** 0.0000 (var=0.0045)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0021) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0022) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0020) (Δ vs real +0.0000)
  - iter 4 (current): 0.0000 (var=0.0025) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0400 (var=0.0007)
- pi_5: 0.0000 (var=0.0037)
- pi_1: 0.9200 (var=0.0019)
- pi_2: 1.0000 (var=0.0083)
- pi_3: 0.0000 (var=0.0019)
- pi_4: 0.0000 (var=0.0011)
- pi_4_1: 0.0000 (var=0.0015)
- pi_7: 0.0000 (var=0.0038)

### Experiment 12
**Design**
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def score_one(g):
        a = np.array([list(x) for x in g['option_a_ratings']], dtype=int)
        b = np.array([list(x) for x in g['option_b_ratings']], dtype=int)
        resp = g['response'].to_numpy().astype(int)

        d = ((a == 0) & (b == 1)).sum(axis=1) - ((a == 1) & (b == 0)).sum(axis=1)
        choice_a = (resp == 0).astype(float)

        far = np.abs(d) == 3
        far_pos = far & (d == 3)
        far_neg = far & (d == -3)
        if np.any(far_pos) and np.any(far_neg):
            orient_diff = float(np.mean(choice_a[far_pos]) - np.mean(choice_a[far_neg]))
            orient = 1.0 if orient_diff >= 0.0 else -1.0
        else:
            corr = float(np.corrcoef(d.astype(float), choice_a)[0, 1])
            orient = 1.0 if np.isfinite(corr) and corr >= 0.0 else -1.0

        aligned_d = orient * d
        majority_opt = np.where(aligned_d > 0, 0, 1)

        raw_top_dir = np.where((a[:, 0] == 0) & (b[:, 0] == 1), 1.0, -1.0)
        top_opt = np.where(orient * raw_top_dir > 0.0, 0, 1)
        top_align = (top_opt == majority_opt)
        followed_top = (resp == top_opt).astype(float)

        near = np.abs(d) == 1
        far2 = np.abs(d) == 3

        def arcs(mask):
            vals = followed_top[mask]
            n = vals.shape[0]
            if n == 0:
                return 0.0
            p = (float(np.sum(vals)) + 0.5) / (float(n) + 1.0)
            p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            return float(np.arcsin(np.sqrt(p)))

        near_aligned = arcs(near & top_align)
        near_opposed = arcs(near & ~top_align)
        far_aligned = arcs(far2 & top_align)
        far_opposed = arcs(far2 & ~top_align)

        return (near_aligned - near_opposed) + (far_aligned - far_opposed)

    if data.empty:
        return 0.0

    scores = []
    for sid in data['subject_id'].unique():
        sub = data[data['subject_id'] == sid]
        scores.append(score_one(sub))
    return float(np.mean(scores))
```

**Observed (real) value:** 1.2973 (var=0.2536)
**Candidate trajectory (this loop):**
  - iter 1: 0.8941 (var=0.0518) (Δ vs real -0.4032)
  - iter 2: 1.2728 (var=0.0520) (Δ vs real -0.0245)
  - iter 3: 1.2470 (var=0.0389) (Δ vs real -0.0503)
  - iter 4 (current): 1.2648 (var=0.0388) (Δ vs real -0.0325)
**Other theories' values on this metric (for reference):**
- pi_5: 1.5845 (var=0.1245)
- pi_6: 1.9342 (var=0.0366)
- pi_1: 0.1091 (var=0.0296)
- pi_2: 1.6796 (var=0.3099)
- pi_3: 1.2888 (var=0.0583)
- pi_4: 1.3818 (var=0.0296)
- pi_4_1: 1.3853 (var=0.0963)
- pi_7: 1.1792 (var=0.1659)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array([list(row) for row in data['option_a_ratings']], dtype=float)
    b = np.array([list(row) for row in data['option_b_ratings']], dtype=float)

    clean_a = (a == 0) & (b == 1)
    clean_b = (a == 1) & (b == 0)

    D = clean_a.sum(axis=1) - clean_b.sum(axis=1)
    x = clean_a[:, :5].sum(axis=1) - clean_b[:, :5].sum(axis=1)

    chose_a = (data['response'].to_numpy(dtype=float) == 0)

    mask_pos = (D == 2) & (x == -3)
    mask_mirror = (D == -2) & (x == 3)

    if mask_pos.sum() == 0 or mask_mirror.sum() == 0:
        return 0.0

    p_pos = float(chose_a[mask_pos].mean())
    p_mirror = float(chose_a[mask_mirror].mean())
    return p_pos - p_mirror

```

**Observed (real) value:** 0.7300 (var=0.0558)
**Candidate trajectory (this loop):**
  - iter 1: 0.5175 (var=0.0450) (Δ vs real -0.2125)
  - iter 2: 0.6975 (var=0.0563) (Δ vs real -0.0325)
  - iter 3: 0.6450 (var=0.0227) (Δ vs real -0.0850)
  - iter 4 (current): 0.6425 (var=0.0344) (Δ vs real -0.0875)
**Other theories' values on this metric (for reference):**
- pi_7: 0.4625 (var=0.0677)
- pi_5: 0.1475 (var=0.1242)
- pi_1: -0.7650 (var=0.0460)
- pi_2: -0.7350 (var=0.0554)
- pi_3: 0.6325 (var=0.0265)
- pi_4: -0.5125 (var=0.0583)
- pi_4_1: -0.2175 (var=0.0968)
- pi_6: 0.1750 (var=0.1163)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    crit_a1 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    crit_b1 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_a2 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_b2 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

    def pair_key(row):
        return (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))

    keys = data.apply(pair_key, axis=1)
    mask1 = (keys == (crit_a1, crit_b1)).to_numpy()
    mask2 = (keys == (crit_a2, crit_b2)).to_numpy()
    resp = data['response'].to_numpy()

    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0

    p_a1 = float((resp[mask1] == 0).mean())
    p_a2 = float((resp[mask2] == 0).mean())
    return p_a1 - p_a2

```

**Observed (real) value:** 0.3950 (var=0.3296)
**Candidate trajectory (this loop):**
  - iter 1: 0.2837 (var=0.0268) (Δ vs real -0.1113)
  - iter 2: 0.4625 (var=0.0230) (Δ vs real +0.0675)
  - iter 3: 0.4587 (var=0.0270) (Δ vs real +0.0638)
  - iter 4 (current): 0.4200 (var=0.0348) (Δ vs real +0.0250)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2938 (var=0.0899)
- pi_7: -0.2175 (var=0.1258)
- pi_1: 0.7537 (var=0.0387)
- pi_2: -0.7450 (var=0.0311)
- pi_3: 0.5400 (var=0.0262)
- pi_4: -0.0587 (var=0.0308)
- pi_4_1: 0.0538 (var=0.0355)
- pi_6: 0.1925 (var=0.1178)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate captures the qualitative inverse-valence majority pattern across most experiments. E3 is nearly exact (-0.4338 simulated vs -0.4300 observed), and E5, E7, E9, E11 are all close. The negative E1 and E2 signatures have the correct sign, and the unique-expert precedence mechanism improves E13 and E14 relative to pi_7 and pi_5. However, the candidate is systematically too shallow on the majority-driven magnitude metrics: E1 is -0.342 vs -0.512 observed, E2 is -0.376 vs -0.644, E4 is 0.361 vs 0.493, E12 is 0.894 vs 1.297, E13 is 0.518 vs 0.730, and E14 is 0.284 vs 0.395. At the same time E8 is too large (1.326 vs 1.010), and the simulated between-subject variance is much smaller than observed in E5, E6, E8, E12, and E14. A key implementation issue is that the subject-level gain parameter is sampled but never used in the softmax: p_core uses exp([E, 0]) rather than exp(gain * [E, 0]), so the default majority tally is weaker than the parameterization implies. The unconditional and fairly large D=0 unique-top tie bonus likely also makes the unique expert too decisive in E8.
Rationale: Regenerate with minor in-family fixes. First, actually apply the softmax gain by scaling the evidence vector by gain before exponentiation; the current dead gain parameter is a clear bug. Second, reduce epsilon or otherwise steepen the default majority mapping so E1, E2, E4, E12, E13, and E14 move closer to the observed stronger inverse-valence effects. Third, recalibrate the unique-top tie bonus downward or make it probabilistic through p_override, since E8 is already too sharp; keep the tie mechanism inside the inverse-valence plus unique-expert-precedence family rather than removing it. Fourth, widen the subject-level distributions, especially p_override and possibly gain or tie preference, to bring the between-subject variance in E6, E8, E12, and E14 closer to the observed variability. Do not reintroduce continuous all-cue validity weighting or probabilistic cue exclusion.

**Outcome of your advice:** iter 1 candidate loss=0.1038 -> iter 2 candidate loss=0.0661 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate is a clear improvement and remains faithful to the prescribed inverse-valence majority with gated unique-expert precedence family. Several experiments are now close to the observed values: E3 is nearly exact, E4, E9, E12, E13, and E11 are all within a small tolerance. The previous dead-gain bug appears fixed, and the majority mapping is substantially steeper. However, the candidate still has two systematic problems. First, the unique-top tie break for abs(D)=0 is applied unconditionally whenever a unique top expert exists, bypassing the subject-level p_override gate. This likely explains why E8 remains far too large, 1.841 simulated versus 1.010 observed, and why the simulated between-subject variance is far too low in E5, E6, E8, E12, and E14: the D=0 tie behavior is deterministic across subjects. Second, the nonzero unique-expert override is still too frequent or too strong in small-conflict trials, leaving the strong inverse-valence majority effects in E1 and E2 too shallow, while pushing E5 and E14 too far in the opposite direction. The candidate's simulated variance remains substantially smaller than observed in many experiments, which suggests the subject-level parameters are not heterogeneous enough or are not applied to the tie mechanism at all.
Rationale: Regenerate with minor in-family fixes only. First, make the D=0 unique-top tie break subject-level and probabilistic, either by gating it through p_override or by introducing a separate subject-level tie probability with a wide range, and reduce its coefficient so it remains large enough to keep E3 near its current excellent fit but no longer makes the unique expert so decisive in E8. Second, tighten the nonzero override: reduce the window from 1,2,3 to 1,2 and lower the p_override range toward approximately 0.15 to 0.60. This should let low-validity majorities drive choices more strongly in E1, E2, and E5, and should pull E8 and E10 toward their observed values, while preserving the unique-expert precedence mechanism inside the prescribed family. Third, widen the subject-level distributions, especially the tie probability and p_override, to bring the between-subject variance in E6, E8, E12, and E14 closer to observed levels. Do not reintroduce continuous all-cue validity weighting or probabilistic cue exclusion, and do not remove the unique-expert precedence mechanism; the change should be a recalibration of when and how strongly the unique top expert is allowed to override the raw equal-weight clean-cue tally.

**Outcome of your advice:** iter 2 candidate loss=0.0661 -> iter 3 candidate loss=0.0660 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed inverse-valence majority with gated unique-expert precedence family and fits several experiments well: E9 is nearly exact, E10 is very close, E11 is exact, and E12 is within about 0.05. However, the recalibration is uneven. E8 is still far too high, 1.744 simulated versus 1.010 observed, so the newly probabilistic D=0 tie break is still much too decisive. E6 and E7 now have the wrong sign, -0.055 versus +0.087 and -0.025 versus +0.073 respectively, indicating that the unique-top override for nonzero small-tally conflicts has become too weak and the model now follows the low-validity majority too often. E4 and E14 overshoot the observed values, while E5 and E13 undershoot, suggesting the override strength is not well calibrated across different raw-tally magnitudes. Simulated between-subject variance also remains substantially smaller than observed in E5, E6, E8, E12, and E14, so the subject-level heterogeneity is still too limited.
Rationale: Regenerate with minor in-family recalibrations only. First, further weaken the D=0 tie decision: lower tie_p from 0.55 to 0.95 down toward roughly 0.25 to 0.60, and reduce the tie bonus coefficient below the current 0.72 plus 0.50 tanh(gap over 0.05), so E8 moves toward the observed value near 1.01 without removing the unique-expert tie preference. Second, strengthen the unique-top override specifically for nonzero small-tally conflicts with abs(D) equal to 1 or 2, which should restore the correct positive signs on E6 and E7; raising p_override toward approximately 0.45 to 0.75 and/or increasing the gap-scaled bonus for small validity gaps near 0.05 is a plausible in-family fix. Third, make the nonzero override strength depend more smoothly on D magnitude or the validity gap because E4 and E14 are currently too strong while E5 and E13 are too weak. Fourth, widen the subject-level parameter ranges, especially p_override, tie_p, gain, and epsilon, to increase simulated between-subject variance toward the observed levels in E5, E6, E8, E12, and E14. Do not return to the unconditional tie behavior and do not switch to continuous validity weighting or probabilistic cue exclusion.

**Outcome of your advice:** iter 3 candidate loss=0.0660 -> CURRENT candidate loss=0.0975 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The current candidate was rejected by the gate and its aggregate loss increased from 0.0660 to 0.0975. The decisive damage comes from weakening the exact-zero-tally unique-expert tie decision. That change collapsed the three experiments where the running-best candidate was already well calibrated: E3 moved from -0.389 to -0.092 against a real -0.430, E9 moved from 0.308 to 0.049 against a real 0.305, and E5 moved from 0.230 to 0.148 against a real 0.266. Those metrics have relatively low between-subject variance, so these errors are costly and stable, not noise. The nonzero override changes were also poorly targeted: E8 remains far too high at 1.776 versus 1.010 real, E6 overshoots to 0.176 versus 0.087 real, and E7 remains near zero at 0.001 versus 0.073 real. The candidate does improve a few majority-magnitude metrics such as E1, E4, E12, and E14, but these gains do not compensate for the collapse of the D=0 tie-driven experiments.",
  "verdict": "regenerate",
  "rationale": "Regenerate starting from the accepted iter 3 base, not from this rejected variant. First, restore the iter 3 D=0 unique-top tie parameters exactly; the previous push to lower tie_p and shrink the tie bonus was rejected by the gate, and it is the main reason E3, E9, and E5 collapsed. Second, do not repeat the global strengthening of the nonzero small-tally override. The base had E6 and E7 only mildly negative, and those metrics have large variance, so they do not justify large parameter moves. If the proposer wants a one-cue boost, it must be gated on the actual number of discriminating cues being one, disc.size == 1, not on abs(D) <= 1, which the current rejected implementation does and which likely drives E8 upward by boosting many multi-cue D=+/-1 conflicts. Third, any nonzero override should use a much smaller bounded bonus, roughly half or less of the current 1.30 plus 2.10 * tanh(gap / 0.10) scale, and p_override should stay near the iter 3 range rather than being raised. Fourth, for E8 specifically, consider making the override bonus shrink as the number of opposing low-validity cues increases, for example by dividing the bonus by the number of discriminating cues or by using a stricter window for multi-cue conflicts. This stays inside the prescribed unique-expert-precedence family but should reduce E8 saturation without disturbing the near-tie experiments E3 and E9."
}
```

## Usage

```json
{
  "input_tokens": 22168,
  "output_tokens": 13763,
  "reasoning_tokens": 13169,
  "provider": "Baidu"
}
```
