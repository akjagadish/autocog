# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Tally-first choice with a salience-gated and size-regularized validity tie-break. On every trial the unweighted signed tally of the feature differences is the default choice signal. When the tally is clear, choice follows it through a moderate softmax plus an explicit lapse. When the tally is tied or near-tied, validity is allowed to matter only if the task's validity distribution itself advertises a clear elite: either a small elite block of two to three maximally valid cues separated from the remaining cues, or a large validity gap combined with a bounded elite-block size. In those gated cases people compute a restricted sub-tally over the elite cues and choose its favored option stochastically. Larger elite blocks receive an additional lapse, preventing overconfident validity-following for wide top tiers. In large stimulus arrays, near-zero tallies are treated as low-confidence and are heavily regularized with extra choice noise, so deterministic regression blows-up do not occur.

**Parameters:**
- beta_tally: [2.0, 4.0]
- epsilon_tally: [0.02, 0.08]
- beta_near: [0.3, 0.8]
- epsilon_near: [0.78, 0.97]
- near_tally_bound: {1.5}
- n_low_conf: {15}
- tau_tie: [0.0, 0.5]
- salience_gap_hi: [0.30, 0.45]
- salience_gap_lo: [0.04, 0.06]
- top_tol: {0.01}
- top_size_small_min: {2}
- top_size_small_max: {3}
- top_size_block_max: {6}
- beta_sub: [0.35, 0.50]
- epsilon_sub: [0.08, 0.14]
- epsilon_sub_size: [0.16, 0.20]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
```


## COMPETING THEORY
**Description:** People choose by a confidence-gated unweighted signed tally. On clear tallies, choice follows the tally through a moderate softmax plus a small lapse. When the tally is tied or nearly tied, people may switch to a restricted sub-tally only if the advertised cue validities themselves flag a small coherent elite (two to three cues separated from the rest, or a large validity gap with at most six top cues). Critically, that restricted block is not defined by validity alone: cues are re-ranked by a mixture of advertised validity and serial display position, where later cues receive exponentially greater recency weight. This allows late relevant cues to dominate early equally-valid cue blocks, producing reversals like Experiment 10 while preserving validity use in Experiments 3 and 4. Large-feature environments with small tallies are treated as low confidence and heavily regularized with near-chance lapse, preventing deterministic regression blow-ups.

**Parameters:**
- beta_tally: [0.6, 1.0]
- epsilon_tally: [0.02, 0.08]
- beta_near: [0.3, 0.6]
- epsilon_near: [0.80, 0.97]
- near_tally_bound: {1.5}
- n_low_conf: {15}
- tau_tie: {0.5}
- top_tol: {0.001}
- small_size_min: {2}
- small_size_max: {3}
- gap_lo: [0.05, 0.10]
- gap_hi: [0.30, 0.45]
- block_max: {6}
- rho: [0.60, 1.00]
- beta_sub: [0.60, 1.00]
- epsilon_sub: [0.05, 0.10]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design uses six binary cues with widely spread validities. It places TTB's first discriminating cue at position 0, 1, or 2 in the validity-sorted order, while packing the later cues to oppose or tie the tallying score. In half the unique pairs the first discriminating cue favors A and in half it favors B. TTB therefore predicts p(A) is uniformly high for all six TTB-A trials and uniformly low for all six TTB-B trials, because later cues are never consulted after the first discriminating cue. Tallying instead ignores validity and counts all features: in the six oppose pairs the tally winner is opposite to TTB (count differences of 1 vs 5, 1 vs 4, and 1 vs 3), producing a crossover; in the six tie pairs tallying predicts 0.5 regardless of beta. The 12 unique pairs are each repeated 8 times for roughly 96 trials per subject. This gives a quantitative dissociation: TTB choice probability should be invariant within first-cue winner type, whereas tallying choice probability should move from near 0 to 0.5 to near 1 across the tally-difference levels.

[1] Eight experts with validities spread from .50 to .92; the highest validity is placed at display index 4 so TTB must re-sort and cannot be mimicked by left-to-right scanning. In every unique pair the top cue discriminates: trials 1-8 have top cue A, trials 9-16 top cue B. TTB therefore predicts a constant p(A) within each block (sigmoid(beta) for A-top, sigmoid(-beta) for B-top, plus lapse), because lower cues are never consulted. Tallying instead counts all eight cues, so p(A) = sigmoid(beta*d) with d = #A wins - #B wins. The d sequence is +8,+6,+4,+2,0,-2,-4,-6 for A-top and -8,-6,-4,-2,0,+2,+4,+6 for B-top. This yields a full psychometric curve for Tallying within each TTB-constant block, including d=0 pairs where Tallying must guess but TTB remains confident, and crossover pairs where the top cue and the majority of cues point in opposite directions. Mirrored blocks control for A/B response bias. 16 unique pairs x 6 repeats = 96 trials.

[2] This design dissociates validity-graded compensatory integration from pure Tallying by crossing tally difference d with a validity-composition score S. Six low-validity (0.5) and six high-validity (1.0) experts give centered validities of -0.25 and +0.25. For each pair define d = #A wins - #B wins and S = 2 * sum of centered validities of features A wins. Tallying depends only on d, so it predicts a flat p(A)=0.5 for all eight d=0 pairs, one common value for the two d=+1 pairs, and one common value for the two d=-1 pairs. Validity-graded integration predicts D = d + kappa*S, so within each d block p(A) should vary with S. The d=0 pairs span S = +3, +2.5, +2, +1, -1, -2, -2.5, -3 (the +/-2.5 pairs include tied experts), producing a psychometric curve where Tallying is flat at 0.5. The d=+1 pairs have S=+3 and -3 (D=1.3 vs 0.7 at kappa=0.1), and the d=-1 pairs have S=+2.5 and -2.5 (D=-0.75 vs -1.25), so Tallying predicts identical p(A) within each d block while the advocated theory predicts a validity-composition shift. This is a quantitative dissociation: Tallying yields a three-level step function in d, whereas the advocated theory yields a continuous S effect within each d level. Because kappa may vary across subjects, the d=0 S curve also predicts overdispersion in individual choice proportions that Tallying cannot produce.

[3] This design dissociates pure Tallying from validity-graded compensatory integration by crossing the unweighted tally difference d = #A wins - #B wins with a validity-composition score S. Ten experts have graded validities [0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0] (mean 0.75), giving centered validities [-0.25, -0.25, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.25, 0.25]. Every pair is complementary (B = 1 - A), so there are no ties, d = sum(A) - sum(B), and S = 2 * sum of centered validities of features where A = 1. The eight d = 0 pairs (A has five 1s) span S = +1.9, +1.3, +0.7, +0.1, -0.1, -0.7, -1.3, -1.9. Tallying depends only on d, so it predicts p(A) = 0.5 for all eight d = 0 pairs at every beta. Validity-graded integration predicts advantage = d + kappa * S, so for any subject with kappa != 0, p(A) must vary monotonically with S across the d = 0 pairs: high-S pairs favor A and low-S pairs favor B when kappa > 0, with the reverse when kappa < 0. This yields a quantitative dissociation: a flat line at 0.5 for Tallying versus a psychometric S curve for the competing theory, plus overdispersion in individual d = 0 choice proportions that Tallying cannot produce. The four d = +/-2 pairs (two with S = +1.8, two with S = -1.8) serve as controls to estimate beta and the d effect. Twelve unique pairs x 8 repeats = 96 trials.

[4] Sixteen experts: eight with validity 0.50 and eight with validity 1.00, so centered validities are -0.25 and +0.25. This binary spread maximizes the validity-composition score S = sum_j (v_j - 0.75)*(A_j - B_j) for a fixed tally difference and makes the competing fallback weights (v_j - 0.5)^gamma exactly zero for low-validity experts. The 16 unique pairs are organized into three blocks. (1) Six d=+1 pairs (A wins one more feature than B) with S = -3.75, -2.75, -1.75, 0.25, 1.25, 3.75. (2) Six d=-1 pairs, the A/B swaps of the first block, with S = 3.75, 2.75, 1.75, -0.25, -1.25, -3.75. (3) Four d=0 tally-tie pairs in which the high-validity tally H = #high A wins - #high B wins is +8, +4, -4, -8. On the non-tie pairs the competing validity-gated tallying rule never opens its validity gate because |d| > tau, so it predicts a single p(A) for all six d=+1 pairs and a single p(A) for all six d=-1 pairs, regardless of S. The advocated validity-graded integration predicts advantage D = d + kappa*S, so within each d block p(A) must vary with S for any subject with kappa != 0 (e.g., D shifts by +/-0.375 at kappa=0.1), giving a within-block S psychometric curve and overdispersion. On the d=0 pairs the competing rule opens its validity gate and, with high beta_v, predicts p(A) strongly follows the high-validity tally (high for H=+8/+4, low for H=-4/-8, scaled by fallback_p), whereas the advocated rule predicts D = kappa*S and, because kappa has a symmetric zero-centered prior, aggregate p(A)=0.5 for all four tie pairs. Thus the design produces a double dissociation: non-tie trials test continuous validity reweighting against a closed gate, and tie trials test the gated validity fallback against a zero-average compensatory rule. Swapped pairs control for A/B response bias; 16 unique pairs x 6 repeats = 96 trials.

[5] Twelve experts with graded validities (mean .75) give centered validities [-.25,-.25,-.25,-.25,-.15,-.05,.05,.15,.25,.25,.25,.25]. The four d=0 pairs are constructed so the centered-validity score S=sum c_j(A_j-B_j) is exactly 0. For validity-graded compensatory integration, advantage D=d+kappa*S=0, so p(A)=.5 for every kappa. For validity-gated tallying, d=0 opens the validity gate; V=sum (v_j-.5)^gamma*(A_j-B_j) is dominated by the 1.0 experts. T1/T2 have V>0 (A favored) and T3/T4 have V<0 (B favored), so p(A) is pushed above/below .5 with strength scaled by fallback_p. This is a flat-line vs high-low dissociation. The four d=+/-1 non-tie pairs cross d with S=+/-2.35. Since |d|=1>tau, gated tallying predicts p(A) depends only on d (identical within d=+1 and within d=-1), while compensatory integration predicts D=d+kappa*S, so p(A) shifts with S within each d block. Mirrored pairs control A/B response bias. 8 unique pairs x 12 repeats = 96 trials.

[6] This design uses 11 graded experts with mean validity 0.75 and centered validities from -0.25 to +0.25. Eight of the twelve unique pairs are tally ties, d = sum(A - B) = 0, constructed to have large absolute linear validity evidence on tie trials (0.75 to 1.50). The competing context-gated tallying rule therefore sees tie prevalence c = 8/12 = 0.667 and mean tie evidence far above its consistency floor, so after warm-up its validity gate opens to lambda near lambda_max (0.93 to 0.96). The four remaining pairs are critical non-tie crossovers. In C1, d = +3 and S = sum centered_validities * (A - B) = -1.40. In C3, d = +1 and S = -1.50. For the advocated validity-graded integration, D = d + kappa * S with absolute kappa at most 0.10, so C1 gives D between 2.86 and 3.14 and C3 gives D between 0.85 and 1.15; p(A) is therefore above 0.5 for both, and higher for C1 than C3, regardless of kappa sign. For the competing rule with the gate open, the top-validity experts favor B in C1 and C3, so V = sum (v_j - 0.5)^gamma * (A_j - B_j) is negative; p_valid(A) is below 0.5 across the specified gamma and beta_v ranges, and mixing that with the near-1 tally route at lambda around 0.94 keeps overall p(A) below 0.5, typically well below. Thus C1 and C3 produce a sign reversal: advocated theory predicts A-favoring choices, competing theory predicts B-favoring choices once the gate is open. C2 and C4 are exact A/B swaps and give mirrored predictions. The eight d = 0 filler pairs add a second dissociation: advocated theory predicts only small kappa * S shifts around chance, while the gated competing rule predicts strong validity-driven near-1 or near-0 choices after gate opening. Finally, because the competing gate is history-dependent, it predicts a temporal change on the critical crossover pairs from early tally-like choices to later validity-like choices, whereas advocated theory predicts no history effect.

[7] This 20-feature design uses 10 experts with validity 1.00 and 10 experts with validity 0.50, so mean validity is 0.75 and centered validities are +0.25 for the first 10 features and -0.25 for the last 10 features. All unique pairs have tally difference d = +1 or -1, so there are zero tally ties. The context-gated tallying theory therefore stays in its unweighted tallying mode throughout: tie prevalence is zero, the validity gate never opens, and p(A) depends only on d. Within the d=+1 block, the validity-composition score S = sum_j (v_j - 0.75) * (a_j - b_j) takes values +4.75, +2.75, +0.75, -0.75, -2.75, and -4.75; the d=-1 block is the exact A/B swap. The advocated theory predicts one flat p(A) across all six d=+1 pairs and one flat p(A) across all six d=-1 pairs. The competing validity-graded integration predicts D = d + kappa * S, so for any subject with kappa not equal to zero p(A) must be monotone in S within each d block; across subjects with a symmetric kappa prior this appears as S-linked slope or overdispersion rather than the advocated flat S-independent curve. Mirrored swaps control response bias and create a quantitative double dissociation: two flat d-based levels versus two S-ordered psychometric gradients.

[8] This design isolates the competing theory's large-array near-tally regularization from the advocated validity-graded integration. The validity set has 20 experts: nine at 1.00, nine at 0.50, and two at 0.75. The mean validity is 0.75, so centered validities are +0.25 for the 1.00 experts, -0.25 for the 0.50 experts, and 0 for the 0.75 experts. The top tier contains nine experts with validity 1.00, the next validity is 0.75, and the gap is 0.25. That gap is below the competing theory's minimum salience_gap_hi of 0.30, and the top block is size 9 rather than a small elite of 2-3, so the validity tie-break never becomes active. All 16 unique pairs have unweighted tally d = sum(a_i - b_i) equal to +1 or -1, never zero, so there is no exact-tie validity fallback either. Because n_features = 20, the competing theory places every pair in the low-confidence near-tally branch. With beta_near between 0.3 and 0.8 and epsilon_near between 0.78 and 0.97, p(A) for d = +1 is only about 0.51 to 0.57, and p(A) for d = -1 is about 0.43 to 0.49, with no S effect across pairs. The advocated theory predicts D = d + kappa*S, where S = sum((v_i - 0.75)*(a_i - b_i)). The first eight pairs have d = +1 and S values -4.5, -3.0, -1.5, -0.5, +0.5, +1.5, +3.0, +4.5; the last eight are A/B swaps with d = -1 and opposite S values. For kappa in [-0.10, 0.10], D remains positive on all d = +1 pairs and negative on all d = -1 pairs. With a moderate beta, advocated choice probabilities are therefore far from chance, and within each d block they vary with S in opposite directions for positive versus negative kappa subjects. The competing theory instead predicts near-chance choices in both blocks with a nearly flat S curve. The mirrored pairs control for A/B response bias, and 16 unique pairs x 6 repeats gives 96 trials.

[9] This experiment uses six experts with validities [1.0, 1.0, 1.0, 0.9, 0.9, 0.5]. The mean validity is 0.8833, so the centered validities are [0.1167, 0.1167, 0.1167, 0.0167, 0.0167, -0.3833]. The top tier is exactly the three 1.0 experts; the second highest validity is 0.9, giving a validity gap of 0.10 and a top size of 3, which opens the advocated theory's small-elite tie-break gate.

Every unique pair is a tally tie: d = sum(A - B) = 0. On these pairs the advocated theory's default tallying signal is silent, and its gated tie-break consults only the three top experts. Let T be the top sub-tally, the number of top experts favoring A minus the number favoring B, and let S be the centered-validity score, S = sum c*(A - B). The construction uses the two 0.9 experts and the one 0.5 expert to vary S independently of T; in this validity set S = 0.5*T + 0.4*U, where U is the difference sum of the two 0.9 experts.

Trials 1-6 have T = +1, +1, +1, +2, +2, +3 with S values -0.30, 0.10, 0.50, 0.20, 0.60, 0.70. Trials 7-12 are the exact A/B mirrors with T = -1, -1, -1, -2, -2, -3 and S values 0.30, -0.10, -0.50, -0.20, -0.60, -0.70.

The advocated theory predicts p(A) follows T only: the three T=+1 pairs are indistinguishable around 0.57-0.61, the two T=+2 pairs around 0.645-0.71, and the T=+3 pair around 0.71-0.79, with mirrored values below 0.5. In particular, within T=+1, the large S change from -0.30 to +0.50 should not move p(A).

The competing validity-graded integrator has d = 0 on every trial, so its only advantage term is kappa*S. Because kappa has a symmetric zero-centered prior, the expected p(A) is exactly 0.5 for every one of the 12 pairs, regardless of T. Positive-kappa subjects would order the choices by S, with P1 below 0.5 and P3 above 0.5, while negative-kappa subjects would reverse that order. Thus the design gives a quantitative dissociation: a T-ordered elite sub-tally psychometric function for the advocated theory versus a flat-at-chance S-only pattern for the competing theory. The 12 unique pairs are repeated K = 8 times to give 96 trials.

[10] All 12 unique pairs are unweighted tally ties, d = sum(A - B) = 0, so both fallback gates open. The validities define a three-cue elite: the first three experts have validity 1.0 and the remaining nine have 0.8, giving a gap of 0.20 and top_size = 3, which meets both theories' small-elite conditions. The first three features define T = #top A wins - #top B wins among positions 0-2; the last three define L = #late A wins - #late B wins among positions 9-11. Middle features 3-8 are tally-balancing fillers. For the advocated model, combined relevance = validity * exp(rho * (position - max_position)) makes the final three positions dominate every earlier 0.8 cue for rho in [0.60, 1.00], so its restricted block is positions 9-11 and its weighted sub-tally is exactly +3 on L=+3 trials and -3 on L=-3 trials. The competing model ignores position and uses the three 1.0 experts, so its sub-tally equals T. The design is a 4 x 2 factorial over T in {+3,+1,-1,-3} and L in {+3,-3}; trials 1-8 are sign-reversal cells. Critical predictions: in T=+3,L=-3 cells, competing p(A) is roughly 0.85-0.91 while advocated p(A) is roughly 0.03-0.07; in T=-3,L=+3 cells these reverse. Same-sign cells provide magnitude controls. A validity-only fallback cannot produce an L effect within a fixed T, and a recency-weighted fallback cannot produce a T effect within a fixed L, giving a quantitative double dissociation. The 12 unique pairs are repeated 8 times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Eight experts are used. The first four validities (1.00, 0.996, 0.993, 0.991) fall within the advocated model's top_tol of 0.01, so it groups them into a top block of size 4. The competing model's top_tol of 0.001 leaves only the 1.00 expert in its top block, so its top_size is 1. With the next highest validity at 0.50, the advocated model sees a large validity gap of 0.50 with a bounded top block of size 4, which activates its block_clear gated tie-break on every trial. The competing model instead sees a validity gap of only 0.004 and no eligible small elite, so its validity gate stays closed. All 12 unique pairs have an exact unweighted tally tie, d = sum(A - B) = 0, so the competing model remains in default tally mode and predicts p(A) = 0.5 for every pair. The advocated model uses the gated top-block sub-tally T over the first four experts. The unique T values are +4, +3, +3, +2, +2, +1, -1, -2, -2, -3, -3, -4. With beta_sub between 0.35 and 0.50 and the top_size=4 extra lapse of about 0.24 to 0.34, advocated p(A) is ordered by T: roughly 0.61 to 0.68 for T = +1, rising to roughly 0.79 to 0.87 for T = +4, with mirrored values below 0.5 for negative T. The competing model is flat at 0.5 for all 12 pairs. Thus the design produces a quantitative T-ordered psychometric function for the advocated theory against a flat chance prediction for the competing theory. The 12 unique pairs are repeated 8 times for 96 trials per subject.",
  "validities": [
    1.0,
    0.996,
    0.993,
    0.991,
    0.5,
    0.5,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 9359,
  "output_tokens": 21127,
  "reasoning_tokens": 20022,
  "provider": "Baidu"
}
```
