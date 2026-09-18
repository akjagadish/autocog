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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** **Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

**Parameters:**
- polarity: [-1.0, -0.35]
- gamma: [0.1, 1.15]
- beta: [2.0, 9.0]
- epsilon: [0.10, 0.55]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two options
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    # ---------- 2. validity-derived weights ----------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n_features:
        # graceful fallback: linearly descending validities
        if n_features == 1:
            val = np.array([0.9])
        else:
            val = np.linspace(0.9, 0.55, n_features)

    gamma = float(parameters["gamma"])
    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)          # normalised validity strength
    w = np.power(r, max(gamma, 0.0))
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        w = np.ones(n_features)
    else:
        w = w / m                                       # mean weight == 1 (scale grows with n cues)

    # ---------- 3. signed compensatory integration ----------
    s = float(parameters["polarity"])                   # negative: a '1' counts AGAINST the option
    score_a = s * float(np.dot(w, a))
    score_b = s * float(np.dot(w, b))
    scores = np.array([score_a, score_b], dtype=float)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # ---------- 4. margin-graded softmax + lapse ----------
    z = beta * scores
    z = z - np.max(z)                                    # numerically stable
    e = np.exp(z)
    p_core = e / e.sum()
    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with well-spread validities (0.90, 0.80, 0.70, 0.60, 0.55) so that the validity ordering is unambiguous and TTB's cue cascade is well defined. The design contains three diagnostic trial classes plus controls, each presented in both A/B mirror forms to remove side bias.

(1) CONFLICT trials (pairs 1-6): the highest discriminating cue favors one option while the majority of remaining cues favors the other. E.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1]: TTB stops on expert 1 and picks A with probability ~sigmoid(beta), whereas Tallying counts 1 vs 4 and picks B with probability ~sigmoid(3*beta) \u2014 the two theories predict opposite responses, and Tallying's tally margin (3) is large so its prediction is nearly deterministic for any moderate beta. Pairs 3-4 use a smaller tally margin (1 vs 2), so Tallying predicts a *weaker* (margin-1 softmax) preference for the option TTB rejects; the graded difference across margin sizes is a quantitative signature that separates the models beyond mere direction.

(2) TALLY-TIE trials (pairs 7-10): the feature-wise wins are exactly balanced, so Tallying is forced to 50/50 regardless of beta, while TTB still discriminates decisively. Pairs 7-8 tie 2-2 with the top cue discriminating (TTB confident); pairs 9-10 tie 1-1 with all three high-validity cues equal so TTB is driven by the low-validity expert 4. If subjects show systematic (non-chance) preferences on these trials, Tallying cannot account for it at any parameter setting.

(3) CONTROL trials (pairs 11-14) where both theories agree, providing clean leverage on beta and epsilon so that the diagnostic trials are interpreted with well-identified noise parameters.

Pair 5-6 additionally makes the top cue non-discriminating (both options rated 1 by expert 1), forcing TTB down to expert 2 while Tallying reads a 1-vs-3 margin the other way \u2014 testing whether people truly stop at the first *discriminating* cue rather than integrating.

Advocated theory (TTB) predicts: choice of the option favored by the highest discriminating cue on ALL of pairs 1-10, with roughly constant accuracy independent of how many lower cues oppose it, and above-chance systematic choice on the tally-tie pairs. Tallying predicts the opposite direction on pairs 1-6 with strength graded by tally margin, and exact chance on pairs 7-10.

[1] I advocate TALLYING and design for a quantitative dissociation that the previous experiment (which emphasized direction reversals only) did not isolate: the *graded* dependence of choice probability on the tally margin versus TTB's flat, margin-independent confidence.

Six experts with clearly ordered, well-spread validities (0.90, 0.82, 0.74, 0.66, 0.58, 0.53) make TTB's cue cascade unambiguous (cue order = feature order). Every pair appears in both A/B mirror forms (16 unique pairs, K=6 -> 96 trials) so side bias cancels.

(1) GRADED-AGREEMENT trials (pairs 1-6). Both models favour the same option, but the tally margin is +1 (3 wins vs 2), +3 (4 vs 1), and +5 (5 vs 0). TTB always stops at expert 1 and therefore predicts EXACTLY the same accuracy, (1-eps)*sigmoid(beta)+eps/2, in all three conditions; its binary score never changes. Tallying predicts monotonically increasing accuracy, sigmoid(1*beta) < sigmoid(3*beta) < sigmoid(5*beta). Because these conditions also pin down beta and epsilon, a reliable accuracy gradient across margin 1 -> 3 -> 5 cannot be absorbed by any TTB parameter setting: TTB is forced to attribute all three to one sigmoid(beta). This is the core quantitative signature, and it is measured on trials where the two theories agree in DIRECTION, so it is not contaminated by any direction-reversal strategy shift.

(2) CONFLICT trials (pairs 7-12). Expert 1 favours one option while the remaining experts favour the other, with tally margins of -1 (2 vs 3), -3 (1 vs 4) and -4 (1 vs 5). TTB predicts choice of the expert-1 option with the SAME probability in all three; Tallying predicts choice of the opposite option with strength graded by margin. Direction and gradient are both diagnostic here, and the -1 case is the crucial one: Tallying predicts a weak reversal (sigmoid(beta)) exactly matching in magnitude what it predicts for the +1 agreement pairs.

(3) DESCENT trial (pairs 13-14): experts 1 and 2 both rate the two products 1, so TTB must descend to expert 3 and choose that option, while the tally is 1 vs 3 the other way. This tests whether people truly stop at the first discriminating cue or integrate across the three low-validity experts.

(4) TALLY-TIE trial (pairs 15-16): A wins on experts 1-2, B wins on experts 3-4 (2-2 tie). Tallying is forced to exactly 50/50 for any parameters; TTB predicts a decisive expert-1-based preference. This is the competing theory's unique prediction and provides a hard falsification test of my advocated model; including it makes the design fair rather than stacked.

Predictions if Tallying is right: accuracy rises steeply with |tally margin| in both the agreement and conflict blocks, choices reverse against expert 1 on pairs 7-12 and 13-14, and choices sit at chance on pairs 15-16. If TTB is right: flat accuracy across all margins, expert-1 option chosen throughout pairs 1-12, expert-3 option on 13-14, and clearly above-chance systematic choice on the tally-tie pairs.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "I advocate TAKE-THE-BEST (TTB) against Polarised Weighted Evidence Accumulation (PWEA). PWEA makes two commitments that TTB flatly denies: (i) a rating of '1' is read as a NEGATIVE flag (polarity s<0), so the option carrying the greater validity-weighted mass of 1s is REJECTED; and (ii) choice probability scales smoothly with that weighted mass margin. TTB instead says the option holding the '1' on the highest discriminating expert is chosen, with a single, margin-independent accuracy (1-eps)*sigmoid(beta) on every discriminating trial.\n\nCrucially, the two earlier experiments in this series pitted TTB against Tallying, using only 'conflict' pairs of the form A=[1,0,0,0,0] vs B=[0,1,1,1,1]. On exactly those pairs PWEA MIMICS TTB (the top-cue option also carries the least mass of 1s), so nothing so far can separate them. This design therefore concentrates on the trial class where PWEA and TTB predict OPPOSITE choices: pairs where the option favoured by the top expert also carries MORE 1s.\n\nFive experts with well-spread, strictly ordered validities (0.93, 0.84, 0.72, 0.62, 0.54) make TTB's cue cascade unambiguous. gamma=1 weights (mean-1 normalised) are w=[1.87, 1.48, 0.96, 0.52, 0.17]. All 8 base pairs are shown in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials) so any side bias cancels.\n\nCLASS I - POLARITY REVERSAL, LARGE MARGIN (pairs 1-2 and mirrors 9-10):\n[1,1,1,1,1] vs [0,0,0,0,0] (mass 5.00 vs 0) and [1,1,1,0,0] vs [0,0,0,1,1] (4.31 vs 0.70). TTB picks the all-/mostly-1 option essentially deterministically; PWEA picks the OPPOSITE option essentially deterministically for every admissible parameter setting (any gamma, any s<0, beta>=2). These are the hard falsification trials: they cannot be reconciled with PWEA by any parameter tuning, and equally, a reliable preference for the sparse-1 option would kill TTB outright.\n\nCLASS II - GRADED REVERSAL (pairs 3, 4, 6 and mirrors): [1,0,0,0,0] vs [0,0,0,0,0] (PWEA margin 1.87), [0,0,0,0,1] vs [0,0,0,0,0] (margin 0.17 at gamma=1), and [1,0,0,0,0] vs [0,0,0,1,0] (margin 1.35). TTB predicts the SAME accuracy toward the option with the single 1 (or with the higher-validity 1) in all three, because only the sign on the first discriminating cue matters. PWEA predicts the opposite direction with strength graded ~10:1 between the top-cue and bottom-cue trials. So the design delivers a within-subject direction test AND a magnitude-gradient test on the same trial family: flat, above-chance, top-cue-consistent accuracy across margins is uniquely TTB; a margin-graded reversal is uniquely PWEA.\n\nCLASS III - CASCADE / NEAR-TIE TRIAL (pair 7 and mirror): [1,1,0,0,0] vs [1,0,1,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and decisively picks A. PWEA's weighted masses are 3.35 vs 3.52 at gamma=1 (and reverse to favour A at gamma<=0.5): across essentially the whole admissible gamma range PWEA is pinned near 50/50 here, whereas TTB predicts full, undiminished accuracy. Systematic choice of A on this pair is diagnostic of one-reason stopping and cannot be produced by PWEA at any evidence sensitivity.\n\nCLASS IV - AGREEMENT / CALIBRATION (pairs 5 and 8, plus mirrors): [1,0,0,0,0] vs [0,1,1,1,1] and [1,0,0,0,0] vs [0,1,1,0,0]. Here both theories favour the same option (the classic TTB-vs-Tallying conflict items on which PWEA mimics TTB). They pin down beta and epsilon for BOTH models independently of the diagnostic classes, so the reversal trials are interpreted against well-identified noise parameters, and they guarantee the design is not simply stacked toward one family.\n\nPREDICTIONS. If TTB is right: on every pair the option bearing the '1' on the highest discriminating expert is chosen at a single, roughly constant above-chance rate, including the mostly-1 dominance pairs (Class I), the bottom-cue pairs (Class II), and the cascade pair (Class III). If PWEA is right: choices REVERSE on Classes I-II (sparse-1 option chosen), near-deterministically on Class I and only weakly on the small-margin Class II items, sit at chance on Class III, and agree with TTB only on Class IV.",
  "validities": [
    0.93,
    0.84,
    0.72,
    0.62,
    0.54
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
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
      1
    ],
    [
      1,
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
      0
    ],
    [
      1,
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
      0
    ],
    [
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
      1,
      1
    ],
    [
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
      0
    ],
    [
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
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
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
      1,
      1
    ],
    [
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
      0
    ],
    [
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
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
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
      1
    ],
    [
      1,
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
      0
    ],
    [
      1,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 8382,
  "output_tokens": 13524
}
```
