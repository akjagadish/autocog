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
**Description:** People treat advertised expert validity as a cue to ignore rather than trust: cues with low validity, especially those near chance, receive the largest decision weight, while very high-validity cues are discounted because they are seen as redundant or overstated. Choices are formed by a continuous reversed-validity weighted additive comparison of the two options, with signed feature differences weighted by an inverse diagnosticity function of the advertised validities. Ties on a feature contribute zero evidence. The accumulated evidence then passes through a softmax choice rule with inverse temperature beta, plus a small uniform lapse probability epsilon that captures attentional or motor noise.

**Parameters:**
- theta: {1.0}
- gamma: [0.3, 0.4]
- beta: [4.8, 5.5]
- epsilon: [0.15, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Reversed-diagnosticity weighted additive integration.
    # state is expected to be array-like of shape (2, n_features),
    # with row 0 = option A and row 1 = option B.
    # History is ignored because validities are advertised in the
    # instructions and no trial-by-trial feedback is provided.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reversed-validity integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1:
        validities = validities.reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {stim.shape[1]}."
        )

    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])

    # Reversed diagnosticity weights. Low-validity cues receive large
    # weights; high-validity cues are discounted. The max guard keeps
    # weights non-negative even for perfect validities.
    weights = np.maximum(theta - validities, 0.0) ** gamma

    # Signed feature-wise evidence: positive favors A, negative favors B.
    # Feature ties contribute zero.
    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the two options [A, B].
    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - logits.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## COMPETING THEORY
**Description:** People normalize each advertised cue validity relative to the current task ensemble, giving salience only to cues whose validity falls below the ensemble median. Low-validity outliers dominate the signed feature tally, with ties contributing zero. The decisiveness of the softmax choice rule is itself context-gated: when a single cue becomes a near-monopoly in normalized decision weight, the response temperature is quenched toward chance; when a small set of low-validity cues is nearly co-equal, the response is sharpened. This keeps equal-weight tallying as a limiting case while allowing reversed-validity-like and regime-sensitive choice behavior.

**Parameters:**
- w0: [0.005, 0.04]
- kappa: [1.5, 4.0]
- gamma: [1.0, 1.5]
- c: [0.005, 0.02]
- beta: [1.7, 2.3]
- epsilon: [0.12, 0.18]
- share_hi_center: [0.84, 0.92]
- share_hi_width: [0.012, 0.035]
- gap_hi_center: [0.42, 0.58]
- gap_hi_width: [0.03, 0.08]
- beta_floor_hi: [0.25, 0.35]
- eff_n_center: [2.95, 3.10]
- eff_n_width: [0.10, 0.18]
- top3_sim_center: [0.92, 0.98]
- top3_sim_width: [0.04, 0.10]
- beta_boost_lo: [0.8, 1.4]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus array.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    median = np.median(validities)
    mad = np.median(np.abs(validities - median))
    c = float(parameters['c'])

    denom = mad + c
    if denom <= 0.0:
        denom = 1e-12
    deviance = np.maximum(0.0, median - validities) / denom

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    weights = w0 + kappa * np.power(deviance, gamma)

    total = weights.sum()
    if total <= 0.0:
        weights = np.ones_like(weights) / max(weights.size, 1)
    else:
        weights = weights / total

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    nw = sorted_w.shape[0]
    top1 = sorted_w[0]
    top2 = sorted_w[1] if nw > 1 else top1
    top3 = sorted_w[2] if nw > 2 else (sorted_w[-1] if nw > 0 else 0.0)
    top3_similarity = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights ** 2)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    beta_floor_hi = float(parameters['beta_floor_hi'])

    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    beta_boost_lo = float(parameters['beta_boost_lo'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_similarity - top3_sim_center) / top3_sim_width) ** 2)
    low_gate = eff_gate * balance_gate

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    beta_eff = beta + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.05, 15.0))

    logits = np.array([evidence_A, 0.0])
    z = beta_eff * (logits - np.max(logits))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Eight experts are used. Expert 0 is advertised as 90% accurate; the remaining seven are clearly lower (75% down to 50%). Every critical trial makes Expert 0 discriminate: it favors one option. The seven lower experts are arranged so that k = 0,1,...,7 of them favor the opposite option. Take The Best stops at Expert 0 and therefore predicts a constant probability of choosing the top-cued option across all k. Tallying counts all eight experts; the tally margin for the top-cued option is 8 - 2k, so its predicted choice probability is a decreasing sigmoid in k, crossing chance at k=4 and reversing for k>4. Two mirrored blocks are included (Expert 0 favors A in one block, B in the other) to double observations and control for response bias. This yields a strong quantitative dissociation: a flat psychometric function over k under Take The Best versus a negatively sloped sigmoid under Tallying. The 16 unique pairs are repeated 6 times for roughly 96 trials.

[1] Six experts with validities 0.90, 0.80, 0.70, 0.60, 0.55, 0.50 define a fixed TTB cue order. The 16 unique pairs form a cue-locus × tally-margin design. In A-first trials (A0-A7) the first discriminating cue in validity order favors A; in B-first trials (B0-B7) it favors B. Higher-validity cues are tied on r=1 and r=2 trials so the first discriminator is not always the top expert, extending the already-explored design. Lower cues are arranged to produce tally margins (#A wins - #B wins) of -4, -3, -2, 0, +2, +3, +4. TTB stops at the first discriminator, so it predicts a single p(A) for all A-first trials and a single p(A) for all B-first trials, independent of the lower-cue margin. Tallying counts all six cues, so it predicts p(A) = sigmoid(beta * margin), increasing monotonically with the margin and equal to chance at margin 0. Thus TTB predicts a flat step function in margin within each first-discriminator direction, whereas Tallying predicts a graded psychometric function; zero-margin trials additionally oppose TTB's directional prediction to Tallying's chance prediction. The 16 unique pairs are repeated K=6 times for 96 trials.

[2] Six experts with validities 0.92, 0.84, 0.76, 0.68, 0.58, 0.50. The design holds the Tallying score constant within two blocks while varying validity-weighted evidence. The first eight trials are zero tally-margin pairs: A wins exactly one feature and B wins exactly one feature, so Tallying gives a 1-1 tie and predicts p(A)=0.5 for every pair. The discriminating feature pair is varied from (A wins the 50% expert, B wins the 92% expert) to the reverse, so the advocated validity-weighted evidence increases monotonically from w5-w0 to w0-w5, predicting a psychometric curve from below to above chance. The last eight trials are constant +2 tally-margin pairs: A wins 4 features, B wins 2, so Tallying predicts the same p(A)=sigmoid(2*beta) for all eight. The four A-win features are chosen along a dominance chain (each adjacent pair swaps one A-win from a lower-validity to a higher-validity feature), so the validity-weighted evidence strictly increases for any monotone validity-weighting function. With offset=0.5, gamma=1 the +2 block runs from -0.24 (weighted favors B despite A winning 4 of 6 features) to +1.12. Thus Tallying predicts two flat lines (0.5 and a single >0.5 value), while the advocated theory predicts two increasing psychometric functions of weighted evidence. The 16 unique pairs are repeated K=6 for about 96 trials.

[3] Twelve experts: six high-validity (0.94-0.99) and six low-validity (0.50-0.55). The eight unique pairs form four groups. Group A (A1-A3): A wins 3-5 high-validity features while B wins 4-6 low-validity features, so Tallying margin is -1 (predicts p(A)<0.5), but validity-weighted evidence is positive (predicts p(A)>0.5). Group B (B1-B3): A wins 4-6 low-validity features while B wins 3-5 high-validity features, so Tallying margin is +1 (predicts p(A)>0.5), but validity-weighted evidence is negative (predicts p(A)<0.5). Groups C and D have zero Tallying margin (predicts p(A)=0.5) but all-high vs all-low wins give positive and negative validity-weighted evidence respectively. Thus Tallying and validity-weighted integration make opposite directional predictions for every pair, and the within-group graded evidence allows estimating beta for both accounts.

[4] This design tests whether low-validity experts are overweighted (reversed-validity) against the equal-count Tallying benchmark. Two experts are advertised near chance (50%, 51%) and seven are near-perfect (98%–99.5%). On every trial the two low-validity experts favor one option while k = 0…7 high-validity experts favor the other option; all remaining experts are tied. In A-low trials Tallying gives A exactly 2 wins and B k wins, so its tally margin is 2 − k: it crosses chance at k = 2 and predicts B for k ≥ 3. Reversed-validity instead gives the 50%/51% experts weights of about 0.75–0.81 each (the largest weights) and the 98%–99.5% experts weights of about 0.12–0.31 each. The two low weights sum to about 1.5–1.6, which outweighs six of the seven high weights (and all seven for gamma ≈ 0.4). Thus reversed-validity predicts A for k = 0…6 in A-low trials and B for k = 0…6 in B-low trials. The critical k = 2…6 trials therefore produce opposite directional predictions: Tallying chooses the high-validity side (or guesses at k = 2), while reversed-validity chooses the low-validity side. The psychometric crossover is shifted from k = 2 under Tallying to k = 7 or beyond under reversed-validity, a large quantitative dissociation. The two mirrored blocks double observations and control for response bias.

[5] Eight experts with four near-chance validities (0.50-0.53) and four near-perfect validities (0.97-0.995). Reversed-validity weights near-chance experts heavily (weight roughly 0.74-0.81) and discounts near-perfect experts (weight roughly 0.12-0.29), while Tallying gives every expert equal unit weight. The 16 unique pairs form two blocks. Block 1 (pairs 1-8) holds the Tallying margin at 0 (A wins exactly 3 features, B wins exactly 3, 2 ties), so Tallying predicts p(A)=0.5 for every pair. The pairs swap A's wins from low-validity to high-validity features, so reversed-weighted evidence decreases from about +1.7 to -1.7, predicting a graded psychometric curve crossing chance. Block 2 (pairs 9-16) holds the Tallying margin at +2 (A wins 5, B wins 3), so Tallying predicts the same >0.5 p(A) for all eight pairs. The same low-to-high validity swap makes reversed-weighted evidence decrease from about +2.8 to -0.7, so reversed-validity predicts a decreasing curve that crosses below chance on the last three pairs, opposite in direction to Tallying. Thus Tallying predicts two flat lines (0.5 and a single >0.5 value), whereas reversed-validity predicts two graded psychometric functions of weighted evidence, with several pairs yielding opposite directional predictions.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Nine experts are used: three near-chance experts (0.53, 0.52, 0.51) and six higher-validity experts (0.99 down to 0.90). The three near-chance experts are deliberately close together. In the competing theory they become the top three normalized weights, each about 0.32 to 0.33, so the top3-similarity gate is near 1 and the effective-n gate is near 3; this holds the context-gated beta roughly constant. All six higher validities together contribute less than about 0.02 of normalized weight, so the competing theory's evidence is set almost entirely by which option wins more of the three low-validity experts. Block A has A win two low experts and B win one, giving competing evidence about +0.33 for A on every trial, so p(A) stays near 0.65 to 0.70 regardless of the high-feature pattern. Block B mirrors this, giving competing evidence about -0.33 and p(A) near 0.30 to 0.35. The advocated theory does not normalize and weights each expert by max(1 - v, 0)^gamma. Its low-expert weights are near 0.80, but its high-expert weights remain substantial, from about 0.50 for the 0.90 expert down to about 0.25 for the 0.99 expert. Eight high-feature patterns are chosen so that the B-winning high-feature weight total S equals 0, 0.349, 0.752, 0.951, 1.202, 1.511, 1.707, or 2.267. In the two-low A block, advocated evidence for A therefore runs +3.08, +2.38, +1.58, +1.18, +0.68, +0.06, -0.33, -1.45, giving p(A) from near ceiling to near floor and crossing chance between the last two intermediate patterns. In the mirrored block it runs +1.45, +0.76, -0.05, -0.45, -0.95, -1.57, -1.96, -3.08. Thus the competitor predicts a flat line within each low-split block, while the advocated theory predicts a steep psychometric function of the high-validity pattern; four pairs are directionally opposite. The 16 unique pairs repeated 6 times give roughly 96 trials.",
  "validities": [
    0.99,
    0.98,
    0.97,
    0.95,
    0.93,
    0.9,
    0.81,
    0.79,
    0.78
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
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
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0,
      1,
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
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      1,
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
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
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
      0,
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
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
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
      1,
      1,
      1,
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
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
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
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5697,
  "output_tokens": 29879,
  "reasoning_tokens": 28364,
  "provider": "Baidu"
}
```
