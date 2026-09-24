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
**Description:** People invert the valence of binary expert ratings: a 1 is treated as a defect and a 0 as a clean, favorable signal. Each feature j probabilistically enters an inverse clean-cue tally with inclusion probability p_include_j = logistic(alpha * validity_j + gamma); excluded cues receive exactly zero weight. The included clean-cue tally drives a noisy softmax choice, and exact zero tallies are resolved by consulting the highest-validity included discriminating cue and choosing the option with the clean 0 there. If no included cue discriminates, people guess. Subject-level alpha, gamma, beta_tally, beta_tiebreak, and epsilon create individual differences, with heavy-tailed/logit-transformed subject-level distributions so that some subjects are nearly deterministic while others remain noisy, preserving the pooled inverse-valence pattern and realistically large between-subject variance.

**Parameters:**
- alpha: [5.0, 9.0]
- gamma_logit: [-2.50, 2.50]
- beta_tally_log: [-0.30, 0.50]
- beta_tiebreak_log: [0.40, 1.10]
- epsilon_logit: [-3.00, -0.40]
- validities: validities

**`predict source code`:**
```python
import numpy as np
from scipy.special import expit

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Heavy-tailed subject-level transforms of the sampled raw parameters.
    beta_tally = float(np.exp(float(parameters['beta_tally_log'])))
    beta_tiebreak = float(np.exp(float(parameters['beta_tiebreak_log'])))
    epsilon = float(expit(float(parameters['epsilon_logit'])))
    alpha = float(parameters['alpha'])
    gamma = -4.5 + 2.0 * float(expit(float(parameters['gamma_logit'])))

    p_include = expit(alpha * val + gamma)
    include = np.random.random(n_features) < p_include

    clean_a = ((b > a) & include).astype(float)
    clean_b = ((a > b) & include).astype(float)
    s = float(np.sum(clean_a) - np.sum(clean_b))

    if s != 0.0:
        p_a = expit(2.0 * beta_tally * s)
        p_core = np.array([p_a, 1.0 - p_a])
    else:
        winner = None
        cue_order = np.argsort(-val, kind='stable')
        for j in cue_order:
            if include[j] and a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        p_a = expit(beta_tiebreak) if winner == 0 else expit(-beta_tiebreak)
        p_core = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People invert the valence of binary expert ratings: a 1 is treated as a costly defect and a 0 as a clean, favorable signal. Across all features they compute an inverse-tally advantage for A, weakly weighting the inverse cues by the experts' stated validities. When the raw inverse tally is nonzero, choice is a noisy softmax over the two weakly validity-weighted inverse tallies. When the raw inverse tally is exactly tied, people do not guess uniformly; instead they break the tie lexicographically by consulting features in descending validity order and choosing the option that has a clean 0 on the first discriminating feature. Response noise is captured by a softmax plus an independent uniform lapse. This differs from Take The Best because it aggregates all cues rather than stopping at the first discriminator, and from plain inverse tallying because ties are resolved by validity-ordered clean-cue use instead of guessing.

**Parameters:**
- beta_tally: [0.43, 0.47]
- beta_tiebreak: [1.00, 1.08]
- epsilon: [0.09, 0.11]
- validity_weight: [1.90, 2.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    beta_tally = float(parameters['beta_tally'])
    beta_tiebreak = float(parameters['beta_tiebreak'])
    epsilon = float(parameters['epsilon'])
    validity_weight = float(parameters['validity_weight'])

    clean_a = (b > a)
    clean_b = (a > b)
    raw_s = float(np.sum(clean_a) - np.sum(clean_b))

    if raw_s != 0:
        if validity_weight > 0.0 and n_features > 1:
            weights = 1.0 + validity_weight * (val - np.mean(val))
        else:
            weights = np.ones(n_features)
        s = float(np.sum(weights * clean_a) - np.sum(weights * clean_b))
        scores = np.array([s, -s])
        beta = beta_tally
    else:
        cue_order = np.argsort(-val, kind='stable')
        winner = None
        for j in cue_order:
            if a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * np.ones(n_opts) / n_opts
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Six features are used. The highest-validity expert (0.90) is always the first cue TTB tests. I hold that cue's discriminating direction constant across blocks and vary the lower five cues so Tallying's net evidence changes while TTB sees the same first discriminating cue. In the top-A block the top cue is 1 vs 0 and lower-cue net wins take values +5, +3, +1, -1, -3, -5; in the top-B block the top cue is reversed. TTB predicts identical p(A) for all six top-A pairs and identical p(A) for all six top-B pairs because the first discriminating cue is unchanged; Tallying predicts a graded logistic change because its tally advantage is 1 + lower_net for top-A pairs and lower_net - 1 for top-B pairs. At lower_net = -1 in the top-A block and lower_net = +1 in the top-B block, Tallying is exactly tied while TTB commits to A or B. At lower_net = -3/-5 in the top-A block and +3/+5 in the top-B block, the two theories make opposite directional predictions. The critical signature is flat TTB choice probabilities across the six lower-net levels versus a monotone logistic gradient for Tallying. Twelve unique pairs repeated K = 8 gives 96 trials.

[1] Twelve unique pairs form a 4 × 3 factorial. Six features have validities [0.92, 0.84, 0.76, 0.68, 0.60, 0.55], so TTB's cue cascade is feature 0 → 1 → 2 → 3 → 4 → 5. The first factor is the direction of the first discriminating cue: feature 0 favors A; feature 0 favors B; feature 0 ties and feature 1 favors A; feature 0 ties and feature 1 favors B. The second factor is the tally difference (+2, 0, -2), implemented by the remaining lower features. Tallying ignores which features discriminate, so it predicts three choice-probability levels that depend only on tally difference and are identical across the four first-cue conditions. TTB ignores all lower features once a discriminating cue is found, so it predicts only two choice-probability levels (high for first-cue A, low for first-cue B) and no effect of tally difference. This yields opposite directional predictions on conflict trials: top A with tally -2 (pair 3), top B with tally +2 (pair 4), top tie/second A with tally -2 (pair 9), and top tie/second B with tally +2 (pair 10). It is the complement of the already-explored design: instead of holding the top cue constant and varying lower net, it holds tally difference constant across cue-order conditions and varies the first discriminating cue. With 12 unique pairs repeated K=8, total trials = 96.

[2] Take The Best stops at the first discriminating cue in validity order and ignores all lower cues. Inverse tallying aggregates all cues with reversed valence (0 = favorable) and chooses the option with more zeros. This design holds the inverse-tallying score at exactly zero for every trial: each option has exactly one 1 and one 0 in the two discriminating positions, and all other positions tie. Thus inverse tallying predicts p(A)=0.5 for every pair, regardless of its beta and epsilon. TTB, however, consults cues in validity order and stops at the first discriminating cue. By shifting the position of that first discriminating cue from feature 0 to feature 5 and varying its direction (A=1/B=0 vs A=0/B=1), TTB predicts p(A) to be high when the first cue favors A and low when it favors B, regardless of the lower balancing cue. This yields a strong quantitative dissociation: a flat 0.5 psychometric function across all 12 pairs for inverse tallying versus a step function of first-cue direction for TTB. Unlike the already-explored designs, which held the top cue constant while varying lower-cue net (experiment 0) or varied first-cue direction and tally difference with standard Tallying (experiment 1), this design uses the inverse-tallying tie prediction on every trial and varies the first-cue position across all six possible positions.

[3] Six features with validities [0.90, 0.80, 0.70, 0.60, 0.55, 0.50] fix TTB's cue order as feature 0 through 5. Every trial has all six features discriminating, so TTB always stops at feature 0; its direction is manipulated across trials. The remaining five features are arranged so that the inverse-tally score s_A = #(A=0,B=1) - #(A=1,B=0) takes values +2, 0, or -2. This creates a 2 (feature-0 direction: A=1/B=0 vs A=0/B=1) x 3 (inverse tally score) factorial with two replications per cell. Inverse tallying ignores which feature discriminates and uses only s_A, so it predicts p(A) ordered by s (+2 > 0 > -2) and no effect of feature-0 direction. TTB stops at feature 0 and ignores all lower features, so it predicts p(A) high when feature 0 favors A and low when feature 0 favors B, with no effect of s. The conflict cells (feature-0 A with s=-2; feature-0 B with s=+2) yield opposite directional predictions, and the s=0 cells yield p(A)=0.5 for inverse tallying while TTB commits to the top cue. This double dissociation quantitatively separates the two theories.

[4] This design dissociates the advocated inverse weighted-tally plus validity-ordered tie-break theory from plain inverse tallying. Six validities are spread as [0.95, 0.85, 0.75, 0.65, 0.55, 0.50], so the advocated weak validity weighting (1.9-2.1) makes high-validity cues roughly 2.5 times as influential as low-validity cues. Six tie trials hold the raw inverse tally at exactly 0 (one A-clean and one B-clean discriminating feature). Competing inverse tallying must predict p(A)=0.5 for all six because the tallies are equal, whereas the advocated tie-break predicts p(A) near 0.71 when the first discriminating cue in descending validity order is A=0/B=1 and near 0.29 when it is A=1/B=0; the first discriminator is varied across features 0, 1, and 2. Six conflict trials hold the raw inverse tally at +1 or -1 but place the high-validity wins on the opposite side, so the advocated validity-weighted score reverses sign. For raw +1 trials the weighted scores are about -0.62, -0.42, and -0.22, predicting p(A) around 0.38-0.46, while competing inverse tallying predicts high p(A) around 0.65-0.85. For raw -1 trials the mirror arrangement gives advocated p(A) around 0.54-0.62, while competing inverse tallying predicts low p(A) around 0.15-0.35. Twelve unique pairs repeated K=8 gives 96 trials.

[5] Eight features with symmetric validities around mean 0.75. At validity_weight=2.0 the competing weights are [1.4,1.3,1.2,1.1,0.9,0.8,0.7,0.6]. The 12 unique pairs form a 2x6 factorial: raw inverse tally d=#(A=0,B=1)-#(A=1,B=0) is +1 or -1, while the validity-weighted score s is set to -0.9,-0.6,-0.3,0,+0.3,+0.6 for d=+1 and the mirror for d=-1. No trial has d=0, so the competing tie-break is never engaged; this isolates the validity-weighting component and avoids repeating design [4]. Plain inverse tallying ignores validities: all d=+1 trials predict p(A) about 0.64-0.85 and all d=-1 trials predict p(A) about 0.15-0.36, flat across s. The competing weighted tally predicts p(A) increasing with s: about 0.33,0.38,0.44,0.50,0.56,0.62,0.67 for s=-0.9,-0.6,-0.3,0,+0.3,+0.6,+0.9. Therefore conflict trials (d=+1 with s<0; d=-1 with s>0) give opposite directional predictions, and s=0 trials oppose p=0.5 to high/low. K=8 repetitions gives 96 trials.

[6] Eight features with validities symmetric around 0.75. For the advocated theory this gives validity weights [1.4,1.3,1.2,1.1,0.9,0.8,0.7,0.6] when the raw tally is nonzero, but every trial here has raw inverse tally exactly zero, so the advocated theory always uses its validity-ordered tie-break. Feature 0 (0.95) discriminates on every trial; top-A trials have feature 0 A=0/B=1 and top-B trials have A=1/B=0, so advocated predicts p(A) approximately 0.715 for all six top-A trials and approximately 0.285 for all six top-B trials, flat across lower-cue composition. The competing inclusion theory does not tie-break on raw zero: cue inclusion is probabilistic, so the included tally is usually nonzero. Lower features are arranged to have net -1 (top A) or +1 (top B) but with increasing numbers of opposite-side lower cues across three levels. With typical inclusion parameters (e.g., alpha=7, gamma=-4.0), competing expected p(A) moves from about 0.81 (level 1) to 0.72 (level 2) to 0.54 (level 3) for top A, and mirror 0.19/0.28/0.46 for top B, crossing the advocated flat predictions. This yields a quantitative interaction: advocated predicts no effect of lower-cue composition, while competing predicts a monotone gradient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Eight features with three perfect experts (1.0) and five moderate experts (0.70). The three high-validity cues are always unanimous for one product; the five low-validity cues oppose them in 3, 4, or 5 positions (remaining low cues tied). This yields six unique pairs: 2 high-cue direction \u00d7 3 low-cue count. Competing theory uses all cues with validity weights 1.375 (high) and 0.775 (low); for high-B/low-A pairs the weighted score is negative (n=4: -1.025, n=5: -0.25), predicting p(A)\u22480.31/0.45, and the n=3 raw tie is broken by the high cue, p(A)\u22480.29. The mirror high-A/low-B pairs predict p(A)\u22480.72/0.69/0.55. The advocated inclusion theory includes high cues with p\u22480.97 and low cues with p\u22480.80 and tallies included cues equally. It predicts p(A)\u22480.18 (n=3), 0.46 (n=4), 0.71 (n=5) for high-B/low-A, and p(A)\u22480.83, 0.54, 0.29 for high-A/low-B. Thus the theories make opposite directional predictions in the n=4/n=5 high-B/low-A cells and the n=5 high-A/low-B cell, and show different slopes across low-cue count. K=16 repetitions of the six unique pairs gives 96 trials.",
  "validities": [
    1.0,
    1.0,
    1.0,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7
  ],
  "trial_a_ratings": [
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
      1,
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
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
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
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
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
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5718,
  "output_tokens": 59015,
  "reasoning_tokens": 39256,
  "provider": "CoreWeave"
}
```
