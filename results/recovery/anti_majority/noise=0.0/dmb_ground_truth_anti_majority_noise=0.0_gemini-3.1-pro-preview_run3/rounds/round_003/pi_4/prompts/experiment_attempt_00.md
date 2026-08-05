# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Parameters:**
- beta: [0.1, 20.0]
- p_ttb: [0.35, 0.95]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Probabilistic Cue Selection (Random Dictator): Decision-makers do not deterministically follow a strict cue hierarchy (as in Take The Best) nor do they exhaustively sum all available cues (as in Tallying or WADD). Instead, on any given trial, they probabilistically sample a single cue from the set of discriminating cues to determine their choice. The probability of sampling a specific cue is proportional to its validity raised to a non-linear scaling parameter (gamma). If gamma is very high, the most valid cue is almost always sampled, perfectly mimicking TTB. If gamma is near zero, cues are sampled uniformly, producing a soft Tallying-like behavior where choice probability reflects the proportion of cues favoring an option. This provides a mathematically elegant, single-process probabilistic mechanism that gracefully spans the spectrum of non-compensatory to compensatory decision-making without requiring ad-hoc response noise or explicit strategy mixtures.

**Parameters:**
- gamma: [0.0, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify cues that discriminate between option A and option B
    diff = a - b
    disc_mask = diff != 0
    
    if not np.any(disc_mask):
        # No cues discriminate; guess uniformly
        p_core = np.array([0.5, 0.5])
    else:
        # Scale validities of discriminating cues non-linearly
        w = val[disc_mask] ** gamma
        w_sum = np.sum(w)
        
        if w_sum == 0:
            p_core = np.array([0.5, 0.5])
        else:
            # Probability of sampling each discriminating cue
            p = w / w_sum
            
            # The choice is determined entirely by the sampled cue.
            # Thus, the probability of choosing A is the sum of sampling probabilities
            # for cues where A > B.
            p_a = np.sum(p[diff[disc_mask] > 0])
            p_b = np.sum(p[diff[disc_mask] < 0])
            p_core = np.array([p_a, p_b])
            
    # Incorporate uniform lapse rate (guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use a 4-feature design with descending validities. TTB relies solely on the highest-validity cue that discriminates between the two options. Tallying ignores validities and simply counts the number of features each option wins. The trials are constructed to pit the highest-validity discriminating cue against a larger quantity of lower-validity cues. For example, in trial 1, Option A wins on the most valid cue but loses on the other three, leading TTB to predict A and Tallying to predict B. Other trials create ties for Tallying where TTB makes a deterministic prediction, further separating their statistical likelihoods.

[1] This design uses 5 features with strictly descending validities to systematically dissociate Tallying from Take The Best (TTB). TTB always chooses the option that wins on the most valid discriminating feature, completely ignoring the sheer number of features won. Tallying, in contrast, simply counts the number of winning features for each option, ignoring their validities. The selected trials pit the single most valid discriminating cue against a larger quantity of lower-validity cues (e.g., Option A wins on the highest validity cue but loses on the other four, leading TTB to predict A and Tallying to predict B). The design also includes trials where Tallying predicts a tie (equal number of wins) but TTB has a strong preference, and trials where the highest validity cues tie, forcing TTB to look further down the hierarchy while Tallying simply tallies the remaining wins.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive with Non-Linear Cue Scaling (WADD), we must prevent WADD from mimicking TTB at high values of the scaling parameter gamma. If validities are spaced too far apart, a high gamma will cause the single best cue to dominate the sum of all lower cues, making WADD's predictions identical to TTB's. By packing several high validities closely together (e.g., 0.95, 0.93, 0.91, 0.89), we ensure that the sum of the slightly-lower validities can outstrip the single highest validity, even when exponentiated by gamma=20. Trials are designed to pit the single best cue against a coalition of slightly inferior cues, forcing a strict divergence: TTB will stubbornly choose the option with the single best cue, while WADD will choose the option with the coalition of slightly inferior cues.

[3] To quantitatively dissociate the Weighted Additive (WADD) model with non-linear cue scaling from Take The Best (TTB), we exploit WADD's ability to interpolate between compensatory and non-compensatory decision making via its gamma parameter. TTB is strictly non-compensatory: it always decides based on the single most valid discriminating cue, entirely ignoring all lower-validity cues. We use a 5-feature design with evenly spaced validities. The trials are constructed such that the most valid discriminating cue favors one option, but a coalition of lower-validity cues favors the other. Depending on the value of gamma, WADD can predict either option, allowing it to capture compensatory behavior that TTB fundamentally cannot model. By including trials where the 'best' cue is tied and the decision falls to lower cues, we further separate the predictions of the two models across the parameter space.

[4] To quantitatively dissociate the Strategy Mixture (TTB + Tallying) model from the WADD model with Non-Linear Cue Scaling, this design exploits the 'invariance' of the Mixture model to lower-validity cue permutations. In the Strategy Mixture model, as long as the TTB prediction (based on the highest discriminating cue) and the Tallying prediction (the simple count of winning cues) remain constant, the predicted choice probabilities are identical. By creating a sequence of trials (e.g., Trials 1-3) where Option A always wins the highest validity cue and Option B always wins 3 out of the 5 cues, the Strategy Mixture model predicts the exact same choice probabilities across these trials. However, the WADD model integrates the actual validities of the specific lower-tier cues. By shifting which specific lower-validity cues Option A vs. B wins across these trials, WADD predicts systematically shifting choice probabilities. We repeat this logic for trials where the first cue is tied, forcing TTB to rely on the second cue, again holding the Tallying ratio constant while altering the WADD score margins.

[5] This design achieves a strict, parameter-free qualitative dissociation between the Strategy Mixture (TTB + Tallying) model and the WADD model with Non-Linear Cue Scaling. By carefully setting the validities and cue distributions, we create 'Adversarial' trials (e.g., Trial 1) where Option A possesses the single highest-validity cue (winning TTB) AND a larger total number of cues (winning Tallying). Because both sub-strategies of the Mixture model prefer Option A, the Mixture model is mathematically forced to predict a preference for A. However, Option B possesses a concentrated coalition of slightly lower but still highly valid cues. Across the entire allowed parameter space for WADD (gamma in [1.0, 20.0]), the exponentiated sum of Option B's coalition strictly outweighs Option A's single best cue plus its low-validity 'garbage' cues. Thus, WADD deterministically predicts a preference for Option B, while the Strategy Mixture deterministically predicts a preference for Option A. We include inverted versions and crossover trials to fully map the parameter space and rule out alternative heuristics.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design leverages an invariance property of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the Probabilistic Cue Selection (Random Dictator) model. In the Strategy Mixture model, the choice probability relies solely on the identity of the single best discriminating cue (for TTB) and the total count of winning cues (for Tallying). By holding the highest-validity cue and the total cue-win counts constant across a subset of trials, the Strategy Mixture model is forced to predict the exact same choice probabilities. However, the Random Dictator model samples among all discriminating cues proportionally to their non-linearly scaled validities. By shifting which specific lower-validity cues belong to Option A versus Option B across these trials, we change the underlying validity distributions. Consequently, the Random Dictator model predicts systematically shifting choice probabilities across trials where the Strategy Mixture model predicts strict equivalence.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
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
      1
    ],
    [
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
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
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
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
      0
    ],
    [
      0,
      1,
      1,
      0,
      1
    ],
    [
      0,
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
  "prompt_token_count": 3948,
  "candidates_token_count": 467,
  "total_token_count": 7750
}
```
