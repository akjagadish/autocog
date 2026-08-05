# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Heuristic Decision Making domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Choose `rating_max` (>= 1) to set the upper bound of each rating value: rating_max=1 gives binary ratings (but note that Tallying ≡ Equal-Weight on binary features), while rating_max >= 2 gives cardinal ratings that dissociate magnitude-based heuristics (EW, WADD) from sign-based ones (Tallying). Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, rating_max]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities, rating_max, and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see a set of ratings for each product. Each rating comes from an expert and is an integer between 0 and an upper bound fixed for this experiment (higher = more positive). The experts differ in how well they typically predict product quality, and their accuracies are communicated before the experiment begins; they are always shown in the same order. On every trial, decide which product you believe is of higher quality and press F for product A or J for product B. There is no time limit, but please respond as accurately as you can. There is no feedback during the task: try to make the best inference you can from the ratings and the experts' accuracies. Press SPACE when you are ready to begin.

## ADVOCATED THEORY
**Description:** Human decision-making in multi-attribute choice relies on a mixture of two boundedly rational heuristics that both utilize cue validities but process feature magnitudes differently. Rather than using full compensatory integration (WADD) or unweighted counting (Tallying), individuals draw from a mixture of 'Take-The-Best' (TTB) and 'Weighted Tallying'. TTB is a lexicographic strategy that bases the choice solely on the highest-validity cue that discriminates between the options. Weighted Tallying integrates information across all cues by binarizing feature differences into strict wins/losses and weighting these binary outcomes by their respective validities. Instead of probabilistically switching between these heuristics, decision-makers integrate the evidence (scores) from both strategies into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus Weighted Tallying evidence, and response noise enters through a single softmax over the mixed scores.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Weighted Tallying Heuristic
    # Binarize feature differences into wins/losses, then weight by validity
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    wt_scores = np.array([np.sum(a_wins * w), np.sum(b_wins * w)])
    
    # Normalize weighted tally scores so they occupy a similar [0, 1] scale as TTB
    w_sum = np.sum(w)
    if w_sum > 0:
        wt_scores = wt_scores / w_sum
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * wt_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Human decision-making in multi-attribute choice is not governed by a single universal heuristic. Instead, individuals differ in their strategies or switch between them, such that the population's choices reflect a mixture of Tallying (which counts strict feature-wise wins and ignores magnitudes and validities) and Weighted Additive (WADD, which integrates both magnitudes and validities). A mixture weight parameter 'alpha' dictates the probability of using Tallying versus WADD on any given trial. Response noise enters through a softmax over the scores of the chosen heuristic, plus an independent lapse rate.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Tallying heuristic
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # WADD heuristic
    w = np.asarray(parameters["validities"], dtype=float)
    wadd_scores = stim @ w
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of heuristics
    alpha = float(parameters["alpha"])
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    return p_mixed
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    return int(np.argmax(probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design dissociates Tallying (advocated) from Weighted Additive (WADD; competing) by exploiting Tallying's insensitivity to both feature validities and cardinal rating magnitudes. We use a 4-feature design with a high rating_max (10) and a spread of validities. In the critical trials, one option wins on a majority of features by a very small margin (which drives Tallying to strongly prefer it), while the other option wins on fewer, higher-validity features by a massive margin (which drives WADD to strongly prefer the latter).

[1] This design dissociates the advocated Weighted Additive (WADD) theory from the competing Tallying heuristic. WADD integrates both the cardinal magnitudes of features and their validities, while Tallying relies solely on the sign of the difference (counting strict feature-wise wins) and ignores both magnitude and validity. By using a cardinal rating scale (rating_max = 10) and a spread of validities, we construct trials where an option wins on a majority of features by a small margin (favored by Tallying) but loses on fewer features by a massive margin or on higher-validity features (favored by WADD). We also include trials where Tallying predicts a tie (equal number of feature wins) but WADD strongly prefers one option due to validity and magnitude differences.

[2] To quantitatively dissociate the advocated mixture model (Tallying + WADD) from the competing pure WADD model, this design systematically pits the two heuristics against each other using cardinal ratings. We include trials where Tallying and WADD strongly disagree (e.g., one option wins by a small margin on many low-validity features, while the other wins by a massive margin on a single high-validity feature). Because the competing model relies entirely on WADD, it will predict extreme choice probabilities for the option with the higher weighted sum. In contrast, the advocated model includes a mixture of Tallying, which will pull the predicted choice probabilities toward the option with more feature-wise wins (or toward 0.5 in case of a Tallying tie). By including trials where the heuristics agree, disagree, and where Tallying predicts a tie but WADD does not, we can isolate the unique variance explained by the Tallying component.

[3] To quantitatively dissociate the advocated pure WADD theory from the competing Mixture (Tallying + WADD) theory, we exploit trials where the two models make qualitatively different predictions about indifference. Specifically, we construct 'WADD-tie' trials where the weighted sums of the two options are exactly identical, meaning pure WADD predicts a 50/50 guess. However, on these same trials, one option strictly wins on a majority of features, meaning Tallying strongly prefers it. The competing mixture model predicts a significant deviation from 50/50 towards the Tallying winner, whereas the advocated pure WADD model predicts strict indifference. We also include 'Tallying-tie' trials where options win on an equal number of features (Tallying predicts 50/50) but the weighted sums differ, allowing us to observe if choice probabilities are dampened (as the mixture model predicts) or extreme (as pure WADD predicts).

[4] To quantitatively dissociate the advocated Mixture theory (Tallying + WADD) from the competing pure WADD theory, this design systematically varies the agreement between Tallying and WADD while carefully controlling the WADD score differences. The critical trials include 'WADD-ties' where the weighted sums are identical (pure WADD predicts 50/50 guessing) but Tallying strongly favors one option (Mixture predicts a preference for the Tallying winner). We also include trials where WADD and Tallying strongly conflict, and trials where Tallying predicts a tie but WADD does not. This allows us to estimate the mixture parameter 'alpha' and identify the presence of Tallying, which pure WADD cannot account for.

[5] To quantitatively dissociate the advocated pure WADD theory from the competing Mixture (Tallying + WADD) theory, we use a 5-feature design with cardinal ratings to precisely control WADD scores and Tallying wins. We construct exact 'WADD-tie' trials where the weighted sums of the two options are identical (e.g., 16 vs 16), meaning pure WADD predicts a 50/50 guess, but Tallying strongly favors one option (3 wins vs 2 wins). The mixture model predicts a significant deviation from 50/50 towards the Tallying winner, whereas pure WADD predicts strict indifference. We also include 'Tallying-tie' trials where options win on an equal number of features but WADD strongly prefers one, and conflict trials where WADD and Tallying favor opposite options. This allows us to test if the choice probabilities are strictly proportional to WADD score differences or if they are systematically pulled by Tallying wins.

[6] This design dissociates the advocated theory (Mixture of Tallying and WADD) from the competing theory (Mixture of TTB and Weighted Tallying) by exploiting a fundamental difference in how they process feature magnitudes. The competing theory is entirely sign-based: both TTB and Weighted Tallying only consider strict feature-wise wins and losses (binarization), completely ignoring the cardinal magnitude of those differences. In contrast, the advocated theory includes WADD, which integrates the cardinal magnitudes of the ratings. By keeping the winner of each feature constant but drastically varying the magnitude of the rating differences across trials, the competing theory predicts identical choice probabilities across these variations. The advocated theory, however, predicts large shifts in choice probabilities as the magnitude of the differences scales up, allowing for a sharp quantitative dissociation.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- rating_max: Upper bound (inclusive) of each rating value. Ratings are integers in [0, rating_max]. Choose rating_max=1 for binary ratings (note: Tallying and Equal-Weight are behaviorally identical on binary features — if you want to dissociate magnitude-based from sign-based heuristics, choose rating_max >= 2 for cardinal ratings).
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is an integer in [0, rating_max]. `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the advocated theory (Mixture of TTB and Weighted Tallying) from the competing theory (Mixture of Tallying and WADD), this design exploits a fundamental difference in how they process cardinal magnitudes. The advocated theory is entirely sign-based: both TTB and Weighted Tallying binarize feature differences into strict wins and losses, completely ignoring the magnitude of those differences. In contrast, the competing theory includes WADD, which integrates the exact cardinal magnitudes. We construct critical 'double-must' dissociation trials where Option A wins on the two highest-validity cues by a tiny margin (e.g., 1 point), while Option B wins on the three lowest-validity cues by a massive margin (e.g., 5 points). For the advocated theory, both heuristics strictly prefer A: TTB selects A based on the highest-validity cue, and Weighted Tallying prefers A because the combined validity of A's two wins (1.0 + 0.9 = 1.9) exceeds B's three wins (0.6 + 0.55 + 0.5 = 1.65). Thus, the advocated model MUST predict A. However, for the competing theory, both heuristics strictly prefer B: Tallying prefers B because it wins on more features (3 vs 2), and WADD strongly prefers B because the massive magnitude of B's wins easily overcomes A's tiny advantage on the higher-validity cues. Thus, the competing model MUST predict B. This guarantees a pure qualitative and quantitative dissociation regardless of either model's mixture parameter.",
  "validities": [
    1.0,
    0.9,
    0.6,
    0.55,
    0.5
  ],
  "rating_max": 5,
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      5,
      5,
      5
    ],
    [
      2,
      2,
      0,
      0,
      0
    ],
    [
      1,
      1,
      5,
      5,
      5
    ],
    [
      3,
      1,
      0,
      0,
      0
    ],
    [
      2,
      0,
      4,
      4,
      4
    ],
    [
      1,
      2,
      0,
      0,
      0
    ],
    [
      0,
      1,
      3,
      3,
      3
    ],
    [
      5,
      5,
      5,
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
      4,
      4,
      4,
      1,
      1
    ],
    [
      1,
      1,
      1,
      2,
      2
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      5,
      5,
      5
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
      1,
      5,
      5,
      5
    ],
    [
      2,
      2,
      0,
      0,
      0
    ],
    [
      2,
      0,
      4,
      4,
      4
    ],
    [
      3,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      3,
      3,
      3
    ],
    [
      1,
      2,
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
      5,
      5,
      5,
      0,
      0
    ],
    [
      1,
      1,
      1,
      2,
      2
    ],
    [
      4,
      4,
      4,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3807,
  "candidates_token_count": 833,
  "total_token_count": 11841
}
```
