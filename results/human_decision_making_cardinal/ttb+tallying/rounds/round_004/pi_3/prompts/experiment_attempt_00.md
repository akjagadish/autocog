# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Heuristic Decision Making domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0] (they are stored **descending**, highest first) — to fix each expert's advertised accuracy; subjects are told these values up front. Choose `rating_max` (>= 1) to set the upper bound of each rating value: rating_max=1 gives binary ratings (but note that Tallying ≡ Equal-Weight on binary features), while rating_max >= 2 gives cardinal ratings that dissociate magnitude-based heuristics (EW, WADD) from sign-based ones (Tallying). Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, rating_max]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities, rating_max, and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, `rating_max`]. The features are NOT necessarily binary: if `rating_max=1` the rating is binary (0 / 1), but if `rating_max>=2` the rating is a cardinal score on a 0..`rating_max` scale and ratings are displayed as a horizontal filled bar with the numeric value (e.g. "4/7") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no trial-by-trial feedback, but at the end of the task subjects see a single summary screen with the percentage of their choices that matched the Lee & Cummins (2004) Bayesian normative benchmark. Subjects are instructed to keep the experiment tab in focus throughout; cumulative blur time is tracked by the BotDetection layer and used post-hoc to exclude participants who switched away for too long. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

**Parameters:**
- beta: [0.01, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Threshold-based Binarization (Satisficing WADD): Decision-makers simplify complex cardinal information by converting continuous or multi-level ratings into binary cues based on a satisficing threshold. A feature is considered satisfactory (1) if its rating meets or exceeds the threshold, and unsatisfactory (0) otherwise. The options are then evaluated by computing the validity-weighted sum of these binarized features (WADD on binary cues). This mechanism naturally explains why extreme cardinal advantages (e.g., 10 vs 5) might be ignored if both options exceed the satisficing threshold, allowing an option with distributed moderate advantages to win against an option with a single extreme advantage.

**Parameters:**
- threshold: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize validities
    sum_val = np.sum(val)
    if sum_val > 0:
        val = val / sum_val
        
    # Binarize features based on the threshold
    bin_a = (a >= threshold).astype(float)
    bin_b = (b >= threshold).astype(float)
    
    # Compute validity-weighted sum of binarized features
    score_a = np.sum(bin_a * val)
    score_b = np.sum(bin_b * val)
    
    scores = np.array([score_a, score_b])
    
    # Convert to probabilities via softmax with max-subtraction
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the most valid cue (or the first discriminating cue) points to one option, while the simple majority of features point to the other. TTB decides solely based on the highest-validity discriminating cue, ignoring all others. Tallying, on the other hand, ignores validities and simply counts the number of features on which each option is superior. We include trials where TTB and Tallying make strictly opposing predictions, as well as trials where the top cue ties and the conflict cascades to the second cue.

[1] To quantitatively dissociate Tallying from Take The Best (TTB), we present a 5-feature design using cardinal ratings (0-5 scale). TTB is a non-compensatory, one-reason decision rule that stops at the first discriminating feature, ordered by validity. Tallying ignores both the feature validities and the cardinal magnitude of the differences, counting only the number of features on which an option is strictly better. We construct trials where the highest-validity discriminating cue points to one option, but a simple majority of the remaining lower-validity cues point to the other option. By including trials where the top cue discriminates, as well as trials where the top cues tie and the conflict cascades to the 2nd or 3rd cue, we can isolate whether subjects are using a single-feature TTB strategy or a multi-feature Tallying strategy.

[2] To quantitatively dissociate WADD from Tallying, we exploit WADD's sensitivity to both feature validities and cardinal rating magnitudes. Tallying, by contrast, only counts strict feature-wise wins, ignoring both the sizes of the differences and the validities of the features. We use a 4-feature design with a high rating_max (10) and varied validities. The trials are constructed such that one option wins on a single feature by a large margin (strongly favored by WADD), while the other option wins on multiple features by small margins (favored by Tallying). We also include trials where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers one option due to magnitude and validity differences.

[3] To quantitatively dissociate Tallying from Weighted Additive (WADD) models, we use a 5-feature design with cardinal ratings (up to 10) and varying validities. Tallying ignores both the magnitude of the differences between ratings and the validities of the features, simply counting the number of features on which one option is strictly better than the other. WADD, however, integrates the cardinal magnitudes weighted by the validities. We design trials where one option has a small advantage on a majority of features (favored by Tallying) but the other option has a massive advantage on one or two high-validity features (favored by WADD). We also include trials where Tallying predicts a tie but WADD strongly prefers one option, as well as baseline trials where both models agree.

[4] To dissociate the purely compensatory Weighted Additive (WADD) theory from the Dynamic Strategy Selection (Threshold) model, we manipulate the maximum cardinal difference ('max_diff') between options across features. The Threshold model predicts that subjects will use Tallying when the maximum feature difference is small (falling below their threshold) and switch to WADD only when a feature difference is large enough to cross the threshold. In contrast, the pure WADD model integrates validity-weighted cardinal differences on every trial, regardless of the maximum difference. We construct a 5-feature design where WADD and Tallying consistently make opposing predictions: one option wins on the two most valid cues, while the other option wins on the three less valid cues. By parametrically varying the magnitude of the differences (max_diff = 1, 2, 4, and 8), we can observe whether subjects' choices flip from Tallying to WADD as the difference grows (supporting the Threshold model) or remain consistent with WADD throughout (supporting the advocated pure WADD theory).

[5] To quantitatively dissociate the Dynamic Strategy Selection (Threshold) model from the pure Weighted Additive (WADD) model, we construct a 5-feature design where the two models make diverging predictions depending on the maximum cardinal difference between the options. We design conflict trials where Option A wins on the two most valid features (sum of validities = 1.9), while Option B wins on the three least valid features (sum of validities = 1.6). Because WADD integrates validity-weighted magnitudes, it consistently prefers Option A on these trials regardless of the scale of the difference. Tallying, conversely, simply counts feature wins and consistently prefers Option B (3 wins to 2). By parametrically varying the magnitude of the difference (max_diff from 1 to 10), we can test the Threshold model's prediction: subjects will prefer Option B (Tallying) when the difference is small and below their threshold, but switch to Option A (WADD) when the difference is large. The pure WADD model predicts a consistent preference for Option A across all magnitudes.

[6] To quantitatively dissociate the advocated Linear Weighted Additive (WADD) theory from the competing Diminishing Returns WADD theory, we exploit their differing treatments of cardinal magnitude baselines. Linear WADD assumes constant marginal utility: a rating difference of 2 points has the same impact whether it occurs at the top of the scale (e.g., 10 vs 8) or the bottom (e.g., 2 vs 0). Diminishing Returns WADD applies a concave transformation, meaning differences at the low end of the scale have a dramatically larger impact on choice than equivalent differences at the high end. We construct trials where Option A wins on the most valid feature at the top of the scale, while Option B wins on the second most valid feature at the bottom of the scale by the exact same cardinal difference. Linear WADD consistently predicts Option A due to the higher validity weighting, while Diminishing Returns WADD consistently predicts Option B because the marginal utility of the low-end difference outweighs the high-end difference despite the validity penalty.

[7] To quantitatively dissociate the advocated Diminishing Returns WADD theory from the competing Linear Weighted Additive (WADD) theory, we exploit their differing treatments of concentrated versus distributed cardinal advantages. Linear WADD assumes constant marginal utility, meaning a single large advantage on a high-validity feature can easily outweigh several smaller advantages on lower-validity features, provided the linear sum is greater. In contrast, Diminishing Returns WADD applies a concave utility transformation to the ratings, heavily penalizing large, concentrated differences while rewarding distributed, smaller differences. We construct conflict trials where Option A has a single massive advantage on a highly valid feature, while Option B has multiple moderate advantages on less valid features. The ratings are finely tuned such that the linear validity-weighted sum slightly favors Option A (leading Linear WADD to predict A), but the concave transformation causes Diminishing Returns WADD to strongly favor Option B. Control trials where both models agree are also included.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]. On load, the list is reordered to **descending** by validity (highest first) and the same permutation is applied to every `trial_a_ratings` / `trial_b_ratings` row, so Expert 1 is always the most valid. Length determines n_features and must match every rating list. Pick a spread (at least one high and one low) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule.
- rating_max: Upper bound (inclusive) of each rating value. Ratings are integers in [0, rating_max]. Choose rating_max=1 for binary ratings (note: Tallying and Equal-Weight are behaviorally identical on binary features — if you want to dissociate magnitude-based from sign-based heuristics, choose rating_max >= 2 for cardinal ratings).
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is an integer in [0, rating_max]. `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the pure Weighted Additive (WADD) theory from the Threshold-based Binarization (Satisficing WADD) theory, we exploit their divergent treatment of cardinal rating magnitudes. Pure WADD integrates all cardinal differences linearly, weighted by validity. Satisficing WADD, however, binarizes ratings based on an internal threshold, ignoring magnitudes once ratings are on the same side of the threshold. We construct a 5-feature design with a rating scale up to 10 and systematically vary the 'plateau' of ratings across trials. In one set of trials (1-3), Option A has a massive advantage on the two most valid features but a slight disadvantage on the remaining three, such that pure WADD strongly prefers A. However, if a subject's satisficing threshold falls between the lower and higher ratings, Satisficing WADD will prefer Option B because B crosses the threshold on all 5 features while A only crosses it on 2. In the reverse set of trials (4-6), the pattern is flipped to catch other potential threshold values. By probing multiple potential threshold boundaries (e.g., 4, 6, 8), we can determine whether subjects continuously integrate magnitudes (supporting pure WADD) or exhibit step-function preference reversals indicative of binarization (supporting Satisficing WADD).",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
  ],
  "rating_max": 10,
  "trial_a_ratings": [
    [
      10,
      10,
      2,
      2,
      2
    ],
    [
      10,
      10,
      4,
      4,
      4
    ],
    [
      10,
      10,
      7,
      7,
      7
    ],
    [
      8,
      8,
      8,
      8,
      8
    ],
    [
      6,
      6,
      6,
      6,
      6
    ],
    [
      4,
      4,
      4,
      4,
      4
    ]
  ],
  "trial_b_ratings": [
    [
      5,
      5,
      5,
      5,
      5
    ],
    [
      7,
      7,
      7,
      7,
      7
    ],
    [
      9,
      9,
      9,
      9,
      9
    ],
    [
      10,
      10,
      7,
      7,
      7
    ],
    [
      10,
      10,
      5,
      5,
      5
    ],
    [
      10,
      10,
      3,
      3,
      3
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4162,
  "candidates_token_count": 575,
  "total_token_count": 8335
}
```
