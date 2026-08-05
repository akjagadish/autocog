# experiment_attempt_00

## System Prompt

You are a neutral experimental designer in computational cognitive science, working in the Decision Making (Binary Features) domain.

You are given two candidate theories of decision making, THEORY 1 and THEORY 2, each operationalized as code. Your only goal is to design an experiment whose data will best distinguish the two theories — you have no stake in either theory.

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

## THEORY 1
**Description:** Strict Random Guessing: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects completely disengage from the task. They ignore the stimulus features, spatial positions, and their own choice history, instead falling back on a pure uniform random guessing strategy (flipping a coin) on every single trial.

**Parameters:**
(none)

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # Predict exactly 0.5 for both options regardless of stimulus or history
    return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## THEORY 2
**Description:** Position Bias / Task Disengagement: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects often disengage from effortful feature integration or search. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., a tendency to select the first option, Option A) or random guessing. Decisions are driven entirely by these spatial/temporal biases, and the actual feature validities and values are ignored.

**Parameters:**
- bias_a: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the stimulus and history entirely.
    # It predicts based solely on a spatial position bias for Option A.
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To effectively dissociate Take The Best (TTB) from Tallying, we use a 5-feature design where the cues have strictly descending validities. The trials are constructed such that the single highest-validity discriminating cue always favors one option, while a numerical majority of lower-validity cues favor the alternative option. Because TTB stops at the first discriminating cue, it will systematically choose the option favored by the single best cue. In contrast, Tallying ignores cue validities and simply counts the number of winning features, leading it to systematically choose the opposite option.

[1] To robustly dissociate Take The Best (TTB) from Tallying, this experiment uses a 6-feature design. We systematically vary which cue is the highest-validity discriminating cue (e.g., the 1st, 2nd, or 3rd most valid cue) by introducing ties on the higher-validity cues. In every trial, the first discriminating cue strongly favors one option (which TTB will choose), while a sheer numerical majority of the remaining lower-validity cues favor the opposite option (which Tallying will choose). By shifting the decisive cue down the validity hierarchy across trials, we also ensure that TTB cannot be mimicked by a simple 'always look at cue 1' heuristic, and we test the stopping rule of TTB against the exhaustive compensatory counting of Tallying.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use a 5-feature design with closely spaced, high validities. In each trial, the options are designed such that the single highest-validity discriminating cue strongly favors one option, which TTB will reliably choose since it ignores all remaining cues. However, a coalition of multiple lower-validity cues favors the alternative option. Because the validities are relatively close to each other, the sum of these lower-validity cues (even after being non-linearly transformed by WADD's gamma parameter, up to gamma=5) will outweigh the single best cue. Thus, WADD will systematically choose the opposite option to TTB across a wide range of parameter values.

[3] To cleanly dissociate Take The Best (TTB) from the Weighted Additive (WADD) strategy, we use 5 closely spaced, high-validity cues. In each trial, the single best discriminating cue strongly favors one option, meaning TTB will invariably select it. However, the alternative option is favored by a coalition of two or three slightly less valid cues. Because the validities are closely spaced (0.82 to 0.90), the sum of the validities for the alternative option will strictly outweigh the single best cue, even if WADD applies a strong non-linear transformation (up to gamma=5). Thus, TTB and WADD yield diametrically opposed predictions across all trials.

[4] To maximally dissociate Theory 1 (Position Bias) from Theory 2 (Weighted Additive), we need to manipulate the stimulus such that the objectively superior option alternates between Option A and Option B. Theory 2's predictions will track the feature values, swinging strongly from preferring A to preferring B depending on which option has higher weighted validities. In stark contrast, Theory 1 ignores the stimulus completely and predicts a constant probability of choosing Option A across all trials. By including extreme cases (e.g., A has all positive features and B has none, and vice versa) as well as closer trade-offs, we ensure that Theory 2 produces highly variable choice probabilities across trials while Theory 1 remains perfectly flat.

[5] To dissociate a Weighted Additive (WADD) strategy from a Position Bias strategy, this design varies the objective superiority of Option A versus Option B across trials. WADD predicts that the probability of choosing Option A will shift dramatically depending on which option has the higher weighted sum of features. In contrast, the Position Bias theory assumes subjects ignore the stimulus completely, predicting a constant probability of choosing A across all trials regardless of the feature values. By including extreme differences as well as moderate trade-offs, we ensure WADD produces highly variable choice probabilities while Position Bias remains perfectly flat.

[6] Both Theory 1 (Position Bias) and Theory 2 (Sequential Dependency) assume that subjects ignore the stimulus features entirely. Theory 1 predicts a constant probability of choosing Option A across all trials, independent of past choices. Theory 2 predicts that the choice on the current trial depends strictly on the choice made on the previous trial (e.g., a tendency to repeat or alternate). Because both theories ignore the stimulus, the specific feature values do not drive the dissociation. Instead, the models are distinguished by analyzing the sequential dependencies in the response data. We provide a standard set of multi-attribute choice trials to serve as a plausible task environment, ensuring subjects have a reason to make choices while we measure their disengagement strategies.

[7] Both Theory 1 (Sequential Dependency) and Theory 2 (Position Bias) assume that subjects completely ignore the stimulus features. Theory 1 predicts that the probability of choosing Option A depends on the previous choice (e.g., repeating or alternating), whereas Theory 2 predicts a constant probability of choosing Option A across all trials, regardless of history. Since the stimulus features themselves do not drive the dissociation, we provide a standard, varied set of multi-attribute choice trials to serve as a plausible task environment. The actual dissociation will come from analyzing the sequential dependence of the subjects' responses.

[8] Both Theory 1 (Position Bias) and Theory 2 (Strict Random Guessing) assume that subjects ignore the stimulus features entirely. Theory 1 allows for an asymmetric preference for one spatial position over the other (e.g., choosing Option A 80% of the time), whereas Theory 2 strictly predicts a 50/50 split on every trial. To ensure that any observed position bias is purely a spatial/response bias and not an artifact of the stimulus, we provide a perfectly balanced set of trials. For every trial where Option A has a given set of features and Option B has another, there is a mirror-image trial where the features are swapped. This guarantees that the objective quality of Option A and Option B is identical over the course of the experiment.

## RESPONSE FORMAT
Design one new experiment that maximally distinguishes THEORY 1 from THEORY 2.
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To definitively distinguish Strict Random Guessing from Position Bias without confounding spatial preference with stimulus features, we present a series of 'Buridan's ass' trials where Option A and Option B are objectively identical in all features. Because there is no stimulus-based reason to prefer one option over the other on any trial, any consistent overall preference for Option A must be driven by a spatial position bias (Theory 2). Theory 1 predicts a strict 50/50 split across these trials, whereas Theory 2 can accommodate an asymmetric preference for one side when facing identical choices.",
  "validities": [
    0.9,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      0,
      1
    ],
    [
      0,
      1,
      1
    ],
    [
      1,
      1,
      1
    ],
    [
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      0,
      1
    ],
    [
      0,
      1,
      1
    ],
    [
      1,
      1,
      1
    ],
    [
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
  "prompt_token_count": 2890,
  "candidates_token_count": 356,
  "total_token_count": 4767
}
```
