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
**Description:** Relative Evidence Accumulation with Conflict Discounting: Decision-makers evaluate options by accumulating evidence from their features, but this evidence is dynamically discounted by the presence of conflicting features in the competing option (cross-option inhibition). This avoids perverse self-dilution (adding cues to an option never hurts it) while capturing conflict-driven non-linearities and context effects through mutual suppression.

**Parameters:**
- alpha: [0.0, 5.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of cue validities (centered at chance 0.5)
    w = np.maximum(0.0, val - 0.5) ** alpha
    
    # Cross-option conflict: number of features present in the competitor but absent in the focal option
    conflict_a = np.sum(b * (1.0 - a))
    conflict_b = np.sum(a * (1.0 - b))
    
    # Evidence accumulation with exponential conflict discounting
    ev_a = np.sum(a * w) * np.exp(-gamma * conflict_a)
    ev_b = np.sum(b * w) * np.exp(-gamma * conflict_b)
    
    # Softmax choice rule
    z = beta * np.array([ev_a, ev_b])
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Lateral Inhibition Evidence Accumulator (Extreme Dilution): Subjects evaluate options by comparing their feature vectors holistically, where the evidence provided by each cue is dynamically suppressed by other active cues via divisive normalization (lateral inhibition). Supporting secondary cues dilute the perceived value of the primary cue (self-inhibition), while conflicting cues from the alternative option also suppress evidence. This non-linear squashing before integration naturally produces strong non-monotonic and 'perverse' effects, where having too many secondary cues can paradoxically weaken an option's overall appeal compared to an option with a single strong primary cue.

**Parameters:**
- alpha: [0.0, 5000.0]
- beta: [0.0, 100.0]
- gamma: [0.1, 5.0]
- theta: [0.1, 500.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    # Transform validities into base weights
    w = np.power(np.maximum(val - 0.5, 0.0), gamma)
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    
    e_a = 0.0
    e_b = 0.0
    
    sum_w_a = np.sum(w * a)
    sum_w_b = np.sum(w * b)
    
    # Calculate laterally inhibited evidence for each option
    for i in range(len(a)):
        if a[i] > 0:
            other_a = sum_w_a - w[i] * a[i]
            conflicting_b = sum_w_b
            # Divisive normalization: self-dilution (alpha) + conflict suppression (beta)
            denom = 1.0 + alpha * other_a + beta * conflicting_b
            e_a += (w[i] * a[i]) / denom
            
        if b[i] > 0:
            other_b = sum_w_b - w[i] * b[i]
            conflicting_a = sum_w_a
            denom = 1.0 + alpha * other_b + beta * conflicting_a
            e_b += (w[i] * b[i]) / denom
            
    theta = float(parameters["theta"])
    z = theta * np.array([e_a, e_b])
    # Numerically stable softmax
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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

[6] This design leverages an invariance property of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the Probabilistic Cue Selection (Random Dictator) model. In the Strategy Mixture model, the choice probability relies solely on the identity of the single best discriminating cue (for TTB) and the total count of winning cues (for Tallying). By holding the highest-validity cue and the total cue-win counts constant across a subset of trials, the Strategy Mixture model is forced to predict the exact same choice probabilities. However, the Random Dictator model samples among all discriminating cues proportionally to their non-linearly scaled validities. By shifting which specific lower-validity cues belong to Option A versus Option B across these trials, we change the underlying validity distributions. Consequently, the Random Dictator model predicts systematically shifting choice probabilities across trials where the Strategy Mixture model predicts strict equivalence.

[7] This design introduces a 'Dilution Effect' to quantitatively dissociate the Probabilistic Cue Selection (Random Dictator) model from the Strategy Mixture (TTB + Tallying) model. The Mixture model's choice probabilities depend solely on the identity of the single best discriminating cue (TTB) and the absolute difference in the number of winning cues (Tallying). By symmetrically adding lower-validity discriminating cues to both options across a series of trials (e.g., Trials 1, 2, and 3), we hold both the best discriminating cue and the tally difference strictly constant. Consequently, the Strategy Mixture model is mathematically forced to predict identical choice probabilities across these trials. In stark contrast, the Random Dictator model samples from the set of all discriminating cues. Adding more discriminating cues dilutes the sampling probability of the highest-validity cue, systematically pulling the choice probabilities toward 0.5. This provides a clear, parameter-independent structural divergence between the two theories.

[8] This design quantitatively dissociates the Strategy Mixture (TTB + Tallying) model from the Weighted Additive with Diminishing Returns (WADD-DR) model. WADD-DR applies a concave transformation (alpha < 1) to the total accumulated evidence for each option, meaning that additional winning cues provide diminishing marginal returns. In contrast, the Strategy Mixture model combines a strictly non-compensatory component (TTB, which only considers the single best discriminating cue) and a strictly linear component (Tallying, which counts net feature wins without diminishing returns). By systematically pitting the highest-validity cue against coalitions of lower-validity cues of varying sizes, WADD-DR predicts compressed probability margins due to its sub-additive evidence transformation. The Mixture model, constrained by linear tallying and strict TTB, cannot mimic these specific non-linear shifts in choice probabilities.

[9] This design quantitatively dissociates the Weighted Additive with Diminishing Returns (WADD-DR) model from the Strategy Mixture (TTB + Tallying) model by exploiting the strict invariance of the Mixture model to symmetric evidence addition. In the Strategy Mixture model, choice probabilities depend exclusively on the identity of the highest-validity discriminating cue (for TTB) and the absolute difference in the number of winning cues (for Tallying). By adding pairs of equal-validity discriminating cues symmetrically to Option A and Option B across paired trials, we hold both the best discriminating cue and the tally difference strictly constant. Consequently, the Strategy Mixture model is mathematically forced to predict identical choice probabilities across these trial pairs. In contrast, WADD-DR accumulates the total evidence and applies a concave transformation (alpha < 1). Adding symmetric evidence to both options pushes them to a flatter region of the concave curve, resulting in a compression of the evidence difference and predicting a systematic shift in choice probabilities toward 0.5. This provides a parameter-free structural divergence between the two theories.

[10] This design leverages an invariance property of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the Sequential Evidence Accumulation model. In the Strategy Mixture model, choice probabilities depend solely on the identity of the single best discriminating cue (TTB) and the total count of winning cues (Tallying). By holding the highest-validity cue and the total cue-win counts constant across a subset of trials (e.g., Trials 1, 2, and 3), the Strategy Mixture model is mathematically forced to predict the exact same choice probabilities. However, the Sequential Evidence Accumulation model integrates the actual validities of the specific lower-tier cues until a threshold is reached. By shifting which specific lower-validity cues Option A vs. B wins across these trials, the accumulated evidence varies. Consequently, if the threshold is large enough to evaluate these cues, the Sequential model predicts systematically shifting choice probabilities across trials where the Strategy Mixture model predicts strict equivalence.

[11] This design leverages an invariance property of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the Sequential Evidence Accumulation model. In the Strategy Mixture model, choice probabilities depend solely on the identity of the highest-validity discriminating cue (for TTB) and the difference in the total count of winning cues (for Tallying). By holding the highest-validity discriminating cue and the total cue-win counts constant across a subset of trials (e.g., Trials 1-3, and Trials 4-6), the Strategy Mixture model is mathematically forced to predict the exact same choice probabilities within each subset. However, the Sequential Evidence Accumulation model integrates the actual validities of the specific lower-tier cues sequentially until a threshold is reached. By shifting which specific lower-validity cues Option B wins across these trials, the accumulated evidence varies. Consequently, if the threshold is large enough to evaluate these cues, the Sequential model predicts systematically shifting choice probabilities across trials where the Strategy Mixture model predicts strict equivalence.

[12] This design leverages a strict, parameter-free qualitative dissociation between the Conflict-Driven Strategy Selection (Advocated) model and the Sequential Evidence Accumulation (Competing) model. The Advocated model computes strategy weights based on the absolute difference in total cue counts (`diff_cues`). In trials where `diff_cues == 0` (i.e., both options have an identical number of positive cues) and the number of winning cues for each option is equal, the Advocated model is forced to rely 100% on Tallying. Because the tally scores are tied, it mathematically must predict a 50/50 choice probability, regardless of the cues' validities. In stark contrast, the Sequential model evaluates cues in strictly descending order of validity. When one option possesses the highest-validity cues and the other possesses the lowest-validity cues, the Sequential model will rapidly accumulate evidence for the former and either stop early or win the full sum, strictly predicting a preference for the option with higher validities. By including both 'zero conflict' tied-tally trials and 'high conflict' trials (where one option has the single best cue but the other has many lower cues), we can fully map the divergent predictions of both models across their parameter spaces.

[13] This design leverages the core structural difference between the Sequential Evidence Accumulation model (Advocated) and the Conflict-Driven Strategy Selection model (Competing). The Competing model relies purely on the difference in the total number of positive cues (`diff_cues`) to weight TTB versus Tallying. When `diff_cues = 0`, the Competing model relies 100% on Tallying. By designing trials where both options have an identical number of positive cues (and thus tie in Tallying) but differ dramatically in the validity of those cues, the Competing model is mathematically forced to predict a 50/50 choice probability. In contrast, the Sequential Evidence Accumulation model evaluates cues in order of validity, so it will consistently predict the option with the higher-validity cues. We also include high-conflict trials where one option wins the single best cue but loses all others, allowing the Sequential model to exhibit compensatory behavior (choosing the many lower cues if the threshold is high) while the Competing model is forced by the large `diff_cues` to rely heavily on TTB.

[14] This design creates a strict qualitative dissociation between the Conflict-Driven Strategy Selection (Advocated) model and the Recency-Biased Evidence Accumulation (Competing) model. The Advocated model relies on 'diff_cues' (absolute difference in total positive cues) to arbitrate between Take The Best (TTB) and Tallying. Crucially, when diff_cues == 0, the Advocated model relies 100% on Tallying. Because the features are binary, any trial where sum(a) == sum(b) mathematically guarantees a tied Tallying score, forcing the Advocated model to predict exactly 50/50 regardless of the cues' validities. The Competing model, however, completely ignores instructed validities and weights cues exponentially based on their left-to-right sequential position. For trials with diff_cues == 0 (e.g., A has the leftmost cue, B has the rightmost cue), the Competing model predicts strong preferences depending on its primacy/recency parameter (lambda). Furthermore, in high-conflict trials (large diff_cues), we pit the single highest-validity cue (placed in the middle of the array) against all other cues. The Advocated model's TTB component will reliably pick the option with the highest-validity cue, whereas the Competing model's exponential position-weighting mathematically guarantees that a middle cue can never outweigh the sum of the outer cues, forcing it to choose the opposite option.

[15] This design creates a strict qualitative dissociation between the Recency-Biased Evidence Accumulation (Advocated) model and the Conflict-Driven Strategy Selection (Competing) model. The Competing model relies on 'diff_cues' (absolute difference in total positive cues) to arbitrate between Take The Best (TTB) and Tallying. Crucially, when diff_cues == 0, the Competing model relies 100% on Tallying. Because the features are binary, any trial where sum(a) == sum(b) mathematically guarantees a tied Tallying score, forcing the Competing model to predict exactly 50/50 regardless of the cues' validities. The Advocated model, however, completely ignores instructed validities and weights cues exponentially based on their left-to-right sequential position. For trials with diff_cues == 0 (e.g., A has the outermost cues, B has the inner cues), the Advocated model predicts strong preferences depending on its primacy/recency parameter (lambda). Furthermore, in high-conflict trials (large diff_cues), we pit the single highest-validity cue (placed in the middle of the array) against all other cues. The Competing model's TTB component will reliably pick the option with the highest-validity cue, whereas the Advocated model's exponential position-weighting mathematically guarantees that a middle cue can never outweigh the sum of the outer cues, forcing it to choose the opposite option.

[16] This design leverages a strict, parameter-free qualitative dissociation between the Conflict-Driven Strategy Selection (Advocated) model and the Rank-Dependent Discriminating Weighting (Competing) model. The Advocated model relies on 'diff_cues' (the absolute difference in total positive cues) to arbitrate between Take The Best (TTB) and Tallying. Critically, when diff_cues == 0, the Advocated model relies 100% on Tallying. Since the features are binary, any trial where sum(a) == sum(b) mathematically guarantees a tied Tallying score, forcing the Advocated model to predict exactly 50/50 regardless of the cues' validities. The Competing model, however, evaluates discriminating cues ranked by validity. For zero-conflict trials where Option A wins on a high-validity cue and Option B wins on a lower-validity cue, the Competing model will strongly prefer Option A. We also include high-conflict trials where one option possesses the single highest-validity cue but the other possesses multiple lower-validity cues. Here, the Advocated model's TTB component is heavily weighted, pushing it toward the highest-validity option, while the Competing model can exhibit compensatory behavior and choose the option with more cues depending on its rank-decay parameter.

[17] This design provides a strict, parameter-free qualitative dissociation between the Rank-Dependent Discriminating Weighting (Advocated) model and the Conflict-Driven Strategy Selection (Competing) model. The Competing model uses the absolute difference in total cue counts ('diff_cues') to weight TTB versus Tallying. Critically, when diff_cues == 0, the Competing model relies 100% on Tallying. Because the features are binary, any trial where both options have the same total number of positive cues mathematically guarantees a tied Tallying score, forcing the Competing model to predict exactly 50/50 regardless of the cues' validities. The Advocated model, however, evaluates discriminating cues ranked by validity. For zero-conflict trials where Option A wins on a high-validity cue and Option B wins on a lower-validity cue, the Advocated model will strongly prefer Option A. We also include high-conflict trials where one option possesses the single highest-validity cue but the other possesses multiple lower-validity cues. Here, the Competing model's TTB component is heavily weighted by the large diff_cues, pushing it toward the highest-validity option, while the Advocated model can flexibly exhibit compensatory behavior depending on its rank-decay parameter.

[18] The Advocated model uses the absolute difference in total positive cues ('diff_cues') to arbitrate between Take The Best (TTB) and Tallying. Critically, when diff_cues == 0, the Advocated model relies 100% on Tallying. Since the features are binary, any trial where both options have the same total number of positive cues mathematically guarantees a tied Tallying score, forcing the Advocated model to predict exactly 50/50 regardless of the cues' validities. In contrast, the Competing model evaluates options based on a fractional Minkowski distance to an ideal exemplar (all ones), which is highly sensitive to the validities of the specific cues present. For zero-conflict trials where Option A possesses higher-validity cues than Option B, the Competing model will strongly prefer Option A. We also include high-conflict trials (where one option has the highest validity cue but the other has many more lower-validity cues) to map out the non-linear distance scaling of the Competing model against the TTB-dominated behavior of the Advocated model.

[19] This design provides a strict, parameter-free qualitative dissociation between the Exemplar-Based (Advocated) model and the Conflict-Driven Strategy Selection (Competing) model. The Competing model uses the absolute difference in total cue counts ('diff_cues') to weight TTB versus Tallying. Critically, when diff_cues == 0, the Competing model relies 100% on Tallying. Because the features are binary, any trial where both options have the same total number of positive cues mathematically guarantees a tied Tallying score, forcing the Competing model to predict exactly 50/50 regardless of the cues' validities. In contrast, the Advocated model evaluates options based on a fractional Minkowski distance to an ideal exemplar (all ones), which is highly sensitive to the validities of the specific cues present. For zero-conflict trials where Option A possesses higher-validity cues than Option B, the Advocated model will strongly prefer Option A. We also include high-conflict trials (where one option has the highest validity cue but the other has many more lower-validity cues) to map out the non-linear distance scaling of the Advocated model against the TTB-dominated behavior of the Competing model.

[20] This design provides a strict qualitative dissociation between the Conflict-Driven Strategy Selection (Advocated) model and the Context-Dependent Strategy Selection with Reverse TTB (Competing) model. The Advocated model relies on 'diff_cues' (absolute difference in total positive cues) to arbitrate between Take The Best (TTB) and Tallying. Critically, when diff_cues == 0, the Advocated model relies 100% on Tallying. Because the features are binary, any trial where sum(a) == sum(b) mathematically guarantees a tied Tallying score, forcing the Advocated model to predict exactly 50/50 regardless of the cues' validities. In contrast, the Competing model explicitly detects tied sums and boosts a 'Bottom-Up' (Reverse TTB) strategy, which evaluates cues from lowest to highest validity. For zero-conflict trials where Option A wins on the lowest-validity cue and Option B wins on the highest-validity cue, the Competing model will predict a preference for Option A, while the Advocated model is locked at 0.5. We also include high-conflict trials to map out the differential weighting of Standard TTB between the two models.

[21] This design provides a strict qualitative dissociation between the Context-Dependent Strategy Selection with Reverse TTB (Advocated) model and the Conflict-Driven Strategy Selection (Competing) model. The Competing model relies on 'diff_cues' (absolute difference in total positive cues) to arbitrate between Take The Best (TTB) and Tallying. Critically, when diff_cues == 0, the Competing model relies 100% on Tallying. Because the features are binary, any trial where sum(a) == sum(b) mathematically guarantees a tied Tallying score, forcing the Competing model to predict exactly 50/50 regardless of the cues' validities. In contrast, the Advocated model explicitly detects tied sums and boosts a 'Bottom-Up' (Reverse TTB) strategy, which evaluates cues from lowest to highest validity. For zero-conflict trials where Option A wins on the lowest-validity cue and Option B wins on the highest-validity cue, the Advocated model will predict a preference for Option A, while the Competing model is locked at 0.5. We also include high-conflict trials to map out the differential weighting of Standard TTB between the two models.

[22] This design achieves a strict, parameter-free qualitative dissociation between the Conflict-Driven Strategy Selection (Advocated) model and the Weighted Additive with Power-Function Validity Scaling (WADD-PWR) (Competing) model. The Advocated model relies on 'diff_cues' (absolute difference in total positive cues) to arbitrate between Take The Best (TTB) and Tallying. Critically, when diff_cues == 0, the Advocated model relies 100% on Tallying. Because the features are binary, any trial where sum(a) == sum(b) mathematically guarantees a tied Tallying score, forcing the Advocated model to predict exactly 50/50 regardless of the cues' validities. In contrast, the Competing model integrates all cues using a power function of their validities. For zero-conflict trials where Option A wins on the highest-validity cue and Option B wins on lower-validity cues, the Competing model will strongly prefer Option A (or B, depending on the gamma parameter), but will not predict 50/50. We also include high-conflict trials where one option possesses the single highest-validity cue but the other possesses multiple lower-validity cues, to map out the non-linear validity scaling of the Competing model against the TTB-dominated behavior of the Advocated model.

[23] This design provides a strict, parameter-free qualitative dissociation between the Weighted Additive with Power-Function Validity Scaling (WADD-PWR) model (Advocated) and the Conflict-Driven Strategy Selection model (Competing). The Competing model uses the absolute difference in total positive cues ('diff_cues') to arbitrate between Take The Best (TTB) and Tallying. Critically, when diff_cues == 0, the Competing model relies 100% on Tallying. Because the features are binary, any trial where both options have the same total number of positive cues mathematically guarantees a tied Tallying score, forcing the Competing model to predict exactly 50/50 regardless of the cues' validities. In contrast, the Advocated WADD-PWR model integrates all cues using a power function of their validities. For zero-conflict trials where Option A wins on the highest-validity cue and Option B wins on lower-validity cues, the Advocated model will strongly prefer Option A (or B, depending on the gamma parameter), but will not predict 50/50. We also include high-conflict trials where one option possesses the single highest-validity cue but the other possesses multiple lower-validity cues, to map out the non-linear validity scaling of the Advocated model against the TTB-dominated behavior of the Competing model.

[24] This design quantitatively dissociates the Advocated model (Take-The-Best with Compensatory Verification) from the Competing model (WADD-PWR) by exploiting a strict invariance in the Advocated model. The Advocated model evaluates the single best discriminating cue to determine the 'TTB winner' (assigning it a fixed weight, completely ignoring its actual validity), and then tallies the remaining cues evaluated *after* the discriminating cue. By systematically shifting a single tied cue or a single winning cue across the validity hierarchy, we can hold both the TTB winner and the tally difference strictly constant. For example, if Option A is all 1s and Option B has a single 1, shifting Option B's 1 from the highest-validity cue down to lower-validity cues changes the identity of the discriminating cue but perfectly preserves the tally difference (remaining_a - remaining_b = 3 in all cases). Consequently, the Advocated model is mathematically forced to predict identical choice probabilities across these trials. In stark contrast, the Competing model (WADD-PWR) integrates all cues using a power function of their specific validities. Shifting the location of the cues fundamentally alters the accumulated evidence, leading the Competing model to predict systematically shifting choice probabilities.

[25] This design quantitatively dissociates the Advocated model (WADD-PWR) from the Competing model (Take-The-Best with Compensatory Verification) by exploiting the Competing model's strict tallying of remaining cues. In the Competing model, once the highest-validity discriminating cue establishes the 'TTB winner', all subsequent cues are merely tallied, completely ignoring their specific validities. By creating a sequence of trials where the highest-validity discriminating cue and the net difference in the remaining winning cues are held strictly constant, the Competing model is mathematically forced to predict identical choice probabilities. However, by shifting which specific lower-validity cues belong to Option A versus Option B across these trials, we alter the underlying validity distribution. Consequently, the Advocated WADD-PWR model, which integrates all cues using a power function of their specific validities, predicts systematically shifting choice probabilities across trials where the Competing model predicts strict equivalence.

[26] This design quantitatively dissociates the Advocated theory (Take-The-Best with Compensatory Verification) from the Competing theory (Probabilistic Strategy Mixture) by exploiting the structural difference between score integration and probability mixing. In the Competing theory, the final choice probability is a mixture of the TTB and Tallying distributions. Consequently, if Option A wins the highest-validity cue, the probability of choosing Option A is strictly bounded from below by the probability of executing TTB; it can never fall below this asymptote, regardless of how many lower-validity cues Option B wins. In contrast, the Advocated theory integrates the TTB and Tallying evidence into a single score before the softmax choice rule. This compensatory mechanism allows a sufficiently large coalition of lower-validity cues for Option B to completely overwhelm Option A's TTB advantage, driving the probability of choosing Option A toward zero. By holding the TTB winner constant and systematically increasing the number of lower-validity cues Option B wins, we can observe whether the choice probabilities asymptote (supporting the Mixture model) or cross over and approach zero (supporting the Compensatory model).

[27] This design quantitatively dissociates the Probabilistic Strategy Mixture (Advocated) from Take-The-Best with Compensatory Verification (Competing) by exploiting the Competing model's sub-linear (square root) transformation of the tally difference. In the Advocated model, Tallying is strictly linear: the log-odds contribution of the Tallying component grows linearly with the absolute difference in the total number of winning cues. In the Competing model, the cues evaluated after the discriminating cue are tallied and then compressed via a square root function before being integrated with the TTB score. By creating a series of trials where Option A wins the highest-validity cue and an exponentially increasing number of lower-validity cues (e.g., +0, +1, +2, +4 advantages in remaining cues), the Advocated model predicts a constant, linear increase in the log-odds of choosing Option A for each additional cue. In contrast, the Competing model predicts a strongly diminishing marginal increase in log-odds due to its sub-linear compression. We also include reversed trials to fully map the symmetry and rule out alternative models.

[28] Both the Advocated model (CDLA) and the Competing model (Strategy Mixture) share a remarkable structural invariance: their predicted choice probabilities depend exclusively on the identity of the highest-validity discriminating cue (the TTB winner) and the absolute difference in the total number of winning cues (the Tally difference). However, they map this Tally difference to choice probabilities using fundamentally different functional forms. The Competing Mixture model uses a linear mixture of a constant TTB probability and a symmetric logistic Tallying curve. In contrast, the Advocated CDLA model modulates the log-odds evidence of the first cue via an exponential function of the 'contradiction' (which is strictly linearly related to the Tally difference). By holding the TTB winner constant (Option A always wins the highest validity cue) and systematically sweeping the Tally difference from extreme consonance (+5) to extreme contradiction (-5), we map out the response curves. The Mixture model is constrained to a symmetric, asymptoting logistic mixture, whereas CDLA's exponential modulation can produce highly asymmetric, non-linear, and even non-monotonic response curves as contradiction increases.

[29] This design provides a strict qualitative dissociation between the Probabilistic Strategy Mixture (Advocated) model and the Context-Dependent Lexicographic Accumulator (CDLA, Competing) model by exploiting CDLA's 'perverse' symmetric dispersion effect. In the Advocated model, choice probability is a linear mixture of TTB and Tallying. By holding the TTB winner constant (Option A always wins the highest-validity cue) and systematically sweeping the Tallying advantage from Option A (consonance) to Option B (contradiction), the Advocated model predicts a strictly monotonic shift in choice probability. In stark contrast, CDLA modulates the weight of the first discriminating cue using an exponential function of the contradiction. If the dispersion parameter is positive, CDLA predicts a perverse non-monotonicity: as Option A gains more supporting cues (consonance), the primary cue's weight exponentially decays, potentially decreasing the overall preference for A. Conversely, as Option B gains more cues (contradiction), the primary cue's weight exponentially explodes, making A more preferred. This allows CDLA to predict that Option A is chosen more often when it has fewer supporting cues, a behavior mathematically impossible for the Advocated Mixture model.

[30] This design quantitatively dissociates the Lateral Inhibition Evidence Accumulator (Advocated) from the Probabilistic Strategy Mixture (Competing) by exploiting the Advocated model's 'perverse' self-dilution effect and the Competing model's structural invariance to symmetric evidence addition. In the Competing model, choice probability is a mixture of TTB (based on the highest validity cue) and Tallying (based on the absolute difference in cue counts). If we hold Option B constant and add lower-validity cues to Option A, the Competing model predicts that Option A's choice probability must strictly increase (or asymptote) because the Tallying advantage grows while the TTB winner remains A. However, the Advocated model features divisive normalization: adding secondary cues to Option A dilutes the perceived value of its primary cue. If this self-inhibition is strong enough, adding supporting cues to A can paradoxically *decrease* its overall choice probability. Furthermore, by symmetrically adding cues to both A and B, the Competing model predicts perfectly constant choice probabilities (since the TTB winner and Tally difference are unchanged), whereas the Advocated model predicts a systematic shift toward 0.5 due to mutual dilution and conflict suppression.

[31] This design quantitatively dissociates the Probabilistic Strategy Mixture (Advocated) from the Lateral Inhibition Evidence Accumulator (Competing) by exploiting the Advocated model's structural invariance to symmetric evidence addition. In the Advocated model, choice probability is a mixture of TTB (based on the single highest-validity discriminating cue) and Tallying (based on the absolute difference in total winning cues). By systematically adding pairs of lower-validity cues to both options across trials, we hold both the TTB winner and the tally difference strictly constant. The Advocated model is mathematically forced to predict identical choice probabilities across these trial blocks. In stark contrast, the Competing model features divisive normalization (lateral inhibition). Adding secondary cues to an option dilutes the perceived value of its primary cue (self-inhibition) and suppresses the alternative option (conflict suppression). Consequently, the Competing model predicts systematic shifts in choice probability as more cues are added, breaking the invariance.

[32] This design quantitatively dissociates the Lateral Inhibition Evidence Accumulator (Advocated) from the Feature-Averaging model (Competing) by exploiting a strict structural invariance in the Competing model: cue-wise cancellation. In the Competing model, features that are active for both options (shared cues) are perfectly cancelled out before any evidence accumulation or normalization occurs. Therefore, adding shared cues to both options mathematically guarantees identical choice probabilities. In stark contrast, the Advocated model evaluates cues holistically with divisive normalization. Shared cues contribute to the total active weight for both options, increasing both self-dilution (alpha) and conflict suppression (beta) for the unique cues. By systematically taking base trials and adding shared lower-validity cues, the Competing model predicts constant choice probabilities, while the Advocated model predicts systematic shifts as the primary cues are laterally inhibited by the shared 'background' cues.

[33] This design quantitatively dissociates the Feature-Averaging model (Advocated) from the Lateral Inhibition Evidence Accumulator (Competing) by exploiting a strict structural invariance in the Advocated model: cue-wise cancellation. In the Advocated model, features that are active for both options (shared cues) are perfectly cancelled out before any evidence accumulation or soft normalization occurs. Therefore, adding shared cues to both options mathematically guarantees identical choice probabilities. In stark contrast, the Competing model evaluates cues holistically with divisive normalization and lacks cue-wise cancellation. Shared cues contribute to the total active weight for both options, increasing both self-dilution and conflict suppression for the unique cues. By systematically taking base trials and adding shared lower-validity or higher-validity cues, the Advocated model is mathematically forced to predict constant choice probabilities across the sets, while the Competing model predicts systematic shifts as the primary cues are laterally inhibited by the shared 'background' cues.

[34] This design quantitatively dissociates the Lateral Inhibition Evidence Accumulator (Advocated) from the Salience-Weighted Evidence Accumulator (Competing) by exploiting the Advocated model's 'perverse' self-dilution effect. In the Competing model, adding unique positive cues to an option strictly increases its accumulated evidence, thereby monotonically increasing its predicted choice probability. In stark contrast, the Advocated model employs divisive normalization before evidence integration. Adding lower-validity secondary cues to an option dilutes the perceived value of its primary high-validity cue (self-inhibition). If this self-dilution parameter is sufficiently strong, adding secondary cues can paradoxically decrease the option's overall accumulated evidence, leading to a non-monotonic drop in choice probability. By starting with a baseline trial that pits a single strong cue against a single medium cue, and systematically adding weak secondary cues to the strong option across subsequent trials, we force the Competing model to predict a strictly increasing preference for the strong option, while the Advocated model can predict a perverse reversal.

[35] This design quantitatively dissociates the Salience-Weighted Evidence Accumulator (Advocated) from the Lateral Inhibition Evidence Accumulator (Competing) by exploiting the strict monotonicity of the Advocated model and the 'perverse' self-dilution of the Competing model. In the Advocated model, adding unique positive cues to an option strictly increases its total accumulated evidence. Because the non-linear squashing function is monotonic, the predicted choice probability for that option must strictly increase. In stark contrast, the Competing model employs divisive normalization: adding a lower-validity unique cue to an option increases its self-inhibition, diluting the value of its primary high-validity cue. If this self-inhibition is strong, adding supporting cues can paradoxically decrease the option's choice probability. By starting with a baseline trial (Option A wins the highest cue, Option B wins the second highest) and systematically adding weaker unique cues to Option A across trials, we test for this strict monotonicity. We also include a sequence where weaker cues are added symmetrically as shared features to both options, which the Advocated model treats as a scalar-weighted evidence boost for both, while the Competing model treats it as a source of massive mutual dilution.

[36] This design quantitatively dissociates the Lateral Inhibition Evidence Accumulator (Advocated) from the Relative Evidence Accumulation with Conflict Discounting (Competing) model by exploiting the Advocated model's 'perverse' self-dilution effect. In the Competing model, adding unique positive cues to an option strictly increases its total accumulated evidence, thereby monotonically increasing its predicted choice probability. In stark contrast, the Advocated model employs divisive normalization before evidence integration. Adding lower-validity secondary cues to an option dilutes the perceived value of its primary high-validity cue (self-inhibition). If this self-dilution parameter is sufficiently strong, adding secondary cues can paradoxically decrease the option's overall accumulated evidence, leading to a non-monotonic drop in choice probability. By starting with a baseline trial that pits a single strong cue against a single medium cue, and systematically adding weak secondary cues to the strong option across subsequent trials, we force the Competing model to predict a strictly increasing preference for the strong option, while the Advocated model can predict a perverse reversal.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Relative Evidence Accumulation with Conflict Discounting (Advocated) model from the Lateral Inhibition Evidence Accumulator (Competing) model by exploiting a strict structural invariance in the Advocated model. In the Advocated model, evidence is accumulated linearly and then discounted by the number of conflicting features (features present in the competitor but absent in the focal option). Because the choice rule is a softmax over the evidence difference, adding shared features (present in both options) to a symmetric-conflict trial mathematically perfectly preserves the evidence difference, forcing the Advocated model to predict identical choice probabilities regardless of how many shared features are added. In stark contrast, the Competing model evaluates cues holistically with divisive normalization. Adding shared features increases the total active weight for both options, which increases both self-dilution (alpha) and conflict suppression (beta) for the primary unique cues. Consequently, the Competing model predicts systematic shifts in choice probability as shared 'background' cues are added, breaking the invariance.",
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
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
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
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0
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
      1,
      0,
      0,
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
      0,
      1,
      1,
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
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 11154,
  "candidates_token_count": 555,
  "total_token_count": 14341
}
```
