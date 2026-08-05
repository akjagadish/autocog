# experiment_attempt_01

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
**Description:** Sequential Cue Evaluation with Probabilistic Stopping: Decision-makers evaluate cues sequentially in descending order of validity. Upon finding a discriminating cue, they stop with a certain probability and choose based on accumulated evidence. If they continue, they integrate further cues, naturally blending non-compensatory (TTB) and compensatory (Tallying/WADD) behaviors.

**Parameters:**
- p_stop: [0.0, 1.0]
- beta: [0.1, 20.0]
- kappa: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    p_stop = float(parameters["p_stop"])
    beta = float(parameters["beta"])
    kappa = float(parameters["kappa"])
    epsilon = float(parameters["epsilon"])
    
    ev_A = 0.0
    ev_B = 0.0
    
    w_reach = 1.0
    p_A_total = 0.0
    p_B_total = 0.0
    
    for i, j in enumerate(cue_order):
        weight = val[j] ** kappa
        ev_A += a[j] * weight
        ev_B += b[j] * weight
        
        is_last = (i == len(cue_order) - 1)
        
        # Stop probabilistically only if the cue discriminates
        if a[j] != b[j]:
            p_s = p_stop
        else:
            p_s = 0.0
            
        # Must stop at the last cue
        if is_last:
            p_s = 1.0
            
        w_stop = w_reach * p_s
        
        # Choice probabilities if stopping at this step
        z = beta * np.array([ev_A, ev_B])
        e = np.exp(z - np.max(z))
        p_choice = e / np.sum(e)
        
        p_A_total += w_stop * p_choice[0]
        p_B_total += w_stop * p_choice[1]
        
        # Update probability of reaching the next step
        w_reach *= (1.0 - p_s)
        
    p_final = np.array([p_A_total, p_B_total])
    return (1.0 - epsilon) * p_final + epsilon * np.array([0.5, 0.5])
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
**Description:** Strategy Mixture: Decision-makers possess a repertoire of distinct, pure heuristic strategies—such as strict Take-The-Best (TTB) and strict unweighted Tallying. On any given trial, an individual probabilistically selects and executes one of these strategies in its entirety. This approach naturally generates the bimodal and contradictory choice patterns observed in human data, especially in conflict trials, by mixing discrete deterministic predictions rather than softening or blending a single sequential process.

**Parameters:**
- w_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Strategy 1: Strict Take-The-Best (TTB)
    p_ttb = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb = 1.0
            break
        elif b[j] > a[j]:
            p_ttb = 0.0
            break
            
    # Strategy 2: Strict Unweighted Tallying
    a_count = np.sum(a)
    b_count = np.sum(b)
    if a_count > b_count:
        p_tally = 1.0
    elif b_count > a_count:
        p_tally = 0.0
    else:
        p_tally = 0.5
        
    w_ttb = float(parameters["w_ttb"])
    
    # Probabilistic mixture of discrete deterministic predictions
    p_a = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Motor/execution noise
    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a + epsilon * 0.5
    
    return np.array([p_a, 1.0 - p_a])
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

[24] This design provides a strict, parameter-free qualitative dissociation between the Advocated model (TTB with Sub-linear Tallying) and the Competing model (Weighted Additive with Power-Function Validity Scaling). The Advocated model assigns a fixed weight (beta_ttb) to the highest-validity discriminating cue, completely ignoring its absolute validity magnitude, and then tallies the remaining cues. By holding the number of remaining cues won by Option B constant (e.g., B always wins the 5th cue) while systematically shifting the identity of Option A's best cue from the 1st to the 4th cue across trials, the Advocated model is mathematically forced to predict identical choice probabilities across these trials. In stark contrast, the Competing model scales the evidence of each cue by a power function of its actual validity. As Option A's single winning cue drops in validity across trials, the Competing model predicts a systematic decrease in the probability of choosing Option A.

[25] This design exploits the Competing model's assumption that cues evaluated after the first discriminating cue are simply tallied, ignoring their validities. By holding the highest-validity discriminating cue constant and fixing the number of remaining cues won by each option, the Competing model predicts identical choice probabilities across a set of trials. However, by systematically swapping which specific lower-validity cues are won by Option A versus Option B, the Advocated model (WADD-PWR) predicts a systematic shift in choice probabilities because it weights every cue by a power function of its actual validity.

[26] This design quantitatively dissociates the Sequential Cue Evaluation with Probabilistic Stopping model (Advocated) from the Weighted Additive with Power-Function Validity Scaling model (WADD-PWR, Competing) by exploiting a strict mathematical invariance in the latter. Because WADD-PWR integrates all cues simultaneously and maps the accumulated evidence to choice probabilities via a softmax function, its predictions are strictly linear in log-odds space. That is, if the cue difference between options in Trial 3 is the exact sum of the cue differences in Trial 1 and Trial 2, WADD-PWR mathematically guarantees that the log-odds of choosing Option A in Trial 3 will exactly equal the sum of the log-odds in Trials 1 and 2. In contrast, the Advocated model evaluates cues sequentially and stops probabilistically. This mechanism creates a mixture of choices made at different stopping points, fundamentally violating log-odds additivity. By presenting pairs of trials with additive cue differences, we can clearly isolate the probabilistic stopping mechanism of the Advocated model from the linear integration of the Competing model.

[27] This design quantitatively dissociates the Advocated model (WADD-PWR) from the Competing model (Sequential Cue Evaluation with Probabilistic Stopping) by manipulating the discrimination status of higher-validity cues. In WADD-PWR, the impact of a low-validity cue on the log-odds of choice is constant, regardless of whether higher-validity cues discriminate or tie, because all cues are integrated simultaneously and independently. In the Sequential model, higher-validity discriminating cues trigger probabilistic stopping, heavily suppressing the probability of ever evaluating lower-validity cues. By creating pairs of trials where the lowest-validity cue flips its preference, and systematically varying how many higher-validity cues tie versus discriminate across pairs, the Sequential model predicts a massive change in the lowest-validity cue's impact (high impact when top cues tie, near-zero impact when top cues discriminate). In stark contrast, WADD-PWR predicts strict invariance in the cue's impact across all contexts.

[28] The Competing model uses unweighted sequential tallying, where stopping is triggered when the absolute difference in won cues exceeds a threshold. Trials where a leading option wins consecutive cues (e.g., A wins cues 1 and 2) will rapidly trigger this threshold, causing strong preferences. When wins are alternating, the evidence difference remains near zero, preventing early stopping and pushing the decision to the final unweighted tally. The Advocated model, however, weights evidence by validity. It does not rely on an unweighted tally threshold, but probabilistically stops at any discriminating cue, making choices based on the accumulated validity-weighted evidence.

[29] This design quantitatively dissociates the Sequential Bounded Evidence Accumulation model (Advocated) from the Sequential Cue Evaluation with Probabilistic Stopping model (Competing) by exploiting how they measure evidence. The Advocated model maintains an strictly unweighted running tally of the evidence difference (E += a[j] - b[j]) and only stops if the absolute unweighted difference exceeds a threshold. In contrast, the Competing model accumulates validity-weighted evidence and stops probabilistically at any discriminating cue, regardless of the tally. By creating trials where the unweighted tally repeatedly returns to zero (e.g., alternating wins) but the validity-weighted sum strictly favors one option throughout the sequence, the Advocated model is forced to predict near-indifference (or exactly 50/50 if cues are exhausted), whereas the Competing model predicts a strong, consistent preference for the option with higher validities at every possible stopping point.

[30] This design quantitatively dissociates the Sequential Cue Evaluation with Probabilistic Stopping model (Advocated) from the Strategy Mixture model (Competing) by manipulating the sequential position of discriminating cues to alter stopping probabilities. In the Competing model, choice probabilities depend solely on the identity of the single best discriminating cue (for TTB) and the total count of winning cues (for Tallying). By holding Option A's win on the highest-validity cue constant and fixing the total number of cues won by Option B across a set of trials, the Competing model is mathematically forced to predict identical choice probabilities. However, the Advocated model evaluates cues sequentially and stops probabilistically ONLY when a cue discriminates. By shifting Option B's winning cue from the 2nd position to the 5th position across trials, we not only change the validity-weighted evidence but also fundamentally alter the probability distribution of stopping points (because non-discriminating intermediate cues force the model to continue). This produces systematically shifting choice probabilities in the Advocated model, while the Competing model remains strictly invariant.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Sequential Cue Evaluation with Probabilistic Stopping model (Advocated) from the Strategy Mixture model (Competing) by exploiting the Competing model's invariance to the absolute validity of the highest discriminating cue. In the Strategy Mixture model, choice probabilities depend exclusively on the identity of the single best discriminating cue (which determines the TTB prediction) and the total count of winning cues (which determines the Tallying prediction). By artificially tying the highest-validity cues across a sequence of trials, we can systematically shift the 'best discriminating cue' down the validity hierarchy (e.g., from Cue 1 to Cue 2 to Cue 3) while holding the Tallying score constant. The Strategy Mixture model is mathematically forced to predict identical choice probabilities across these trials because the TTB winner and Tallying winner remain unchanged. In stark contrast, the Advocated model accumulates validity-weighted evidence and stops probabilistically. Tying the top cues forces the Advocated model to evaluate deeper into the cue array, changing both the accumulated evidence margins and the stopping distribution, thus predicting a systematic shift in choice probabilities.",
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
      1
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
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      1,
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
      1
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
  "prompt_token_count": 9473,
  "candidates_token_count": 513,
  "total_token_count": 12104
}
```
