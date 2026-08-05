# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Human decision-making in multi-attribute choice is not governed by a single universal heuristic. Instead, individuals differ in their strategies or switch between them, such that the population's choices reflect a mixture of Tallying (which counts strict feature-wise wins and ignores magnitudes and validities) and Weighted Additive (WADD, which integrates both magnitudes and validities). A mixture weight parameter 'alpha' dictates the probability of using Tallying versus WADD on any given trial. Response noise enters through a softmax over the scores of the chosen heuristic, plus an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
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
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

## THEORY 2 — pi_2
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    w = np.asarray(parameters["validities"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"validities length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ w

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.6, 0.55]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  trial 2: A=[10, 5, 5, 5]  B=[1, 6, 6, 6]
  trial 3: A=[10, 6, 6, 6]  B=[1, 5, 5, 5]
  trial 4: A=[1, 5, 5, 5]  B=[10, 6, 6, 6]
  trial 5: A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  trial 6: A=[2, 2, 10, 10]  B=[10, 10, 2, 2]
  trial 7: A=[0, 8, 8, 8]  B=[9, 7, 7, 7]
  trial 8: A=[9, 7, 7, 7]  B=[0, 8, 8, 8]

**Rationale:** To quantitatively dissociate the advocated mixture model (Tallying + WADD) from the competing pure WADD model, this design systematically pits the two heuristics against each other using cardinal ratings. We include trials where Tallying and WADD strongly disagree (e.g., one option wins by a small margin on many low-validity features, while the other wins by a massive margin on a single high-validity feature). Because the competing model relies entirely on WADD, it will predict extreme choice probabilities for the option with the higher weighted sum. In contrast, the advocated model includes a mixture of Tallying, which will pull the predicted choice probabilities toward the option with more feature-wise wins (or toward 0.5 in case of a Tallying tie). By including trials where the heuristics agree, disagree, and where Tallying predicts a tie but WADD does not, we can isolate the unique variance explained by the Tallying component.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the Tallying heuristic and the WADD heuristic make opposite predictions. Specifically, it identifies trials where one option is vastly superior on the single most valid feature (which dominates WADD), while the other option has a slight edge on all three remaining features (which dominates Tallying). The metric calculates the proportion of choices that align with the Tallying prediction on these conflict trials. The pure WADD model will almost never choose the Tallying option (except due to random lapse), resulting in a value close to 0. The advocated mixture model, which includes a Tallying component, will choose the Tallying option significantly more often, leading to a much higher expected value.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_align = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Identify trials where Tallying and WADD disagree.
        # In this design, these are trials where one option wins on the first (highest validity) feature,
        # but the other option wins on the remaining three features.
        if a[0] < b[0] and a[1] > b[1]:
            # Tallying prefers A (response 0), WADD prefers B
            tally_align.append(row['response'] == 0)
        elif a[0] > b[0] and a[1] < b[1]:
            # Tallying prefers B (response 1), WADD prefers A
            tally_align.append(row['response'] == 1)
            
    if len(tally_align) == 0:
        return 0.0
    return float(np.mean(tally_align))

### RESULTS
- Predicted under pi_3 (simulated): 0.6179 (var=0.0477)
- Predicted under pi_2 (simulated): 0.2200 (var=0.1716)
- Observed on real data: 0.6800 (var=0.2176)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.6, 0.6, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[5, 6, 6, 6]  B=[7, 5, 5, 5]
  trial 2: A=[3, 7, 7, 7]  B=[7, 5, 5, 5]
  trial 3: A=[1, 8, 8, 8]  B=[7, 5, 5, 5]
  trial 4: A=[8, 8, 2, 2]  B=[2, 2, 8, 8]
  trial 5: A=[9, 9, 1, 1]  B=[1, 1, 9, 9]
  trial 6: A=[7, 4, 4, 4]  B=[5, 5, 5, 5]
  trial 7: A=[8, 3, 3, 3]  B=[4, 5, 5, 5]
  trial 8: A=[9, 2, 2, 2]  B=[3, 5, 5, 5]

**Rationale:** To quantitatively dissociate the advocated pure WADD theory from the competing Mixture (Tallying + WADD) theory, we exploit trials where the two models make qualitatively different predictions about indifference. Specifically, we construct 'WADD-tie' trials where the weighted sums of the two options are exactly identical, meaning pure WADD predicts a 50/50 guess. However, on these same trials, one option strictly wins on a majority of features, meaning Tallying strongly prefers it. The competing mixture model predicts a significant deviation from 50/50 towards the Tallying winner, whereas the advocated pure WADD model predicts strict indifference. We also include 'Tallying-tie' trials where options win on an equal number of features (Tallying predicts 50/50) but the weighted sums differ, allowing us to observe if choice probabilities are dampened (as the mixture model predicts) or extreme (as pure WADD predicts).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The advocated theory (pure WADD) uses a deterministic argmax policy over its computed probabilities. Because the parameters (validities, beta, epsilon) are fixed per subject, the choice probabilities for a given trial type are identical across all repetitions. Furthermore, the epsilon lapse rate only scales and shifts the probabilities without changing their relative ordering, meaning the argmax is unaffected. Thus, the advocated theory predicts zero intra-subject variance across repetitions of the exact same trial. In contrast, the competing theory uses a probabilistic sampling policy (`np.random.choice`), which inherently produces choice variability across repetitions. By measuring the mean intra-subject, intra-trial variance of responses, we perfectly capture this structural difference.

Source:
def metric(data: pd.DataFrame) -> float:
    data['trial_key'] = data['option_a_ratings'].apply(tuple).astype(str) + "_" + data['option_b_ratings'].apply(tuple).astype(str)
    var_df = data.groupby(['subject_id', 'trial_key'])['response'].var().fillna(0)
    return float(var_df.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1459 (var=0.0040)
- Predicted under pi_2 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 0.0000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  A=[2, 8, 8, 8]  B=[10, 5, 5, 5]
  A=[10, 1, 1, 1]  B=[0, 2, 2, 2]
  A=[0, 5, 5, 5]  B=[10, 4, 4, 4]
  A=[9, 3, 3, 3]  B=[0, 4, 4, 4]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = (b_wins > a_wins).astype(int)
    return float((data['response'] == tally_choice).mean())

```

**Observed (real) value:** 0.6400 (var=0.2304)
**Predicted under pi_3:** 0.5660 (var=0.0427)
**Predicted under pi_2:** 0.1950 (var=0.0838)

### Experiment 4
**Design**
  A=[10, 0, 0, 0]  B=[0, 3, 3, 3]
  A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 2, 2, 2]  B=[10, 0, 0, 0]
  A=[5, 5, 5, 5]  B=[6, 4, 4, 4]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (10, 0, 0, 0) and b == (0, 3, 3, 3):
            if resp == 0: wadd_match += 1
            count += 1
        elif a == (2, 2, 2, 0) and b == (0, 0, 0, 10):
            if resp == 1: wadd_match += 1
            count += 1
        elif a == (0, 2, 2, 2) and b == (10, 0, 0, 0):
            if resp == 1: wadd_match += 1
            count += 1
            
    return float(wadd_match / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.3667 (var=0.0856)
**Predicted under pi_3:** 0.3702 (var=0.0519)
**Predicted under pi_2:** 0.6800 (var=0.0576)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (pi_3) consistently outperforms Theory 2 (pi_2) in predicting the correct choice proportions across Experiments 1, 3, and 4, demonstrating that human decision-making in this task is indeed better captured by a mixture of Tallying and Weighted Additive (WADD) heuristics rather than pure WADD. However, Experiment 2 reveals a crucial structural flaw in Theory 1's model implementation: the observed intra-subject, intra-trial variance is exactly 0.0, meaning human subjects are entirely deterministic when facing the same trial multiple times. Theory 1 currently uses a probabilistic sampling policy (`np.random.choice`), which predicts non-zero intra-subject variance. By switching Theory 1's policy to a deterministic `argmax`, the model can maintain its accurate population-level choice predictions (driven by between-subject parameter differences in the mixture weight and validities) while perfectly capturing the zero intra-subject variance observed in Experiment 2.",
  "verdict": "new_model",
  "target_theory_idx": 1,
  "recommendation": "Preserve the theory description (mixture of Tallying and WADD). Modify the `policy` function to be strictly deterministic by returning `int(np.argmax(probs))` instead of sampling via `np.random.choice`. In the `predict` function, you can remove the epsilon lapse rate (or keep it, but note it won't affect the argmax unless it's applied differently) since the deterministic policy will ignore uniform noise. The population-level choice probabilities will still naturally emerge from the between-subject distributions of the parameters (alpha, beta, validities)."
}
```

## Usage

```json
{
  "prompt_token_count": 5199,
  "candidates_token_count": 361,
  "total_token_count": 6414
}
```
