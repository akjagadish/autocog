# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3_1
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
    
    return p_mixed


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    return int(np.argmax(probs))


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


## EXPERIMENT 1 (proposed by pi_3_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[8, 0, 0, 0, 0]  B=[0, 4, 4, 2, 0]
  trial 2: A=[10, 10, 0, 0, 0]  B=[0, 0, 10, 10, 8]
  trial 3: A=[0, 10, 10, 0, 0]  B=[10, 0, 0, 5, 4]
  trial 4: A=[10, 0, 0, 10, 10]  B=[0, 10, 10, 0, 0]
  trial 5: A=[5, 5, 5, 5, 5]  B=[10, 10, 0, 0, 0]
  trial 6: A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  trial 7: A=[9, 9, 0, 0, 0]  B=[0, 0, 10, 10, 10]
  trial 8: A=[0, 8, 8, 8, 0]  B=[10, 0, 0, 0, 9]

**Rationale:** To quantitatively dissociate the advocated Mixture theory (Tallying + WADD) from the competing pure WADD theory, this design systematically varies the agreement between Tallying and WADD while carefully controlling the WADD score differences. The critical trials include 'WADD-ties' where the weighted sums are identical (pure WADD predicts 50/50 guessing) but Tallying strongly favors one option (Mixture predicts a preference for the Tallying winner). We also include trials where WADD and Tallying strongly conflict, and trials where Tallying predicts a tie but WADD does not. This allows us to estimate the mixture parameter 'alpha' and identify the presence of Tallying, which pure WADD cannot account for.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Contrasts Trial 5 and Trial 3. Under true validities, pure WADD has a stronger preference for A on Trial 3 (15 vs 14, diff=1.0) than on Trial 5 (17.5 vs 17, diff=0.5). Thus, pure WADD naturally predicts P(A|T3) > P(A|T5). However, the Tallying heuristic strongly favors B on Trial 3 (3 wins to 2) and favors A on Trial 5 (3 wins to 2). The advocated Mixture model, which incorporates Tallying, will therefore show a suppressed probability of choosing A on Trial 3 and an enhanced probability of choosing A on Trial 5, reversing the inequality to P(A|T5) > P(A|T3). This structural reversal is difficult for a pure WADD model to fit simultaneously with the rest of the trials.

Source:
def metric(data: pd.DataFrame) -> float:
    # Trial 3: A=[0, 10, 10, 0, 0], B=[10, 0, 0, 5, 4]
    t3_mask = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 10 and x[2] == 10)
    # Trial 5: A=[5, 5, 5, 5, 5], B=[10, 10, 0, 0, 0]
    t5_mask = data['option_a_ratings'].apply(lambda x: x[0] == 5 and x[1] == 5 and x[2] == 5)
    
    if t3_mask.sum() == 0 or t5_mask.sum() == 0:
        return 0.0
        
    # Probability of choosing A (response == 0)
    p_a_t3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    p_a_t5 = 1.0 - data.loc[t5_mask, 'response'].mean()
    
    return float(p_a_t5 - p_a_t3)

### RESULTS
- Predicted under pi_3_1 (simulated): 0.7000 (var=0.2900)
- Predicted under pi_2 (simulated): 0.2600 (var=0.4724)
- Observed on real data: 0.1400 (var=0.4804)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [1.0, 0.8, 0.6, 0.5, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 10, 10, 10]  B=[10, 5, 0, 0, 4]
  trial 2: A=[10, 5, 0, 0, 4]  B=[0, 0, 10, 10, 10]
  trial 3: A=[10, 10, 0, 0, 0]  B=[0, 0, 10, 10, 0]
  trial 4: A=[0, 0, 10, 10, 0]  B=[10, 10, 0, 0, 0]
  trial 5: A=[0, 0, 8, 8, 8]  B=[10, 10, 0, 0, 0]
  trial 6: A=[10, 10, 0, 0, 0]  B=[0, 0, 8, 8, 8]
  trial 7: A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 10, 10]
  trial 8: A=[0, 0, 0, 10, 10]  B=[10, 10, 10, 0, 0]

**Rationale:** To quantitatively dissociate the advocated pure WADD theory from the competing Mixture (Tallying + WADD) theory, we use a 5-feature design with cardinal ratings to precisely control WADD scores and Tallying wins. We construct exact 'WADD-tie' trials where the weighted sums of the two options are identical (e.g., 16 vs 16), meaning pure WADD predicts a 50/50 guess, but Tallying strongly favors one option (3 wins vs 2 wins). The mixture model predicts a significant deviation from 50/50 towards the Tallying winner, whereas pure WADD predicts strict indifference. We also include 'Tallying-tie' trials where options win on an equal number of features but WADD strongly prefers one, and conflict trials where WADD and Tallying favor opposite options. This allows us to test if the choice probabilities are strictly proportional to WADD score differences or if they are systematically pulled by Tallying wins.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the overall rate at which a subject's choices align with the Tallying heuristic, restricted to trials where Tallying has a strict preference (i.e., one option wins on more features than the other). The Pure WADD model will only align with Tallying when the subjective weighted sum happens to agree, which is frequent but not guaranteed. The Mixture model, however, explicitly incorporates Tallying as a standalone strategy on a subset of trials (governed by the alpha parameter). Therefore, the Mixture model predicts a systematically higher and more consistent alignment with the Tallying heuristic across these trials compared to Pure WADD.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    subject_means = []
    for subj, subj_df in data.groupby('subject_id'):
        tally_aligned = []
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            tally_a = np.sum(a > b)
            tally_b = np.sum(b > a)
            
            # Only consider trials where Tallying has a strict preference
            if tally_a != tally_b:
                tally_choice = 0 if tally_a > tally_b else 1
                tally_aligned.append(1 if row['response'] == tally_choice else 0)
        
        if tally_aligned:
            subject_means.append(np.mean(tally_aligned))
            
    if not subject_means:
        return 0.5
        
    return float(np.mean(subject_means))

### RESULTS
- Predicted under pi_3_1 (simulated): 0.8867 (var=0.0383)
- Predicted under pi_2 (simulated): 0.7067 (var=0.0784)
- Observed on real data: 0.5867 (var=0.0603)

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
**Predicted under pi_3_1:** 0.5650 (var=0.2258)
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
**Predicted under pi_3_1:** 0.3733 (var=0.1495)
**Predicted under pi_2:** 0.6800 (var=0.0576)

### Experiment 5
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[10, 5, 5, 5]  B=[1, 6, 6, 6]
  A=[10, 6, 6, 6]  B=[1, 5, 5, 5]
  A=[1, 5, 5, 5]  B=[10, 6, 6, 6]
  A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  A=[2, 2, 10, 10]  B=[10, 10, 2, 2]
  A=[0, 8, 8, 8]  B=[9, 7, 7, 7]
  A=[9, 7, 7, 7]  B=[0, 8, 8, 8]

**Metric**
```python
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
```

**Observed (real) value:** 0.6800 (var=0.2176)
**Predicted under pi_3_1:** 0.5600 (var=0.2464)
**Predicted under pi_2:** 0.2200 (var=0.1716)

### Experiment 6
**Design**
  A=[5, 6, 6, 6]  B=[7, 5, 5, 5]
  A=[3, 7, 7, 7]  B=[7, 5, 5, 5]
  A=[1, 8, 8, 8]  B=[7, 5, 5, 5]
  A=[8, 8, 2, 2]  B=[2, 2, 8, 8]
  A=[9, 9, 1, 1]  B=[1, 1, 9, 9]
  A=[7, 4, 4, 4]  B=[5, 5, 5, 5]
  A=[8, 3, 3, 3]  B=[4, 5, 5, 5]
  A=[9, 2, 2, 2]  B=[3, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['trial_key'] = data['option_a_ratings'].apply(tuple).astype(str) + "_" + data['option_b_ratings'].apply(tuple).astype(str)
    var_df = data.groupby(['subject_id', 'trial_key'])['response'].var().fillna(0)
    return float(var_df.mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_3_1:** 0.0000 (var=0.0000)
**Predicted under pi_2:** 0.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments (e.g., Exp 3, 4, and 5), Theory 1 (the Mixture of Tallying and WADD) clearly outperforms Theory 2 (Pure WADD). Pure WADD fails to capture the significant proportion of choices that align with the Tallying heuristic when WADD and Tallying conflict. However, Theory 1 overshoots the observed metrics in Exp 1 and Exp 2, suggesting its exact mixture formulation might still need tuning. Regardless, Theory 2's pure WADD assumption is demonstrably degenerate given the persistent evidence of non-compensatory or tallying-like behavior. Replacing Theory 2 with a stronger competitor is necessary.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory that goes beyond pure WADD. Instead of pure WADD or an unweighted Tallying mixture, consider a 'Weighted Tallying' theory (where feature differences are binarized into strict wins/losses but are then weighted by their validities) or a mixture involving 'Take-The-Best' (lexicographic choice based on the highest-validity discriminating cue). This will provide a much more competitive alternative to Theory 1's Tallying+WADD mixture."
}
```

## Usage

```json
{
  "prompt_token_count": 6527,
  "candidates_token_count": 291,
  "total_token_count": 7911
}
```
