# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
Lexicographic Parity (Tightened)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Lexicographic Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = len(validities)
    
    # Determine the number of top cues to consider based on k_frac
    k_frac = float(parameters["k_frac"])
    k = int(np.round(k_frac * n_features))
    if k < 1:
        k = 1
        
    # Get indices of the top k validities (stable sort to preserve original order on ties)
    top_k_idx = np.argsort(validities, kind='mergesort')[-k:]
    
    a_top = a[top_k_idx]
    b_top = b[top_k_idx]
    
    # Count strict A wins within the top K valid cues
    a_wins = int(np.sum(a_top > b_top))
    
    # Odd number of A-wins -> A (0); even -> B (1)
    winner = 0 if (a_wins % 2 == 1) else 1
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 4: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** To dissociate Cue-Parity (which computes parity over ALL features) from Lexicographic Parity (which computes parity only over the top K most valid features), we design trials where the parity of A-wins changes depending on whether the least valid features are included. By making A strictly beat B on specific lower-validity cues, the two models will predict opposite choices for most values of the k_frac parameter.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the exact agreement proportion between the subject's choices and the 5-feature Cue-Parity rule. The advocated theory (Cue-Parity) will score near 1.0 (subject to noise) because it uses all 5 features to compute parity. The competing theory (Lexicographic Parity) typically uses only the top 3 or 4 cues (due to k_frac in [0.5, 1.0]), which reverses the parity prediction on 6 out of 8 trials. Thus, the competing theory will yield an average agreement around 0.4, providing a massive, highly significant contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract options as 2D arrays
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Compute strict A-wins across all 5 cues
    a_wins = np.sum(a_mat > b_mat, axis=1)
    
    # Cue-Parity rule: Odd A-wins -> A (0), Even A-wins -> B (1)
    cp_preds = (a_wins % 2 == 0).astype(int)
    
    # Compute agreement between subjects' responses and Cue-Parity predictions
    responses = data['response'].values
    return float(np.mean(responses == cp_preds))

### RESULTS
- Predicted under pi_1 (simulated): 0.8479 (var=0.0101)
- Predicted under pi_3 (simulated): 0.4615 (var=0.0550)
- Observed on real data: 0.8358 (var=0.0105)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.98, 0.92, 0.85, 0.7, 0.6, 0.52]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 3: A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** To dissociate Lexicographic Parity from Cue-Parity, we manipulate whether the parity of A-wins changes when only the most valid cues are considered versus when all cues are considered. Lexicographic Parity restricts its parity calculation to the top K valid features (where K depends on k_frac), whereas Cue-Parity always computes parity across all features. By designing trials where option A strictly beats B on an odd number of the top features but an even number of features overall (and vice versa), the two models yield strictly opposite deterministic predictions. We vary the distribution of A-wins across the high-validity and low-validity features to ensure robust dissociation regardless of the exact k_frac value.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the parity of A-wins on the top 3 most valid cues differs strictly from the parity of A-wins across all 6 cues. On these trials, Cue-Parity will systematically predict the opposite of the top-3 parity rule, resulting in a score near 0. Lexicographic Parity, depending on the exact value of k_frac, will sometimes rely exactly on the top 3 cues, yielding a higher expected match rate. The strong divergence in predictions on this subset of trials guarantees robust discrimination between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        top3_awins = np.sum(a[:3] > b[:3])
        all_awins = np.sum(a > b)
        
        if (top3_awins % 2) != (all_awins % 2):
            pred = 0 if top3_awins % 2 == 1 else 1
            matches.append(1.0 if row['response'] == pred else 0.0)
            
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.1700 (var=0.0169)
- Predicted under pi_3 (simulated): 0.3667 (var=0.0840)
- Observed on real data: 0.1367 (var=0.0064)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_wins = data.apply(lambda row: sum(a > b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    b_wins = data.apply(lambda row: sum(b > a for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    
    mask_123 = (a_wins.isin([2, 4])) & (b_wins.isin([0, 1]))
    mask_45 = (a_wins == 1) & (b_wins.isin([2, 3]))
    
    val_123 = data.loc[mask_123, 'response'].mean() if mask_123.any() else 0.5
    val_45 = data.loc[mask_45, 'response'].mean() if mask_45.any() else 0.5
    
    return float(val_123 - val_45)
```

**Observed (real) value:** 0.7539 (var=0.0403)
**Predicted under pi_1:** 0.7083 (var=0.0555)
**Predicted under pi_3:** 0.5431 (var=0.0498)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if x > y), axis=1)
    b_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if y > x), axis=1)
    
    mask = a_wins == b_wins
    if not mask.any():
        return 0.5
        
    tied_data = data[mask]
    tied_a_wins = a_wins[mask]
    
    # Cue-Parity predicts A (0) if a_wins is odd, B (1) if a_wins is even
    cp_preds = (tied_a_wins % 2 == 0).astype(int)
    
    matches = (tied_data['response'] == cp_preds).astype(float)
    return float(matches.mean())
```

**Observed (real) value:** 0.8178 (var=0.0117)
**Predicted under pi_1:** 0.8596 (var=0.0092)
**Predicted under pi_3:** 0.8516 (var=0.0121)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Cue-Parity) consistently and highly accurately predicts the observed data, whereas Theory 2 (Lexicographic Parity) fails significantly whenever its predictions diverge from full Cue-Parity. In Experiment 1, the observed agreement with Cue-Parity is ~0.84, which perfectly matches Theory 1's prediction, while Theory 2 predicts only ~0.46. In Experiment 2, the metric isolates trials where top-3 parity diverges from overall parity; the observed score of ~0.14 strongly indicates subjects are following overall parity (Theory 1) rather than top-K parity (Theory 2). Experiments 3 and 4 further corroborate the robustness of the full Cue-Parity rule. Theory 2 is degenerate as it systematically mispredicts behavior on the very trials designed to test it.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Theory 1 (Cue-Parity) is exceptionally well-supported by the data, indicating that subjects compute parity across all available cues rather than truncating to a subset of high-validity cues. Since Lexicographic Parity (Theory 2) fails, propose a brand-new theory that acts as a strong competitor to Cue-Parity. A potential direction is to explore whether the parity computation is modulated by cue validities (e.g., a 'Weighted Parity' model where features contribute to the parity count probabilistically based on their validity) or if there is an interaction between Tallying (simple majority) and Parity on trials where the two conflict. The new theory should preserve the non-monotone, parity-like nature of the decision rule but introduce a novel psychological mechanism for how the features are integrated."
}
```

## Usage

```json
{
  "prompt_token_count": 5412,
  "candidates_token_count": 388,
  "total_token_count": 7283
}
```
