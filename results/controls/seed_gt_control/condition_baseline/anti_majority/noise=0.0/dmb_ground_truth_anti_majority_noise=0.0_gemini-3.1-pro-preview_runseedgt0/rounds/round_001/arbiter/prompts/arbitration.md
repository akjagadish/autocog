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
People are modeled as running three classic heuristics in parallel — Take The Best (one-reason, top-validity cue), Tallying (count of feature-wise wins), and Weighted Additive (validity-weighted sum) — reading off the single option each heuristic deterministically prefers, and then leaning toward whichever option the majority of the three did NOT prefer. With three binary voters at least two always agree, so a majority option always exists; the model targets its complement. Response noise enters only at the final, ensemble level: a softmax over the binary "flipped-majority" score with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options. The individual heuristics contribute only their hard preference (their argmax) to the vote — they carry no internal noise — so beta and epsilon describe noise in the contrarian response itself, not in the constituent heuristics.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Anti-majority-vote ensemble. Each of Tallying / TTB / WADD
    # contributes a single hard preference (the argmax of its
    # deterministic score — no per-model softmax or lapse). We take the
    # majority vote across the three binary votes and flip it: when at
    # least two models agree on an option, we target the other one.
    # Choice noise (softmax over beta, lapse over epsilon) is applied
    # ONCE here, to the flipped-majority decision.

    def _pref_tallying(stimulus):
        # Tallying (Dawes 1979; Gigerenzer & Goldstein 1999): count
        # strict feature-wise wins; prefer the option with more wins.
        # Ties (equal win counts) fall to option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        a, b = stim[0], stim[1]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        return int(np.argmax(np.array([a_wins, b_wins])))

    def _pref_ttb(parameters, stimulus):
        # Take The Best (Gigerenzer & Goldstein 1996): consult cues in
        # descending validity; the first discriminating cue decides.
        # If no cue discriminates, the score is uniform and argmax
        # falls to option A.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        n_features = stim.shape[1]
        val = np.asarray(parameters["validities"], dtype=float)
        if val.shape[0] != n_features:
            raise ValueError(
                f"validities length {val.shape[0]} != n_features {n_features}."
            )
        # Descending validity; argsort is stable so validity ties break
        # toward the earlier feature index.
        cue_order = np.argsort(-val, kind="stable").tolist()
        a, b = stim[0], stim[1]
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            scores = np.ones(2) / 2.0
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        return int(np.argmax(scores))

    def _pref_wadd(parameters, stimulus):
        # Weighted Additive (Payne, Bettman & Johnson 1993): prefer the
        # option with the higher validity-weighted sum. Ties fall to
        # option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        n_features = stim.shape[1]
        validities = np.asarray(parameters["validities"], dtype=float)
        w = np.asarray(parameters["weights"], dtype=float)
        if w.shape[0] != n_features:
            raise ValueError(
                f"weights length {w.shape[0]} != n_features {n_features}."
            )
        scores = stim @ (validities * w)
        return int(np.argmax(scores))

    # Each model's hard preference (no noise inside the models).
    c_tallying = _pref_tallying(stimulus)
    c_ttb = _pref_ttb(parameters, stimulus)
    c_wadd = _pref_wadd(parameters, stimulus)

    # Majority vote across the three binary votes, then flip it.
    votes = [c_tallying, c_ttb, c_wadd]
    majority = 0 if votes.count(0) >= 2 else 1
    flipped = 1 - majority

    # Binary score for the flipped-majority choice (winner = 1, loser = 0),
    # then ONE softmax + lapse over the two options.
    scores = np.array([1.0, 0.0]) if flipped == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # flipped score this collapses to sigmoid(beta) for the targeted
    # option (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic flip).
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
Anti-Tallying: Subjects systematically misinterpret the expert ratings, treating a rating of 0 as superior to 1. They tally the number of features where an option 'wins' under this reversed interpretation (i.e., has a 0 while the other has a 1) and prefer the option with more reversed wins. Choice noise is modeled via a softmax over these anti-tallies and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Anti-Tallying expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    # Anti-Tallying: subjects treat 0 as better than 1.
    # Thus, option A wins a feature if a < b (i.e., a=0, b=1).
    a_anti_wins = float(np.sum(a < b))
    b_anti_wins = float(np.sum(b < a))
    
    scores = np.array([a_anti_wins, b_anti_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the anti-tallies with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=7):** [1.0, 0.9, 0.6, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1]
  trial 5: A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]

**Rationale:** To maximally dissociate the Advocated Theory (Anti-majority ensemble) from the Competing Theory (Anti-Tallying), we exploit the fact that Anti-Tallying is effectively the exact opposite of Tallying. We design trials where Tallying prefers Option A (because it has more winning features), but both Take The Best (TTB) and Weighted Additive (WADD) prefer Option B (because Option B wins on fewer but much higher-validity features). In this scenario, the majority of the three heuristics (TTB and WADD) prefer Option B. The Advocated Theory flips this majority and thus strongly predicts Option A. The Competing Theory, interpreting 0 as better than 1, counts the 'anti-wins' and strongly prefers Option B, as Option B has more 0s where Option A has 1s. This creates a stark qualitative and quantitative dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Across all trials in the design, one option always has more 1s (4 ones) while the other has fewer 1s (3 ones). The Advocated Theory (Anti-majority ensemble) consistently predicts the option with more 1s, because that option loses under TTB and WADD (which prioritize the high-validity features where it has 0s) but wins under Tallying, causing a majority vote against it that the model then flips. Conversely, the Competing Theory (Anti-Tallying) explicitly treats 0 as better than 1 and tallies these 'anti-wins', so it systematically prefers the option with more 0s (and thus fewer 1s). This metric simply calculates the proportion of trials where the subject chose the option with more 1s, expecting ~1.0 for the Advocated Theory and ~0.0 for the Competing Theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ones = data['option_a_ratings'].apply(sum)
    b_ones = data['option_b_ratings'].apply(sum)
    
    chosen_more_ones = np.where(
        data['response'] == 0,
        a_ones > b_ones,
        b_ones > a_ones
    )
    
    return float(np.mean(chosen_more_ones))

### RESULTS
- Predicted under pi_1 (simulated): 0.6208 (var=0.0656)
- Predicted under pi_3 (simulated): 0.1338 (var=0.0093)
- Observed on real data: 0.5967 (var=0.0436)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.55, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To maximally dissociate the Advocated Theory (Anti-Tallying) from the Competing Theory (Anti-Majority), we construct trials where Tallying prefers one option, while both Take The Best (TTB) and Weighted Additive (WADD) prefer the other. In such a scenario, the majority of the three standard heuristics (TTB and WADD) agree on an option, causing the Anti-Majority model to strongly predict the *opposite* option (which aligns with Tallying's choice). Conversely, the Anti-Tallying model, which treats 0s as wins and thus acts as the exact opposite of Tallying, will prefer the option that Tallying dislikes. This results in the two models predicting completely opposite choices.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits Anti-Tallying against Anti-Majority by constructing trials where the majority of standard heuristics (TTB and WADD) agree on one option, while Tallying prefers the other. Anti-Majority flips the majority choice, effectively aligning with Tallying. Anti-Tallying, by treating 0s as wins, acts as the exact opposite of Tallying. Therefore, the two models predict completely opposite choices on every single trial. This metric calculates the proportion of trials where the subject's choice matches the Anti-Tallying prediction, yielding a value near 1 for Anti-Tallying and near 0 for Anti-Majority.

Source:
def metric(data: pd.DataFrame) -> float:
    # Anti-Tallying prefers the option with more 0s.
    # In this design, option A has either three 1s and two 0s (A[0]=0) 
    # or three 0s and two 1s (A[0]=1).
    # If A[0] == 0, A has two 0s and B has three 0s, so Anti-Tallying prefers B (1).
    # If A[0] == 1, A has three 0s and B has two 0s, so Anti-Tallying prefers A (0).
    # Thus, Anti-Tallying always predicts (1 - A[0]).
    # Conversely, Anti-Majority always predicts A[0].
    # We measure the proportion of choices matching Anti-Tallying.
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    anti_tallying_pred = 1 - a_first
    return float((data['response'] == anti_tallying_pred).mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.4473 (var=0.1230)
- Predicted under pi_3 (simulated): 0.8573 (var=0.0090)
- Observed on real data: 0.3583 (var=0.1157)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins and row['response'] == 0:
            matches += 1
        elif b_wins > a_wins and row['response'] == 1:
            matches += 1
        elif a_wins == b_wins:
            matches += 0.5
    return float(matches / len(data))
```

**Observed (real) value:** 0.2675 (var=0.0463)
**Predicted under pi_1:** 0.2288 (var=0.0317)
**Predicted under pi_3:** 0.1494 (var=0.0090)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_sums = data['option_a_ratings'].apply(np.sum)
    b_sums = data['option_b_ratings'].apply(np.sum)
    
    mask = a_sums != b_sums
    if mask.sum() == 0:
        return 0.5
        
    responses = data.loc[mask, 'response']
    a_sums_filtered = a_sums[mask]
    b_sums_filtered = b_sums[mask]
    
    tallying_choices = (b_sums_filtered > a_sums_filtered).astype(int)
    
    return float((responses == tallying_choices).mean())
```

**Observed (real) value:** 0.2350 (var=0.0495)
**Predicted under pi_1:** 0.2308 (var=0.0618)
**Predicted under pi_3:** 0.1108 (var=0.0075)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Anti-majority ensemble) consistently outperforms Theory 2 (Anti-Tallying). In Experiment 1, the observed metric is 0.5967, which Theory 1 predicts closely (0.6208) while Theory 2 fails completely (0.1338). In Experiment 2, the observed metric is 0.3583; Theory 1 predicts 0.4473, whereas Theory 2 predicts 0.8573. Experiments 3 and 4 further confirm that Theory 1 captures the empirical choice patterns much better than Theory 2. Theory 2's assumption that subjects systematically treat 0s as superior to 1s (Anti-Tallying) is empirically unsupported and degenerate. However, Theory 1's 'Anti-majority' mechanism is highly idiosyncratic and unintuitive, despite fitting the data well. It is highly likely that a more standard, well-established decision heuristic can explain these results without resorting to an 'anti-majority' framing. We should replace Theory 2 with a strong, classic baseline to see if a simpler mechanism can match or exceed Theory 1's performance.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory based on a standard Take-The-Best (TTB) or a Validity-Weighted Additive (WADD) model. The data shows that subjects often prefer options with fewer overall positive features if those features have higher validities (which drives the low Tallying match rates in Exp 3 and 4). A standard TTB or WADD model with a softmax response and a lapse rate is a theoretically grounded, biologically plausible mechanism that will likely serve as a much stronger and more realistic competitor to Theory 1 than the current Anti-Tallying model."
}
```

## Usage

```json
{
  "prompt_token_count": 5676,
  "candidates_token_count": 414,
  "total_token_count": 6861
}
```
