# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
First Impression Anchoring: Subjects find the task repetitive and cognitively demanding. To minimize effort, they genuinely evaluate the options on the very first trial using a simple tallying heuristic (choosing the option with the most positive features, or guessing if tied). They then use this initial choice as an anchor, deterministically repeating the same position choice (e.g., always choosing A or always B) for the remainder of the experiment. This results in a completely zero within-subject variance while maintaining an even 50/50 split across the population.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # Dummy parameter to satisfy the contract
    _ = float(parameters["dummy"])
    
    stim = np.asarray(state, dtype=float)
    n_opts = stim.shape[0]
    
    # On the first trial, evaluate based on simple feature tallying
    if len(history.get("response", [])) == 0:
        sums = np.sum(stim, axis=1)
        max_sum = np.max(sums)
        is_max = (sums == max_sum).astype(float)
        return is_max / np.sum(is_max)
    else:
        # On all subsequent trials, repeat the first trial's choice
        first_resp = int(history["response"][0])
        probs = np.zeros(n_opts)
        if first_resp < n_opts:
            probs[first_resp] = 1.0
        else:
            probs[0] = 1.0
        return probs

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Task Disengagement / Position Bias Theory: Subjects find the fictitious product choice task unengaging or too cognitively demanding to evaluate systematically. Instead of using the expert ratings, they adopt a completely deterministic, zero-effort heuristic of exclusively selecting the same option position (e.g., always choosing Option A or always Option B) on every single trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    pref = int(parameters["preferred_option"])
    n_opts = np.asarray(state).shape[0]
    
    probs = np.zeros(n_opts)
    if pref < n_opts:
        probs[pref] = 1.0
    else:
        probs[0] = 1.0
        
    return probs

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Rationale:** To dissociate First Impression Anchoring from Task Disengagement (Position Bias), we design trials with strong, unambiguous variance in the Tallying heuristic's favored option. Under the Advocated Theory, subjects evaluate the very first trial using Tallying and then deterministically anchor on that position for the rest of the experiment. By ensuring that Tallying strongly favors Option A on half the trials and Option B on the other half, we create a scenario where the population's anchored position depends entirely on the randomized presentation order (specifically, the first trial), rather than an intrinsic, stimulus-independent position bias.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
The Advocated theory posits that subjects use a Tallying heuristic on their very first trial to determine their anchor, meaning their first choice will always match the Tallying prediction. The Competing theory posits a fixed position bias independent of the stimuli. Because the experimental design ensures that Tallying favors Option A on half the trials and Option B on the other half, a subject with a fixed position bias will only match the Tallying prediction on their first trial 50% of the time. This metric calculates the proportion of subjects whose first trial response matches the Tallying prediction, yielding 1.0 for the Advocated theory and ~0.5 for the Competing theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    first_trials = data.groupby('subject_id').first()
    
    matches = []
    for _, row in first_trials.iterrows():
        a_sum = sum(row['option_a_ratings'])
        b_sum = sum(row['option_b_ratings'])
        
        pred = 0 if a_sum > b_sum else 1
        matches.append(1.0 if row['response'] == pred else 0.0)
        
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_6 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_5 (simulated): 0.5800 (var=0.2436)
- Observed on real data: 0.4400 (var=0.2464)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]

**Rationale:** To conclusively dissociate the Advocated Theory (Task Disengagement / Position Bias) from the Competing Theory (First Impression Anchoring), this design ensures that the simple Tallying heuristic strictly favors Option B on every single trial. Under the Competing Theory, subjects use Tallying on the very first trial and then deterministically anchor on that choice for the rest of the experiment. Because Tallying always favors Option B in this design, the Competing Theory predicts that 100% of subjects will anchor on Option B, completely failing to account for any subjects who exhibit a position bias for Option A. The Advocated Theory, possessing a free parameter for the preferred position, can perfectly fit subjects who always choose Option A.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Competing Theory (First Impression Anchoring), subjects use a simple tallying heuristic on the very first trial. Because the experimental design ensures that Option B strictly has more positive features than Option A on every single trial, the tallying heuristic will deterministically favor Option B on the first trial. The subject will then anchor on Option B for all subsequent trials, resulting in a 0% rate of choosing Option A. By contrast, the Advocated Theory (Task Disengagement) posits that subjects deterministically choose a preferred position (A or B) independent of the features, predicting that approximately 50% of subjects will exclusively choose Option A. Therefore, the proportion of trials where Option A is chosen will be exactly 0.0 for all subjects under the Competing Theory (zero variance), while under the Advocated Theory it will be 1.0 for half the subjects and 0.0 for the other half, creating a massive and highly significant difference in means.

Source:
def metric(data: pd.DataFrame) -> float:
    return float((data['response'] == 0).mean())

### RESULTS
- Predicted under pi_6 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_5 (simulated): 0.5200 (var=0.2496)
- Observed on real data: 0.4800 (var=0.2496)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # TTB prediction (cues ordered 0 to 4 based on validities)
        ttb_winner = None
        for j in range(5):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = sum(1 for i in range(5) if a[i] > b[i])
        b_wins = sum(1 for i in range(5) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Only evaluate on trials where the two heuristics strictly oppose
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_match += 1
            total += 1
            
    return float(ttb_match / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    b_wins = (b_ratings > a_ratings).sum(axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    matches = (data['response'] == tally_pred)
    return float(matches.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_has_f1 = data['option_b_ratings'].apply(lambda x: x[0] == 1).astype(int)
    chose_f1 = (data['response'] == b_has_f1)
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    chose_f1 = ((a_f1 == 1) & (data['response'] == 0)) | ((b_f1 == 1) & (data['response'] == 1))
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(tuple)
    trial_means = df.groupby(['subject_id', 'trial_type'])['response'].mean()
    subj_devs = (trial_means - 0.5).abs().groupby('subject_id').mean()
    return float(subj_devs.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    dom_a = (a_sum > 0) & (b_sum == 0)
    dom_b = (a_sum == 0) & (b_sum > 0)
    subset = data[dom_a | dom_b]
    if len(subset) == 0:
        return 0.5
    correct = ((dom_a) & (data['response'] == 0)) | ((dom_b) & (data['response'] == 1))
    return float(correct[dom_a | dom_b].mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 9
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data.groupby('subject_id')['response'].var(ddof=0).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_6:** 0.0000 (var=0.0000)
**Predicted under pi_5:** 0.0000 (var=0.0000)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data.groupby('subject_id')['response'].var(ddof=0).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_6:** 0.0000 (var=0.0000)
**Predicted under pi_5:** 0.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the data show that subjects exhibit exactly zero within-subject variance (Experiments 9 and 10), indicating that they choose deterministically after or starting from the first trial. Theory 1 (pi_6) posits that this initial choice is determined by a Tallying heuristic. However, Experiments 1 and 2 clearly falsify this: in Experiment 1, only 44% of first-trial choices matched the Tallying prediction (compared to the 100% predicted by Theory 1), and in Experiment 2, where Tallying always favored Option B, subjects still chose Option A 48% of the time (compared to the 0% predicted by Theory 1). Theory 2 (pi_5), which posits a stimulus-independent Position Bias, perfectly captures the ~50% splits and the 0.0 within-subject variance across all experiments. Theory 1 is completely degenerate and relies on a falsified mechanism for the initial anchor.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Theory 1 is falsified because the first trial choice does not align with the Tallying heuristic. Propose a brand-new theory that serves as a stronger, more plausible competitor to the Position Bias theory (Theory 2). For instance, consider a 'Take-The-Best (TTB) Anchoring' theory where subjects evaluate the very first trial using the single most valid cue (rather than tallying all cues) and then anchor on that choice for the remainder of the experiment. Alternatively, propose a theory where subjects use a 'Satisficing' rule on the first trial (e.g., choosing the first option that meets a certain threshold on the top cues) before disengaging."
}
```

## Usage

```json
{
  "prompt_token_count": 7229,
  "candidates_token_count": 388,
  "total_token_count": 8684
}
```
