"""
Toy illustration of the user's JSD pipeline, on the Hilbig (2014) Exp 1
stimulus sequence.

Mirrors the user's architecture:
  * Theory.predict(parameters, stimulus, history) -> p over 2 options
    (softmax over model scores with inverse temperature beta, plus
    epsilon lapse), history ignored.
  * Parameters drawn once per simulated subject ("run"); every subject
    sees the SAME trial ordering.
  * choice_matrix -> (n_runs, n_trials) binary responses.
  * static JSD : per-trial Bernoulli over the response, pooled over runs.
  * sequence JSD: per-trial lag-1 joint over (y_{t-1}, y_t), 4 cells,
    trial 0 = marginal padded into cells (00, 01).

Stimuli & validities come from the eval_hilbig human data
(results/heuristic_decision_making/hilbig2014/exp1.txt). Human
participants each saw their own trial permutation; the toy assumes one
shared ordering, so we use the FIRST participant's 96-trial sequence.
Validities are the task vector stored in the data ([0.9, 0.8, 0.7, 0.6]).

Added for diagnosis (the points from the discussion):
  * Rao-Blackwellized estimator: average predict() PROBABILITIES over
    parameter draws instead of sampling responses — same target, no
    binomial noise.
  * Split-half null: JSD of a theory against itself, measuring the
    plug-in bias floor at this n_runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# ======================================================================
# Theories (toy versions of the user's YAML specs)
# ======================================================================
def softmax_lapse(scores, beta, epsilon):
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p = e / e.sum()
    return (1 - epsilon) * p + epsilon / 2.0


class TTB:
    name = "Take The Best"

    def __init__(self, validities):
        self.cue_order = np.argsort(-np.asarray(validities), kind="stable")

    def sample_parameters(self, rng):
        return {"beta": rng.uniform(0.1, 20.0), "epsilon": rng.uniform(0.0, 0.5)}

    def predict(self, parameters, stimulus, history=None):
        a, b = stimulus
        for j in self.cue_order:                  # cue cascade
            if a[j] != b[j]:
                w = 0 if a[j] > b[j] else 1
                scores = np.array([1.0, 0.0]) if w == 0 else np.array([0.0, 1.0])
                return softmax_lapse(scores, parameters["beta"], parameters["epsilon"])
        return np.array([0.5, 0.5])               # no discriminating cue


class Tally:
    name = "Tallying"

    def sample_parameters(self, rng):
        return {"beta": rng.uniform(0.1, 20.0), "epsilon": rng.uniform(0.0, 0.5)}

    def predict(self, parameters, stimulus, history=None):
        a, b = stimulus
        scores = np.array([float(np.sum(a > b)), float(np.sum(b > a))])
        if scores[0] == scores[1]:
            return np.array([0.5, 0.5])
        return softmax_lapse(scores, parameters["beta"], parameters["epsilon"])


# ======================================================================
# Fixed experiment: the Hilbig (2014) Exp 1 sequence of participant 0,
# same ordering for every simulated subject.
# ======================================================================
_REPO_ROOT = Path(__file__).resolve().parents[1]
HUMAN_DATA = (
    _REPO_ROOT / "results" / "heuristic_decision_making" / "hilbig2014" / "exp1.txt"
)


def _parse_array_string(s, dtype=float):
    return tuple(dtype(x) for x in s.strip().strip("[]").split())


_df = pd.read_csv(HUMAN_DATA)
validities = np.array(_parse_array_string(_df["validities"].iloc[0]))
K = len(validities)

_first = _df[_df["participant"] == _df["participant"].min()].sort_values("trial")
stimuli = np.array(
    [
        [_parse_array_string(a), _parse_array_string(b)]
        for a, b in zip(_first["stimulus_0"], _first["stimulus_1"])
    ]
)                                                # (n_trials, 2, K) fixed sequence
n_trials = len(stimuli)

# ======================================================================
# User's pipeline: simulate -> choice matrix -> pooled tables -> JSD
# ======================================================================
def choice_matrix(theory, stimuli, n_runs, rng):
    m = np.empty((n_runs, len(stimuli)), dtype=int)
    for r in range(n_runs):
        theta = theory.sample_parameters(rng)     # one draw per subject
        for t, stim in enumerate(stimuli):
            p = theory.predict(theta, stim)
            m[r, t] = rng.choice(2, p=p)          # policy: sample response
    return m


def jsd(p, q):
    """JSD (nats) between two discrete distributions on the same support."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def per_trial_bernoulli(m):
    p1 = m.mean(axis=0)
    return np.stack([1 - p1, p1], axis=1)         # (n_trials, 2)


def per_trial_lag1(m):
    n_runs, n_trials = m.shape
    out = np.zeros((n_trials, 4))
    out[0, 0] = (m[:, 0] == 0).mean()
    out[0, 1] = (m[:, 0] == 1).mean()
    for t in range(1, n_trials):
        idx = 2 * m[:, t - 1] + m[:, t]           # cells (00, 01, 10, 11)
        out[t] = np.bincount(idx, minlength=4) / n_runs
    return out


def static_jsd_from(m1, m2):
    P, Q = per_trial_bernoulli(m1), per_trial_bernoulli(m2)
    return float(np.mean([jsd(a, b) for a, b in zip(P, Q)]))


def sequence_jsd_from(m1, m2):
    P, Q = per_trial_lag1(m1), per_trial_lag1(m2)
    return float(np.mean([jsd(a, b) for a, b in zip(P, Q)]))


# ======================================================================
# Rao-Blackwellized version: average PROBABILITIES over parameter draws
# ======================================================================
def predictive_tables(theory, stimuli, n_draws, rng):
    P = np.empty((n_draws, len(stimuli), 2))
    for d in range(n_draws):
        theta = theory.sample_parameters(rng)
        for t, stim in enumerate(stimuli):
            P[d, t] = theory.predict(theta, stim)
    static = P.mean(axis=0)                                      # E_theta[p_t]
    joint = np.einsum("dti,dtj->dtij", P[:, :-1], P[:, 1:]).mean(0).reshape(-1, 4)
    first = np.array([[static[0, 0], static[0, 1], 0.0, 0.0]])
    return static, np.vstack([first, joint])


# ======================================================================
# Run everything
# ======================================================================
N_RUNS = 300
ttb, tal = TTB(validities), Tally()

m_ttb = choice_matrix(ttb, stimuli, N_RUNS, rng)
m_tal = choice_matrix(tal, stimuli, N_RUNS, rng)

s_plug = static_jsd_from(m_ttb, m_tal)
q_plug = sequence_jsd_from(m_ttb, m_tal)

# Split-half nulls: theory vs itself = pure plug-in bias floor.
m_ttb2 = choice_matrix(ttb, stimuli, N_RUNS, rng)
m_tal2 = choice_matrix(tal, stimuli, N_RUNS, rng)
s_null = 0.5 * (static_jsd_from(m_ttb, m_ttb2) + static_jsd_from(m_tal, m_tal2))
q_null = 0.5 * (sequence_jsd_from(m_ttb, m_ttb2) + sequence_jsd_from(m_tal, m_tal2))

# Rao-Blackwellized estimates (no response sampling).
P_ttb, J_ttb = predictive_tables(ttb, stimuli, 4000, rng)
P_tal, J_tal = predictive_tables(tal, stimuli, 4000, rng)
s_rb = float(np.mean([jsd(a, b) for a, b in zip(P_ttb, P_tal)]))
q_rb = float(np.mean([jsd(a, b) for a, b in zip(J_ttb, J_tal)]))

print(f"Design: Hilbig (2014) Exp 1, participant 0's sequence — "
      f"{n_trials} trials, validities {validities.tolist()}, "
      f"n_runs = {N_RUNS}, JSD in nats (max {np.log(2):.3f})\n")
print(f"{'':28s}{'static':>10s}{'lag-1 seq':>12s}")
print(f"{'plug-in (your pipeline)':28s}{s_plug:10.4f}{q_plug:12.4f}")
print(f"{'split-half null (bias)':28s}{s_null:10.4f}{q_null:12.4f}")
print(f"{'bias-corrected':28s}{s_plug - s_null:10.4f}{q_plug - q_null:12.4f}")
print(f"{'Rao-Blackwellized':28s}{s_rb:10.4f}{q_rb:12.4f}")

# ======================================================================
# Where does the signal live? Per-unique-pair static JSD (RB version),
# classified by what TTB and Tally each prescribe for the pair.
# ======================================================================
def ttb_winner(stim):
    a, b = stim
    for j in ttb.cue_order:
        if a[j] != b[j]:
            return 0 if a[j] > b[j] else 1
    return None                                   # guess


def tally_winner(stim):
    a, b = stim
    sa, sb = int(np.sum(a > b)), int(np.sum(b > a))
    if sa == sb:
        return None                               # guess
    return 0 if sa > sb else 1


def _fmt(w):
    return {0: "A", 1: "B", None: "guess"}[w]


print("\nPer-pair static JSD (Rao-Blackwellized), sorted by JSD:")
seen = {}
for t, stim in enumerate(stimuli):
    key = (tuple(stim[0].astype(int)), tuple(stim[1].astype(int)))
    if key not in seen:
        seen[key] = t
rows = []
for (a, b), t in seen.items():
    tw, yw = ttb_winner(stimuli[t]), tally_winner(stimuli[t])
    kind = "agree" if tw == yw else ("DISAGREE" if None not in (tw, yw) else "one guesses")
    rows.append((jsd(P_ttb[t], P_tal[t]), a, b, tw, yw, kind))
n_per_kind = {}
for d, a, b, tw, yw, kind in sorted(rows, reverse=True):
    n_per_kind[kind] = n_per_kind.get(kind, 0) + 1
    sa = "".join(map(str, a))
    sb = "".join(map(str, b))
    print(f"  {sa} vs {sb}   TTB->{_fmt(tw):5s} tally->{_fmt(yw):5s} "
          f"[{kind:11s}] JSD = {d:.4f}")
print(f"\nPair-type counts: {n_per_kind}")
