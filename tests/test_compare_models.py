"""Tests for `scripts/compare_models.py`.

Closed-form checks where possible:

1. Free-parameter counting: a context-pinned entry is fixed, a degenerate
   interval / singleton set is fixed, real intervals / sets are free.
2. Code length ignores comments and docstrings.
3. The MLE of a constant-probability model is the pooled proportion
   sum(k) / sum(n), and its AIC/BIC follow from that NLL.
4. Pairwise prediction similarity: SSE / MAE / r on hand-computed vectors.
5. Run loading: round truncation, roles and generations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.theory import Theory

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = REPO_ROOT / "theories" / "heuristic_decision_making"

CONST_PREDICT = """def predict(parameters, state, history):
    q = float(parameters["q"])
    return np.array([1.0 - q, q])
"""
POLICY = """def policy(probs):
    return int(np.argmax(probs))
"""


def _theory(params: dict, predict: str = CONST_PREDICT, desc: str = "d") -> Theory:
    return Theory(description=desc, predict_source=predict, policy_source=POLICY,
                  parameters=params)


# ---------------------------------------------------------------------------
# 1. Parameters.
# ---------------------------------------------------------------------------


def test_ttb_has_two_free_params_and_pinned_validities():
    from scripts.compare_models import parameter_space, task_context

    th = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    fixed, free = parameter_space(th, task_context())
    assert [p.name for p in free] == ["beta", "epsilon"]
    assert free[0].bounds == (0.1, 20.0)
    assert fixed["validities"] == [0.9, 0.8, 0.7, 0.6]


def test_degenerate_ranges_are_fixed_and_sets_are_discrete():
    from scripts.compare_models import parameter_metrics, parameter_space, task_context

    th = _theory({"q": "[0.0, 1.0]", "a": "[2.0, 2.0]", "k": "{1, 2, 3}",
                  "s": "{5}", "dead": "[0, 1]"})
    fixed, free = parameter_space(th, task_context())
    assert fixed["a"] == 2.0 and fixed["s"] == 5
    assert {p.name: p.kind for p in free} == {
        "q": "continuous", "k": "discrete", "dead": "continuous"}
    m = parameter_metrics(th, task_context())
    assert (m["n_free_params"], m["n_free_continuous"], m["n_free_discrete"]) == (3, 2, 1)
    # Only "q" is read by predict; "k" and "dead" are declared but unused.
    assert m["n_free_unused"] == 2


# ---------------------------------------------------------------------------
# 2. Code length.
# ---------------------------------------------------------------------------


def test_code_metrics_ignore_comments_and_docstrings():
    from scripts.compare_models import code_metrics

    plain = "def predict(parameters, state, history):\n    x = 1\n    return x\n"
    noisy = ('def predict(parameters, state, history):\n'
             '    """A long docstring\n    over lines."""\n'
             '    # a comment\n\n    x = 1  # trailing\n    return x\n')
    a, b = code_metrics(plain), code_metrics(noisy)
    for key in ("code_lines", "code_tokens", "code_ast_nodes", "code_branches"):
        assert a[key] == b[key], key
    assert a["code_lines"] == 3
    assert b["code_chars_raw"] > a["code_chars_raw"]


def test_uses_history_detects_reads_only():
    from scripts.compare_models import code_metrics

    reads = "def predict(p, s, h):\n    return len(h['label'])\n"
    ignores = "def predict(p, s, history):\n    return 1\n"
    assert code_metrics(reads)["uses_history"] is True
    assert code_metrics(ignores)["uses_history"] is False


def test_ncd_identical_below_unrelated():
    from scripts.compare_models import ncd

    x = Theory.from_yaml(YAML_DIR / "ttb.yaml").predict_source
    y = "".join(chr(33 + (i * 7919) % 90) for i in range(len(x)))
    assert ncd(x, x) < 0.1 < 0.8 < ncd(x, y)


# ---------------------------------------------------------------------------
# 3. Fitting.
# ---------------------------------------------------------------------------


def test_constant_model_mle_is_pooled_proportion():
    from scripts.compare_models import binomial_nll, fit_theory, task_context

    pairs = [((1, 0, 0, 0), (0, 1, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 1))]
    n, k = np.array([40.0, 60.0]), np.array([10.0, 20.0])
    fit = fit_theory(_theory({"q": "[0.0, 1.0]"}), pairs, n, k, task_context(),
                     n_random=50, n_starts=2, seed=1)
    q_hat = k.sum() / n.sum()                                   # 0.3
    assert fit.params["q"] == pytest.approx(q_hat, abs=1e-3)
    expected_nll = binomial_nll(np.full(2, q_hat), n, k)
    assert fit.nll == pytest.approx(expected_nll, rel=1e-6)
    assert fit.aic == pytest.approx(2 + 2 * expected_nll, rel=1e-6)
    assert fit.bic == pytest.approx(np.log(100) + 2 * expected_nll, rel=1e-6)


def test_fit_enumerates_discrete_choices():
    from scripts.compare_models import fit_theory, task_context

    predict = """def predict(parameters, state, history):
    q = 0.2 if parameters["k"] == 1 else 0.7
    return np.array([1.0 - q, q])
"""
    pairs = [((1, 0, 0, 0), (0, 1, 0, 0))]
    fit = fit_theory(_theory({"k": "{1, 2}"}, predict), pairs,
                     np.array([10.0]), np.array([7.0]), task_context())
    assert fit.params["k"] == 2


def test_broken_theory_reports_error_not_crash():
    from scripts.compare_models import fit_theory, task_context

    bad = "def predict(parameters, state, history):\n    raise RuntimeError('boom')\n"
    fit = fit_theory(_theory({"q": "[0, 1]"}, bad), [((1, 0, 0, 0), (0, 1, 0, 0))],
                     np.array([5.0]), np.array([2.0]), task_context(), n_random=5)
    assert fit.error is not None and "boom" in fit.error
    assert np.isnan(fit.nll)


# ---------------------------------------------------------------------------
# 4. Pairwise similarity.
# ---------------------------------------------------------------------------


def test_pairwise_prediction_similarity_closed_form():
    from scripts.compare_models import pairwise_prediction_similarity

    p = np.array([0.1, 0.5, 0.9])
    s = pairwise_prediction_similarity(p, p)
    assert (s["pearson_r"], s["sse"], s["mae"]) == (pytest.approx(1.0), 0.0, 0.0)
    s = pairwise_prediction_similarity(p, 1 - p)
    assert s["pearson_r"] == pytest.approx(-1.0)
    assert s["sse"] == pytest.approx(0.8 ** 2 + 0 + 0.8 ** 2)
    assert s["mae"] == pytest.approx(1.6 / 3)
    assert np.isnan(pairwise_prediction_similarity(p, np.full(3, 0.5))["pearson_r"])


# ---------------------------------------------------------------------------
# 5. Run loading.
# ---------------------------------------------------------------------------


def _write_round(run: Path, idx: int, starting: list[tuple[str, bool]], repl: str | None):
    th = {"description": "d", "predict_source": CONST_PREDICT, "policy_source": POLICY,
          "parameters": {"q": "[0, 1]"}}
    d = {"round_idx": idx,
         "starting_theories": [{"slot": i, "label": lab, "killed": k, "theory": th}
                               for i, (lab, k) in enumerate(starting)],
         "replacement": None if repl is None else {"label": repl, "theory": th}}
    rd = run / "rounds" / f"round_{idx:03d}"
    rd.mkdir(parents=True)
    (rd / "theories.json").write_text(json.dumps(d))


def test_run_theories_roles_generations_and_truncation(tmp_path):
    from scripts.compare_models import parse_run_dir, run_theories

    run = tmp_path / "dmb_ground_truth_single_cue_noise=0.0_some-llm-1.0_run2"
    _write_round(run, 0, [("pi_1", True), ("pi_2", False)], "pi_3")
    _write_round(run, 1, [("pi_3", True), ("pi_2", False)], "pi_4")

    full = {lab: (role, gen) for lab, role, gen, _ in run_theories(run)}
    assert full == {"pi_1": ("base", 0), "pi_2": ("surfaced", 0),
                    "pi_3": ("intermediate", 1), "pi_4": ("surfaced", 2)}

    one = {lab: (role, gen) for lab, role, gen, _ in run_theories(run, max_rounds=1)}
    assert one == {"pi_1": ("base", 0), "pi_2": ("surfaced", 0), "pi_3": ("surfaced", 1)}

    assert parse_run_dir(run) == {"family": "single_cue", "noise": 0.0,
                                  "llm": "some-llm-1.0", "replicate": "2"}


def test_time_budget_truncates_and_keeps_best_so_far():
    from scripts.compare_models import fit_theory, task_context

    slow = """def predict(parameters, state, history):
    import time
    time.sleep(0.02)
    q = float(parameters["q"])
    return np.array([1.0 - q, q])
"""
    pairs = [((1, 0, 0, 0), (0, 1, 0, 0))]
    fit = fit_theory(_theory({"q": "[0, 1]"}, slow), pairs, np.array([10.0]),
                     np.array([3.0]), task_context(), n_random=200, time_budget=0.3)
    assert fit.truncated
    assert fit.error is None and 0.0 <= fit.params["q"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Per-LLM summaries and figures.
# ---------------------------------------------------------------------------


def _models_frame():
    import pandas as pd

    rows = [  # llm, run, label, role, generation, fit_mse, n_free_params
        ("a", "r1", "pi_1", "surfaced", 0, 0.001, 2),   # seed: never "best"
        ("a", "r1", "pi_3", "intermediate", 1, 0.010, 3),
        ("a", "r1", "pi_4", "surfaced", 2, 0.050, 5),
        ("a", "r2", "pi_3", "surfaced", 1, 0.030, 4),
        ("a", "r2", "pi_4", "surfaced", 2, np.nan, 9),  # failed fit: ignored
        ("b-1.5", "r3", "pi_1", "base", 0, 0.200, 2),
        ("b-1.5", "r3", "pi_3", "intermediate", 1, 0.090, 6),  # no surfaced non-seed
        ("b-1.5", "r3", "pi_4", "intermediate", 2, 0.120, 8),
    ]
    df = pd.DataFrame(rows, columns=["llm", "run", "label", "role", "generation",
                                     "fit_mse", "n_free_params"])
    df["family"] = "single_cue"
    df["model"] = df["llm"] + "/" + df["run"] + "/" + df["label"]
    df["code_lines"] = 10 * df["n_free_params"]
    return df


def test_best_per_llm_prefers_surfaced_non_seed_then_falls_back():
    from scripts.compare_models import best_per_llm

    best = best_per_llm(_models_frame()).set_index("llm")
    # a: the intermediate pi_3 fits better (0.010) but surfaced wins -> r2/pi_3.
    assert best.loc["a", "model"] == "a/r2/pi_3"
    # b: no surfaced non-seed model -> best non-seed of any role.
    assert best.loc["b-1.5", "model"] == "b-1.5/r3/pi_3"


def test_llm_trajectory_is_median_across_runs_per_generation():
    from scripts.compare_models import llm_trajectory

    t = llm_trajectory(_models_frame(), "n_free_params").set_index(["llm", "generation"])
    assert t.loc[("a", 0), "median"] == 2
    assert t.loc[("a", 1), "median"] == 3.5          # median(3, 4)
    assert t.loc[("a", 2), "median"] == 7.0          # median(5, 9)
    assert t.loc[("a", 1), "n"] == 2


def test_llm_label_known_and_fallback():
    from scripts.compare_models import llm_label

    assert llm_label("claude-opus-5") == "Opus 5"
    assert llm_label("some-new-model") == "some-new-model"


def test_saved_outputs_keep_binary_stimulus_codes(tmp_path):
    import pandas as pd
    from scripts.compare_models import load_saved_outputs

    _models_frame().to_csv(tmp_path / "models.csv", index=False)
    pd.DataFrame({"model": ["a/r1/pi_1"], "option_a": ["0001"], "option_b": ["0110"],
                  "p_b_fit": [0.4], "p_b_prior": [0.5], "p_b_human": [np.nan]}
                 ).to_csv(tmp_path / "predictions.csv", index=False)
    pd.DataFrame({"family": ["single_cue"], "option_a": ["0001"], "option_b": ["0110"],
                  "p_b_target": [0.3]}).to_csv(tmp_path / "targets.csv", index=False)
    models, preds, targets = load_saved_outputs(tmp_path)
    assert preds["option_a"].iloc[0] == "0001" and targets["option_b"].iloc[0] == "0110"


def test_llm_figures_are_written(tmp_path):
    import matplotlib
    import pandas as pd
    from scripts.compare_models import plot_llm_figures

    matplotlib.use("Agg")
    models = _models_frame()
    pairs = [("0001", "0110"), ("1000", "0100"), ("0011", "1100")]
    preds = pd.DataFrame([{"model": m, "option_a": a, "option_b": b,
                           "p_b_fit": 0.1 * i, "p_b_prior": 0.5, "p_b_human": np.nan}
                          for m in models["model"] for i, (a, b) in enumerate(pairs)])
    targets = pd.DataFrame([{"family": "single_cue", "option_a": a, "option_b": b,
                             "p_b_target": 0.2 * i} for i, (a, b) in enumerate(pairs)])
    written = plot_llm_figures(models, preds, targets, tmp_path)
    names = {p.name for p in written}
    assert {"llm_a.png", "llm_b-1-5.png", "best_per_llm.png", "llm_trajectories.png",
            "best_per_llm.csv"} <= names
