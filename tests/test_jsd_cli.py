import json
import subprocess
import sys


def test_cli_outputs_both_marginals(tmp_path):
    spec = tmp_path / "exp.json"
    spec.write_text(json.dumps({
        "validities": [0.95, 0.7, 0.65, 0.6],
        "trial_a_ratings": [[1, 0, 0, 0]] * 4,
        "trial_b_ratings": [[0, 1, 1, 1]] * 4,
    }))
    out = subprocess.run(
        [sys.executable, "scripts/jsd_discriminability.py",
         "--theory_1", "theories/heuristic_decision_making/ttb_sampling.yaml",
         "--theory_2", "theories/heuristic_decision_making/wadd_sampling.yaml",
         "--experiment", str(spec), "--n_runs", "100"],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(out.stdout)
    assert set(result) >= {"static_jsd", "sequence_jsd", "per_trial_static_jsd"}
    assert 0.0 <= result["static_jsd"] <= 0.6932
