import json

from scripts.summarize_compute import summarize_run


def _write_prompt_log(d, name, usage):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(
        f"# {name}\n\n## System Prompt\n\nx\n\n## User Prompt\n\ny\n\n"
        f"## Response\n\n```json\nz\n```\n\n## Usage\n\n"
        f"```json\n{json.dumps(usage)}\n```\n"
    )


def test_summarize_run_counts_calls_and_tokens(tmp_path):
    _write_prompt_log(tmp_path / "round_000" / "pi_1" / "prompts",
                      "experiment_attempt_00",
                      {"input_tokens": 100, "output_tokens": 50})
    _write_prompt_log(tmp_path / "round_000" / "arbiter" / "prompts",
                      "arbitration",
                      {"input_tokens": 200, "output_tokens": 80})
    s = summarize_run(tmp_path)
    assert s["n_llm_calls"] == 2
    assert s["input_tokens"] == 300
    assert s["output_tokens"] == 130
