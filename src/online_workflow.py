from __future__ import annotations

import json
from typing import Any, Callable

_experiment_runner = None
_experiment_runner_prolific = None


def _read_prolific_settings(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_firestore_safe(value):
    """
    Convert nested Python structures into Firestore-safe JSON-like values.
    """
    if isinstance(value, dict):
        return {str(k): _to_firestore_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_firestore_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Fallback for uncommon objects (e.g., numpy scalars)
    return str(value)


def _build_payload(experiments, conditions, condition_doc_fn, *, share_template=True):
    """
    Build Firebase payload.

    With ``share_template=True`` (the default), compile the first experiment
    once via Sweetbean / Transcrypt and reuse that template for every
    condition, injecting only the per-condition block timelines. This is
    fast but **only correct when every experiment has identical Sweetbean
    structure** (same stimuli, same `FunctionVariable` args). Anything that
    is not a `TimelineVariable` — including values baked into `FunctionVariable`
    args — is frozen to the **first** experiment's value.

    With ``share_template=False``, every experiment is compiled
    independently. Use this when you mix experiments whose stimulus
    structure / `FunctionVariable` args differ across conditions
    (e.g. autopi domain experiments with per-experiment design parameters
    like `validities` / `rating_max`).
    """
    if not experiments:
        return []

    n = len(experiments)

    if share_template:
        first = experiments[0]
        print(f"Compiling experiment JS template (1/{n}) …", flush=True)
        template = first.compile(as_function=True, is_async=True)
        payload = []
        for i, exp in enumerate(experiments):
            print(f"\r  Building payload {i + 1}/{n}", end="", flush=True)
            timelines = [b.timeline for b in exp.blocks]
            js = exp.to_js_string_from_template(template, timelines)
            payload.append({
                "condition": condition_doc_fn(conditions[i]),
                "experiment_code": js,
            })
        print(flush=True)
        return payload

    payload = []
    for i, exp in enumerate(experiments):
        print(f"\r  Compiling experiment {i + 1}/{n}", end="", flush=True)
        js = exp.to_js_string(as_function=True, is_async=True)
        payload.append({
            "condition": condition_doc_fn(conditions[i]),
            "experiment_code": js,
        })
    print(flush=True)
    return payload


def _get_experiment_runner(credentials_path: str, *, time_out_seconds: int):
    global _experiment_runner
    if _experiment_runner is None:
        # The patched `firebase_runner` joins `autora_meta.<slot>.pId` into
        # each returned observation (envelope dict with `slot_key`,
        # `prolific_pid`, `obs`). Sweetbean's parse_autora_observations
        # picks up the prolific_pid from that envelope. The patches landed
        # in AutoResearch/autora-experiment-runner-firebase-prolific +
        # AutoResearch/autora-experiment-runner-experimentation-manager-firebase
        # so we use the canonical namespace and rely on the @main pin in
        # requirements.txt.
        from autora.experiment_runner.firebase_prolific import firebase_runner

        with open(credentials_path, encoding="utf-8") as f:
            firebase_credentials = json.load(f)
        _experiment_runner = firebase_runner(
            firebase_credentials=firebase_credentials,
            time_out=time_out_seconds,
            sleep_time=5,
        )
    return _experiment_runner


def _get_experiment_runner_prolific(
    credentials_path: str,
    prolific_settings_path: str,
    *,
    study_name: str,
    study_description: str,
    study_url: str,
    study_completion_time: int,
    completion_code: str,
    reward: int = 0,
):
    """Firebase + Prolific runner (recruitment on Prolific, data on Firebase).

    ``reward`` is the per-participant payment in the smallest currency
    unit (cents / pence). 0 (default) keeps the upstream auto-calc
    (20 * study_completion_time, i.e. $12/hour). Forwarded as-is to the
    vendored ``firebase_prolific_runner`` which forwards to ``setup_study``.
    """
    global _experiment_runner_prolific
    if _experiment_runner_prolific is None:
        from autora.experiment_runner.firebase_prolific import firebase_prolific_runner

        settings = _read_prolific_settings(prolific_settings_path)
        prolific_token = settings["token"].strip()
        # Optional project pin. The Prolific token belongs to a SHARED lab
        # account (Cocosci) that owns many member workspaces, so without a
        # project_id the study lands in an unpredictable default location and
        # the PUBLISH transition 400s (error_code 140007). Set "project_id" in
        # prolific_settings.json to the target project; None preserves the
        # upstream default-location behaviour for single-workspace accounts.
        project_id = (settings.get("project_id") or "").strip() or None

        with open(credentials_path, encoding="utf-8") as f:
            firebase_credentials = json.load(f)

        _experiment_runner_prolific = firebase_prolific_runner(
            firebase_credentials=firebase_credentials,
            prolific_token=prolific_token,
            sleep_time=5,
            study_name=study_name,
            study_description=study_description,
            study_url=study_url,
            study_completion_time=study_completion_time,
            completion_code=completion_code,
            reward=int(reward or 0),
            project_id=project_id,
        )
    return _experiment_runner_prolific


def run_online(
    experiments,
    conditions,
    *,
    credentials_path: str,
    is_prolific: bool,
    study_completion_time: int,
    prolific_settings_path: str | None = None,
    study_name: str = "autora",
    study_description: str = "autora online experiment",
    study_url: str | None = None,
    completion_code: str = "",
    reward: int = 0,
    share_template: bool = True,
):
    """
    Args:
        study_completion_time: study/condition timeout in **minutes**. Used by both
            backends: Prolific receives minutes (its API expects minutes); the plain
            Firebase runner receives ``minutes * 60`` as ``time_out`` (seconds).
        reward: Prolific per-participant payment in the smallest currency
            unit (cents / pence). 0 (default) keeps the upstream auto-calc
            (~$12/hour). Ignored when ``is_prolific=False``.
        share_template: when True (the default), all conditions share the
            first experiment's compiled Sweetbean template (fast; correct
            only when every experiment has identical stimulus structure).
            Set False when batching heterogeneous experiments — see
            ``_build_payload``.
    """
    if is_prolific:
        if not study_url:
            raise ValueError("Prolific requires a non-empty study_url.")
        if not prolific_settings_path:
            raise ValueError("Prolific requires prolific_settings_path.")
        # Narrow optionals for static type-checkers after validation above.
        _study_url: str = study_url
        _prolific_settings_path: str = prolific_settings_path

    def _condition_doc(condition):
        safe = _to_firestore_safe(condition)
        # Match the older working payload contract: store condition as a single
        # JSON string field under `condition.serialized`.
        return {
            "serialized": json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
        }

    payload = _build_payload(
        experiments, conditions, _condition_doc, share_template=share_template
    )
    print(f"Uploading {len(payload)} experiment(s) to Firebase …", flush=True)
    if is_prolific:
        return _get_experiment_runner_prolific(
            credentials_path,
            _prolific_settings_path,
            study_name=study_name,
            study_description=study_description,
            study_url=_study_url,
            study_completion_time=study_completion_time,
            completion_code=completion_code,
            reward=reward,
        )(payload)
    # Non-Prolific path: derive condition timeout from configured duration (minutes).
    return _get_experiment_runner(
        credentials_path,
        time_out_seconds=max(60, int(study_completion_time) * 60),
    )(payload)


def run_experiment(
    config,
    *,
    generate_experiment: Callable[..., Any],
    credentials_path: str,
    is_prolific: bool,
    study_completion_time: int,
    prolific_settings_path: str | None = None,
    study_name: str = "autora",
    study_description: str = "autora online experiment",
    study_url: str | None = None,
    completion_code: str = "",
    process_result: Callable[..., Any] | None = None,
    **generate_experiment_kwargs: Any,
):
    """
    Run an online study from a ``config`` dict. All further keyword arguments are
    forwarded to ``generate_experiment`` (design parameters live in domain
    generators, not here). Generators may return ``(experiments, conditions)`` or
    ``(experiments, conditions, process_result_kwargs)`` for online post-processing.

    ``study_completion_time`` is in **minutes** end-to-end (matches Prolific's API);
    ``run_online`` converts to seconds for the plain Firebase runner.
    """
    print("── Step 1/3: Generating experiments …", flush=True)
    gen_out = generate_experiment(config, **generate_experiment_kwargs)
    if isinstance(gen_out, tuple) and len(gen_out) == 3:
        experiments, conditions, process_meta = gen_out
    else:
        experiments, conditions = gen_out
        process_meta = {}

    print("── Step 2/3: Compiling JS & building payload …", flush=True)
    raw = run_online(
        experiments,
        conditions,
        credentials_path=credentials_path,
        is_prolific=is_prolific,
        prolific_settings_path=prolific_settings_path,
        study_name=study_name,
        study_description=study_description,
        study_url=study_url,
        study_completion_time=study_completion_time,
        completion_code=completion_code,
    )
    print("── Step 3/3: Processing results …", flush=True)
    if process_result is not None:
        processed = process_result(raw, **process_meta)
    else:
        processed = raw

    return processed
