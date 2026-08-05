from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.experiment import Experiment
    from src.theory import Theory


SYSTEM_PROMPT = """\
You are a renowned cognitive scientist designing an experiment in the {domain} domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by \
your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — \
how they respond differently to specific stimuli in addition to differences in overall \
performance.
"""


TEMPLATE = """\
## EXPERIMENTAL DOMAIN
{experiment_description}

{design_header}

Subjects see the following instructions:
{introduction_text}

## ADVOCATED THEORY
{advocating_description}

## COMPETING THEORY
{competing_description}

## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
{ledger}

## RESPONSE FORMAT
Return a JSON object with the following fields:
{instruction_format}
"""





def _format_ledger(experiments: list["Experiment"]) -> str:
    if not experiments:
        return "(none yet)"
    return "\n\n".join(
        f"[{i}] {e.rationale or '(no rationale)'}" for i, e in enumerate(experiments)
    )


def render(
    *,
    experiment_class: type["Experiment"],
    advocating: "Theory",
    competing: "Theory",
    ledger: list["Experiment"] | None = None,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for proposing an experiment.

    All design facts (domain description, n_blocks/n_repeats, instructions, JSON
    schema, parameter variables) are pulled directly from `experiment_class` —
    one source of truth.
    """
    system_prompt = SYSTEM_PROMPT.format(domain=experiment_class.name)

    user_prompt = TEMPLATE.format(
        experiment_description=experiment_class.description,
        design_header=experiment_class.pretty_print_header(),
        introduction_text=experiment_class.introduction_text,
        advocating_description=advocating.pretty_print(),
        competing_description=competing.pretty_print(),
        ledger=_format_ledger(ledger or []),
        instruction_format=experiment_class.instruction_format(),
    )
    return system_prompt, user_prompt
