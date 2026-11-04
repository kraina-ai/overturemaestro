"""Step helpers over the shared progress display."""

from rq_geo_toolkit.rich_utils import (
    FORCE_TERMINAL,
    VERBOSITY_MODE,
    ProgressTask,
    progress_display,
)

__all__ = [
    "FORCE_TERMINAL",
    "VERBOSITY_MODE",
    "ProgressTask",
    "add_steps",
    "progress_display",
    "skip_step",
    "step_group",
]


def add_steps(count: int) -> None:
    """Extend the total number of steps of the current operation."""
    with progress_display() as display:
        if display.steps:
            display.steps.add_steps(count)


def step_group(description: str, total: int = 0) -> None:
    """Start the next major step of the current operation, with minor steps shown under it."""
    with progress_display() as display:
        display.step_group(description, total)


def skip_step() -> None:
    """Count a step of the current operation that has nothing to do."""
    with progress_display() as display:
        if display.steps:
            display.steps.next_step()
