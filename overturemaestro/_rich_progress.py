"""Progress bars over the shared progress display, with the subset of the rich `Progress` API."""

from collections.abc import Generator, Iterable
from contextlib import ExitStack
from types import TracebackType
from typing import Any, Optional, TypeVar

from rq_geo_toolkit.rich_utils import (
    FORCE_TERMINAL,
    STEP,
    VERBOSITY_MODE,
    ProgressDisplay,
    ProgressTask,
    progress_display,
)

__all__ = [
    "FORCE_TERMINAL",
    "VERBOSITY_MODE",
    "TrackProgressBar",
    "TrackProgressSpinner",
    "add_steps",
    "skip_step",
    "step_group",
]

T = TypeVar("T")


class _SharedDisplayContext:
    def __init__(self, verbosity_mode: VERBOSITY_MODE = "verbose") -> None:
        self.verbosity_mode = verbosity_mode
        self._stack = ExitStack()

    def _enter_display(self) -> ProgressDisplay:
        return self._stack.enter_context(progress_display(self.verbosity_mode))

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._stack.__exit__(exc_type, exc_val, exc_tb)


class TrackProgressSpinner(_SharedDisplayContext):
    def __init__(
        self,
        task_name: str,
        verbosity_mode: VERBOSITY_MODE = "verbose",
        step: Optional[STEP] = None,
    ) -> None:
        super().__init__(verbosity_mode)
        self.task_name = task_name
        self.step = step

    def __enter__(self) -> None:
        self._stack.enter_context(self._enter_display().spinner(self.task_name, step=self.step))


class TrackProgressBar(_SharedDisplayContext):
    def __enter__(self) -> "TrackProgressBar":
        self._display = self._enter_display()
        self._tasks: list[ProgressTask] = []
        return self

    @property
    def task_ids(self) -> list[int]:
        return list(range(len(self._tasks)))

    def add_task(
        self, description: str, total: Optional[float] = None, step: Optional[STEP] = None
    ) -> int:
        self._tasks.append(
            self._stack.enter_context(self._display.bar(description, total, step=step))
        )
        return len(self._tasks) - 1

    def update(self, task_id: int, completed: Optional[float] = None, **_: Any) -> None:
        self._tasks[task_id].update(completed)

    def track(
        self,
        sequence: Iterable[T],
        total: Optional[float] = None,
        description: str = "Working...",
        step: Optional[STEP] = None,
    ) -> Generator[T, None, None]:
        with self._display.bar(description, step=step) as task:
            yield from task.track(sequence, total)

    def print_progress(self) -> None:
        self._tasks[0].print_progress()


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
