from datetime import timedelta
from threading import Lock
from typing import Any

from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    Text,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rq_geo_toolkit.rich_utils import FORCE_TERMINAL, VERBOSITY_MODE

__all__ = [
    "show_total_elapsed_time",
    "FORCE_TERMINAL",
    "VERBOSITY_MODE",
    "TrackProgressSpinner",
    "TrackProgressBar",
]


def show_total_elapsed_time(elapsed_seconds: float) -> None:
    elapsed_time_formatted = str(timedelta(seconds=int(elapsed_seconds)))
    rprint(f"Finished operation in [progress.elapsed]{elapsed_time_formatted}")


_DISPLAY_LOCK = Lock()


class _ConcurrentProgress(Progress):  # type: ignore[misc]
    """Let overlapping operations proceed without competing for a live display."""

    def __init__(self, *columns: Any, **kwargs: Any) -> None:
        super().__init__(*columns, **kwargs)
        self._owns_display = False
        self._suppressed_display = False

    def start(self) -> None:
        """Display one operation; keep concurrent operations running silently."""
        if self.disable or self._owns_display:
            return
        if not _DISPLAY_LOCK.acquire(blocking=False):
            self.disable = True
            self._suppressed_display = True
            return
        self._owns_display = True
        try:
            super().start()
        except BaseException:
            self._owns_display = False
            _DISPLAY_LOCK.release()
            raise

    def stop(self) -> None:
        """Release only the display owned by this operation."""
        if self._owns_display:
            try:
                super().stop()
            finally:
                self._owns_display = False
                _DISPLAY_LOCK.release()
        if self._suppressed_display:
            self.disable = False
            self._suppressed_display = False


class TrackProgressSpinner(_ConcurrentProgress):
    def __init__(
        self,
        task_name: str,
        verbosity_mode: VERBOSITY_MODE = "verbose",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            refresh_per_second=1,
            speed_estimate_period=3600,
            transient=verbosity_mode == "transient",
            disable=verbosity_mode == "silent",
            console=Console(force_interactive=False, force_jupyter=False, force_terminal=True)
            if FORCE_TERMINAL
            else None,
            **kwargs,
        )
        self.task_name = task_name

    def __enter__(self):  # type: ignore
        self.add_task(description=self.task_name, total=None)
        self.start()

    def __exit__(self, exc_type, exc_value, exc_tb):  # type: ignore
        self.stop()


class SpeedColumn(ProgressColumn):  # type: ignore[misc]
    def render(self, task: "Task") -> Text:
        if task.speed is None:
            return Text("0 it/s")
        elif task.speed >= 1:
            return Text(f"{task.speed:.2f} it/s")
        else:
            return Text(f"{1 / task.speed:.2f} s/it")  # noqa: FURB126


class TrackProgressBar(_ConcurrentProgress):
    def __init__(
        self,
        verbosity_mode: VERBOSITY_MODE = "verbose",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            SpinnerColumn(),
            TextColumn(
                "[progress.description]{task.description}"
                " [progress.percentage]{task.percentage:>3.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("•"),
            SpeedColumn(),
            refresh_per_second=1,
            speed_estimate_period=3600,
            transient=verbosity_mode == "transient",
            disable=verbosity_mode == "silent",
            console=Console(force_interactive=False, force_jupyter=False, force_terminal=True)
            if FORCE_TERMINAL
            else None,
            **kwargs,
        )
        self.verbosity_mode = verbosity_mode

    def print_progress(self) -> None:
        if self.verbosity_mode != "silent":
            current_task = self.tasks[0]
            mofn = self.columns[3].render(current_task)
            elapsed = self.columns[5].render(current_task)
            remaining = self.columns[7].render(current_task)
            speed = self.columns[9].render(current_task)
            self.print(
                f"{current_task.description} {current_task.percentage:>6.2f}% • "
                f"{mofn} • {elapsed} < {remaining} • {speed}"
            )
