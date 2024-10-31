import os
from datetime import timedelta
from typing import Any, Literal

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

VERBOSITY_MODE = Literal["silent", "transient", "verbose"]
FORCE_TERMINAL = os.getenv("FORCE_TERMINAL_MODE", "false").lower() == "true"


def show_total_elapsed_time(elapsed_seconds: float) -> None:
    elapsed_time_formatted = str(timedelta(seconds=int(elapsed_seconds)))
    rprint(f"Finished operation in [progress.elapsed]{elapsed_time_formatted}")


class TrackProgressSpinner(Progress):  # type: ignore[misc]
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
            return Text(f"{1/task.speed:.2f} s/it")  # noqa: FURB126


class TrackProgressBar(Progress):  # type: ignore[misc]
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
