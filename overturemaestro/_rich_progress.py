from typing import Any

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
            **kwargs,
        )

    def print_progress(self) -> None:
        current_task = self.tasks[0]
        mofn = self.columns[3].render(current_task)
        elapsed = self.columns[5].render(current_task)
        remaining = self.columns[7].render(current_task)
        speed = self.columns[9].render(current_task)
        self.print(
            f"{current_task.description} {current_task.percentage:>6.2f}% • "
            f"{mofn} • {elapsed} < {remaining} • {speed}"
        )
