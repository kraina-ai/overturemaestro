"""Common components for tests."""

import os
import shutil
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from pytest import Item
from rq_geo_toolkit.rich_utils import ProgressDisplay, StepCounter
from shapely import Polygon, box, to_geojson, to_wkt

from overturemaestro.release_index import (
    download_existing_release_index,
    get_newest_release_version,
)

TEST_RELEASE_VERSION = get_newest_release_version()

__all__ = [
    "test_release_version",
    "TEST_RELEASE_VERSION",
    "bbox",
    "check_step_counts",
    "geometry_bbox_str",
    "geometry_box",
    "geometry_wkt",
    "geometry_geojson",
    "geometry_boundary_file_path",
]


def pytest_runtest_setup(item: Item) -> None:
    """Setup python encoding before `pytest_runtest_call(item)`."""
    os.environ["PYTHONIOENCODING"] = "utf-8"


@pytest.fixture(scope="session")  # type: ignore
def test_release_version() -> str:
    """Release version for tests."""
    return TEST_RELEASE_VERSION


@pytest.fixture(autouse=True, scope="session")  # type: ignore
def download_release_index(test_release_version: str) -> None:
    """Download release index for testing purposes."""
    download_existing_release_index(release=test_release_version)


@pytest.fixture(autouse=True)  # type: ignore
def check_step_counts(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Fail a test when its operations run a different number of steps than they declare.

    Major steps are compared with `total_steps` and minor steps with the total of their group. Every
    step is checked for exceeding the total. Missing steps are checked when a group ends and when an
    operation finishes, unless it failed with an exception. Steps abandoned by a retry aren't
    missing, so a group ending at a rewound step number isn't checked.
    """
    errors: list[str] = []
    highest_major_steps: dict[StepCounter, int] = {}
    original_next_step = StepCounter.next_step
    original_reset = StepCounter.reset
    original_close = ProgressDisplay._close

    def check_minor_steps(major_step: int, minor_steps: int, total: int) -> None:
        if minor_steps != total:
            errors.append(f"Step {major_step} has {minor_steps} of {total} minor steps.")

    def check_finished(counter: StepCounter) -> None:
        if counter.is_new:
            return
        check_minor_steps(
            counter.major_step_number,
            counter.minor_step_number or 0,
            counter.minor_steps_total or 0,
        )
        if counter.major_step_number != counter.total_steps:
            errors.append(
                f"Finished after {counter.major_step_number} of {counter.total_steps} steps."
            )

    def next_step(self: StepCounter, *args: Any, **kwargs: Any) -> str:
        major_step = self.major_step_number
        minor_steps, minor_total = self.minor_step_number or 0, self.minor_steps_total or 0
        label = original_next_step(self, *args, **kwargs)

        is_rewound = major_step < highest_major_steps.get(self, major_step)
        if self.major_step_number > major_step and not is_rewound:
            check_minor_steps(major_step, minor_steps, minor_total)
        highest_major_steps[self] = max(self.major_step_number, highest_major_steps.get(self, 0))

        if self.major_step_number > self.total_steps or (self.minor_step_number or 0) > (
            self.minor_steps_total or 0
        ):
            errors.append(f"Step {label.strip()} exceeds its total.")
        return label  # type: ignore[no-any-return]

    def reset(self: StepCounter, *args: Any, **kwargs: Any) -> None:
        check_finished(self)
        highest_major_steps.pop(self, None)
        original_reset(self, *args, **kwargs)

    def close(self: ProgressDisplay) -> None:
        # called in a `finally` block, so an active exception means a failed operation
        if self.steps is not None and sys.exc_info()[0] is None:
            check_finished(self.steps)
        original_close(self)

    monkeypatch.setattr(StepCounter, "next_step", next_step)
    monkeypatch.setattr(StepCounter, "reset", reset)
    monkeypatch.setattr(ProgressDisplay, "_close", close)
    yield
    assert not errors


@pytest.fixture(autouse=True, scope="session")  # type: ignore
def copy_geocode_cache() -> None:
    """Load cached geocoding results."""
    existing_cache_directory = Path(__file__).parent / "test_files" / "geocoding_cache"
    geocoding_cache_directory = Path("cache")
    geocoding_cache_directory.mkdir(exist_ok=True)
    for file_path in existing_cache_directory.glob("*.json"):
        destination_path = geocoding_cache_directory / file_path.name
        shutil.copy(file_path, destination_path)


def bbox() -> tuple[float, float, float, float]:
    """Bounding Box."""
    return (7.416486207767861, 43.7310867041912, 7.421931388477276, 43.73370705597216)


def geometry_bbox_str() -> str:
    """Bounding Box in str form."""
    return ",".join(map(str, bbox()))


def geometry_box() -> Polygon:
    """Geometry box."""
    return box(*bbox())


def geometry_wkt() -> str:
    """Geometry box in WKT form."""
    return str(to_wkt(geometry_box()))


def geometry_geojson() -> str:
    """Geometry box in GeoJSON form."""
    return str(to_geojson(geometry_box()))


def geometry_boundary_file_path() -> str:
    """Geometry Monaco boundary file path."""
    return str(Path(__file__).parent / "test_files" / "monaco_boundary.geojson")
