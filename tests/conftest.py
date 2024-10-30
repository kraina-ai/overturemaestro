"""Common components for tests."""

import os
from pathlib import Path

import pytest
from pytest import Item
from shapely import Polygon, box, to_geojson, to_wkt

from overturemaestro.release_index import download_existing_release_index

TEST_RELEASE_VERSION = "2024-08-20.0"


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
