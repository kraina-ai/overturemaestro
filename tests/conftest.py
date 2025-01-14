"""Common components for tests."""

import os
import shutil
from pathlib import Path

import pytest
from pytest import Item
from shapely import Polygon, box, to_geojson, to_wkt

from overturemaestro.conftest import TEST_RELEASE_VERSIONS_LIST, patch_get_available_versions
from overturemaestro.release_index import download_existing_release_index

TEST_RELEASE_VERSION = "2024-08-20.0"

__all__ = [
    "patch_get_available_versions",
    "test_release_version",
    "TEST_RELEASE_VERSION",
    "TEST_RELEASE_VERSIONS_LIST",
    "bbox",
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
