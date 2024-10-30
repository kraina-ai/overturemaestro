"""Tests for public API functions."""

import pytest

from overturemaestro.functions import (
    convert_bounding_box_to_geodataframe,
    convert_bounding_box_to_parquet,
)
from overturemaestro.release_index import get_available_theme_type_pairs
from tests.conftest import TEST_RELEASE_VERSION, bbox


@pytest.mark.parametrize(
    "theme_type_pair",
    get_available_theme_type_pairs(TEST_RELEASE_VERSION),
)  # type: ignore
def test_theme_type_pairs(theme_type_pair: tuple[str, str], test_release_version: str) -> None:
    """Test if all theme type pairs are working."""
    convert_bounding_box_to_parquet(
        theme=theme_type_pair[0],
        type=theme_type_pair[1],
        bbox=bbox(),
        release=test_release_version,
        verbosity_mode="verbose",
        ignore_cache=True,
    )


def test_pyarrow_filtering(test_release_version: str) -> None:
    """Test if pyarrow filtering works properly."""
    gdf = convert_bounding_box_to_geodataframe(
        theme="places",
        type="place",
        release=test_release_version,
        bbox=(-0.120077, 51.498164, -0.090809, 51.508849),
        pyarrow_filter=[
            [
                (("categories", "primary"), "=", "museum"),
                ("confidence", ">", 0.95),
            ]
        ],
        verbosity_mode="verbose",
        ignore_cache=True,
    )
    assert (gdf["confidence"] > 0.95).all()
    assert (gdf["categories"].apply(lambda x: x["primary"] == "museum")).all()
