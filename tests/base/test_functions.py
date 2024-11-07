"""Tests for public API functions."""

from contextlib import nullcontext as does_not_raise
from typing import Any, Optional

import pytest
from parametrization import Parametrization as P

from overturemaestro._exceptions import MissingColumnError
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


@P.parameters("columns_to_download", "expectation")  # type: ignore
@P.case("No columns", [], does_not_raise())  # type: ignore
@P.case("No columns", None, does_not_raise())  # type: ignore
@P.case("Base columns", ["subtype", "class", "id"], does_not_raise())  # type: ignore
@P.case("Base columns without id", ["subtype", "class"], does_not_raise())  # type: ignore
@P.case(
    "Non-existent columns", ["nonexistent_column"], pytest.raises(MissingColumnError)
)  # type: ignore
def test_columns_download(
    test_release_version: str, columns_to_download: Optional[list[str]], expectation: Any
) -> None:
    """Test if columns download restricting works properly."""
    with expectation:
        gdf = convert_bounding_box_to_geodataframe(
            theme="buildings",
            type="building",
            release=test_release_version,
            bbox=(-0.120077, 51.498164, -0.090809, 51.508849),
            columns_to_download=columns_to_download,
            verbosity_mode="verbose",
            ignore_cache=True,
        )
        assert "geometry" in gdf.columns
        assert all(
            column_name in gdf.columns
            for column_name in (columns_to_download or [])
            if column_name != "id" # skip indexed column
        )
