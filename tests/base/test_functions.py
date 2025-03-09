"""Tests for public API functions."""

from contextlib import nullcontext as does_not_raise
from typing import Any, Optional

import pyarrow.parquet as pq
import pytest
from parametrization import Parametrization as P

from overturemaestro._constants import GEOMETRY_COLUMN, PARQUET_COMPRESSION
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
    result = convert_bounding_box_to_parquet(
        theme=theme_type_pair[0],
        type=theme_type_pair[1],
        bbox=bbox(),
        release=test_release_version,
        verbosity_mode="verbose",
        ignore_cache=True,
    )
    pq_file = pq.ParquetFile(result).metadata

    if pq_file.num_row_groups > 0:
        assert pq_file.row_group(0).column(0).compression.lower() == PARQUET_COMPRESSION.lower()


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
        assert GEOMETRY_COLUMN in gdf.columns
        assert all(
            column_name in gdf.columns
            for column_name in (columns_to_download or [])
            if column_name != "id"  # skip indexed column
        )


def test_empty_region(test_release_version: str) -> None:
    """Test if regions without data are properly parsed."""
    gdf = convert_bounding_box_to_geodataframe(
        "places", "place", (-10, -10, -9.9, -9.9), release=test_release_version
    )

    assert gdf.empty
    assert all(
        c in gdf.columns
        for c in (
            GEOMETRY_COLUMN,
            "bbox",
            "version",
            "sources",
            "names",
            "categories",
            "confidence",
            "websites",
            "socials",
            "emails",
            "phones",
            "brand",
            "addresses",
        )
    )
