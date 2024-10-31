"""Tests related to using pregenerated release indexes."""

from contextlib import nullcontext as does_not_raise
from typing import Optional

import pytest
from parametrization import Parametrization as P
from shapely import box
from shapely.geometry.base import BaseGeometry

from overturemaestro.release_index import (
    MINIMAL_SUPPORTED_RELEASE_VERSION,
    ReleaseVersionNotSupportedError,
    _check_release_version,
    get_available_theme_type_pairs,
    load_release_index,
    load_release_indexes,
)
from tests.conftest import TEST_RELEASE_VERSION


@pytest.mark.parametrize(
    "expectation,release_version",
    [
        (pytest.raises(ReleaseVersionNotSupportedError), "2024-03-12-alpha.0"),
        (does_not_raise(), MINIMAL_SUPPORTED_RELEASE_VERSION),
        (does_not_raise(), TEST_RELEASE_VERSION),
    ],
)  # type: ignore
def test_supported_release_version(expectation, release_version: str) -> None:
    """Test if raises errors for unsupported release versions."""
    with expectation:
        _check_release_version(release_version)


def test_get_available_theme_type_pairs(test_release_version: str) -> None:
    """Test is loading available theme type pairs works."""
    pairs = get_available_theme_type_pairs(test_release_version)
    assert len(pairs) == 14


@P.parameters("theme_type_pair", "geometry_filter", "expected_number_of_rows")  # type: ignore
@P.case(
    "Base water - no filter",
    ("base", "water"),
    None,
    74348,
)  # type: ignore
@P.case(
    "Buildings - bbox filter",
    ("buildings", "building"),
    box(-0.120077, 51.498164, -0.090809, 51.508849),
    4,
)  # type: ignore
def test_load_release_index(
    test_release_version,
    theme_type_pair: tuple[str, str],
    geometry_filter: Optional[BaseGeometry],
    expected_number_of_rows: int,
) -> None:
    """Test is load_release_index function works."""
    theme_value, type_value = theme_type_pair
    release_index = load_release_index(
        release=test_release_version,
        theme=theme_value,
        type=type_value,
        geometry_filter=geometry_filter,
    )
    assert (
        len(release_index) == expected_number_of_rows
    ), f"Mismatch in number of rows: {len(release_index)} vs {expected_number_of_rows}"
    assert (
        release_index["filename"]
        .str.startswith(
            f"overturemaps-us-west-2/release/{test_release_version}/theme={theme_value}/type={type_value}"
        )
        .all()
    ), "Some rows don't match expected filename structure"


def test_load_release_indexes(test_release_version: str) -> None:
    """Test is load_release_indexes function works."""
    theme_type_pairs = [("base", "water"), ("buildings", "building")]
    combined_release_index = load_release_indexes(
        release=test_release_version,
        theme_type_pairs=theme_type_pairs,
    )
    total_expected_length = sum(
        len(
            load_release_index(
                release=test_release_version,
                theme=theme_type_pair[0],
                type=theme_type_pair[1],
            )
        )
        for theme_type_pair in theme_type_pairs
    )
    assert len(combined_release_index) == total_expected_length
