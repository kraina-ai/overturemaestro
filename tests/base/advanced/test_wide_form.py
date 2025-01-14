"""Tests for public API advanced wide form functions."""

from contextlib import nullcontext as does_not_raise
from typing import Any, Optional

import pytest
from parametrization import Parametrization as P

from overturemaestro._exceptions import HierarchyDepthOutOfBoundsError
from overturemaestro.advanced_functions import (
    convert_bounding_box_to_wide_form_geodataframe,
    convert_bounding_box_to_wide_form_geodataframe_for_all_types,
    convert_bounding_box_to_wide_form_geodataframe_for_multiple_types,
    convert_bounding_box_to_wide_form_parquet,
)
from overturemaestro.advanced_functions.wide_form import THEME_TYPE_CLASSIFICATION
from overturemaestro.data_downloader import PYARROW_FILTER
from tests.conftest import bbox


@pytest.mark.parametrize(
    "theme_type_pair",
    list(THEME_TYPE_CLASSIFICATION.keys()),
)  # type: ignore
def test_theme_type_pairs(theme_type_pair: tuple[str, str], test_release_version: str) -> None:
    """Test if all theme type pairs are working."""
    convert_bounding_box_to_wide_form_parquet(
        theme=theme_type_pair[0],
        type=theme_type_pair[1],
        bbox=bbox(),
        release=test_release_version,
        verbosity_mode="verbose",
        ignore_cache=True,
    )


@pytest.mark.parametrize(
    "hierarchy_value",
    [None, 1],
)  # type: ignore
def test_all_theme_type_pairs(hierarchy_value: Optional[int], test_release_version: str) -> None:
    """Test if downloading all theme type pairs at once is working."""
    gdf = convert_bounding_box_to_wide_form_geodataframe_for_all_types(
        bbox=bbox(),
        release=test_release_version,
        hierarchy_depth=hierarchy_value,
        verbosity_mode="verbose",
        ignore_cache=True,
    )
    prepared_theme_type_prefixes = tuple("|".join(t) for t in THEME_TYPE_CLASSIFICATION.keys())
    feature_columns = [
        column_name for column_name in gdf.columns if column_name not in ("id", "geometry")
    ]
    assert all(
        column_name.startswith(prepared_theme_type_prefixes) for column_name in feature_columns
    )
    assert (gdf.dtypes.loc[feature_columns] == "bool").all()


# TODO: add hierarchy working and not working
@P.parameters("theme_type_pairs", "pyarrow_filters", "hierarchy_value", "expectation")  # type: ignore
@P.case(
    "Basic example without filters",
    [("base", "water"), ("base", "infrastructure")],
    None,
    None,
    does_not_raise(),
)  # type: ignore
@P.case(
    "Basic example with filters",
    [("base", "water"), ("base", "infrastructure")],
    [[("subtype", "==", "river")], [("class", "==", "bridge")]],
    None,
    does_not_raise(),
)  # type: ignore
@P.case(
    "Empty pyarrow filters list",
    [("base", "water"), ("base", "infrastructure")],
    [],
    None,
    pytest.raises(ValueError),
)  # type: ignore
@P.case(
    "Shorter pyarrow filters list",
    [("places", "place"), ("base", "infrastructure")],
    [[("confidence", ">", 0.95)]],
    None,
    pytest.raises(ValueError),
)  # type: ignore
@P.case(
    "Longer pyarrow filters list",
    [("places", "place"), ("base", "infrastructure")],
    [[("confidence", ">", 0.95)], [("class", "==", "bridge")], [("confidence", ">", 0.95)]],
    None,
    pytest.raises(ValueError),
)  # type: ignore
def test_multiple_theme_type_pairs(
    test_release_version: str,
    theme_type_pairs: list[tuple[str, str]],
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]],
    hierarchy_value: Optional[int],
    expectation: Any,
) -> None:
    """Test if downloading multiple theme type pairs at once is working."""
    with expectation:
        gdf = convert_bounding_box_to_wide_form_geodataframe_for_multiple_types(
            theme_type_pairs=theme_type_pairs,
            bbox=bbox(),
            release=test_release_version,
            hierarchy_depth=hierarchy_value,
            pyarrow_filters=pyarrow_filters,
            verbosity_mode="verbose",
            ignore_cache=True,
        )
        prepared_theme_type_prefixes = tuple("|".join(t) for t in theme_type_pairs)
        feature_columns = [
            column_name for column_name in gdf.columns if column_name not in ("id", "geometry")
        ]
        assert all(
            column_name.startswith(prepared_theme_type_prefixes) for column_name in feature_columns
        )
        assert (gdf.dtypes.loc[feature_columns] == "bool").all()


@P.parameters("hierarchy_value", "theme_type_pair", "expectation")  # type: ignore
@P.case("Empty value", None, ("base", "water"), does_not_raise())  # type: ignore
@P.case("Zero", 0, ("base", "water"), pytest.raises(HierarchyDepthOutOfBoundsError))  # type: ignore
@P.case("First value", 1, ("base", "water"), does_not_raise())  # type: ignore
@P.case("Second value", 2, ("base", "water"), does_not_raise())  # type: ignore
@P.case("Third value", 3, ("base", "water"), pytest.raises(HierarchyDepthOutOfBoundsError))  # type: ignore
def test_hierarchy_values(
    test_release_version: str,
    hierarchy_value: Optional[int],
    theme_type_pair: tuple[str, str],
    expectation: Any,
) -> None:
    """Test if hierarchy values are parsed correctly."""
    with expectation:
        gdf = convert_bounding_box_to_wide_form_geodataframe(
            theme=theme_type_pair[0],
            type=theme_type_pair[1],
            bbox=bbox(),
            hierarchy_depth=hierarchy_value,
            release=test_release_version,
            verbosity_mode="verbose",
            ignore_cache=True,
        )
        assert "geometry" in gdf.columns
        feature_columns = [
            column_name for column_name in gdf.columns if column_name not in ("id", "geometry")
        ]
        if hierarchy_value == 1:
            # only first group in column, no combinations
            assert all("|" not in column_name for column_name in feature_columns)
        else:
            # combined column names
            assert all("|" in column_name for column_name in feature_columns)
        assert (gdf.dtypes.loc[feature_columns] == "bool").all()
