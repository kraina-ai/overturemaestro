"""Tests for public API advanced wide form functions."""

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any, Optional

import pytest
from parametrization import Parametrization as P
from shapely import box

from overturemaestro._constants import GEOMETRY_COLUMN
from overturemaestro._exceptions import (
    HierarchyDepthOutOfBoundsWarning,
    NegativeHierarchyDepthError,
)
from overturemaestro.advanced_functions import (
    convert_bounding_box_to_wide_form_geodataframe,
    convert_bounding_box_to_wide_form_geodataframe_for_all_types,
    convert_bounding_box_to_wide_form_geodataframe_for_multiple_types,
    convert_bounding_box_to_wide_form_parquet,
)
from overturemaestro.advanced_functions.wide_form import (
    _generate_result_file_path,
    get_all_possible_column_names,
    get_theme_type_classification,
)
from overturemaestro.data_downloader import PYARROW_FILTER
from overturemaestro.release_index import MINIMAL_SUPPORTED_RELEASE_VERSION
from tests.conftest import TEST_RELEASE_VERSION, bbox


@pytest.fixture(scope="session")  # type: ignore
def wide_form_working_directory() -> Path:
    """Working directory for tests."""
    return Path("files/wide_form")


@pytest.fixture(autouse=True, scope="session")  # type: ignore
def clear_working_directory(wide_form_working_directory: Path) -> None:
    """Clear working directory."""
    for file_path in wide_form_working_directory.glob("**/*.parquet"):
        file_path.unlink()


@pytest.mark.parametrize(
    "theme_type_pair",
    list(get_theme_type_classification(release=TEST_RELEASE_VERSION).keys()),
)  # type: ignore
def test_theme_type_pairs(
    test_release_version: str, wide_form_working_directory: Path, theme_type_pair: tuple[str, str]
) -> None:
    """Test if all theme type pairs are working."""
    convert_bounding_box_to_wide_form_parquet(
        theme=theme_type_pair[0],
        type=theme_type_pair[1],
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
    )


@pytest.mark.parametrize(
    "hierarchy_value",
    [None, 1],
)  # type: ignore
def test_all_theme_type_pairs(
    test_release_version: str, wide_form_working_directory: Path, hierarchy_value: Optional[int]
) -> None:
    """Test if downloading all theme type pairs at once is working."""
    gdf = convert_bounding_box_to_wide_form_geodataframe_for_all_types(
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        hierarchy_depth=hierarchy_value,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
    )
    prepared_theme_type_prefixes = tuple(
        "|".join(t) for t in get_theme_type_classification(release=test_release_version).keys()
    )
    feature_columns = [
        column_name for column_name in gdf.columns if column_name not in ("id", GEOMETRY_COLUMN)
    ]
    assert all(
        column_name.startswith(prepared_theme_type_prefixes) for column_name in feature_columns
    )
    assert (gdf.dtypes.loc[feature_columns] == "bool").all()


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
    "Empty example with filters",
    [("base", "water"), ("base", "infrastructure")],
    [[("subtype", "==", "nonexistent")], [("class", "==", "nonexistent")]],
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
@P.case(
    "Basic example with hierarchy depth",
    [("base", "water"), ("base", "land_cover")],
    None,
    1,
    does_not_raise(),
)  # type: ignore
@P.case(
    "Example with wrong hierarchy depth",
    [("base", "water"), ("base", "land_cover")],
    None,
    2,
    pytest.warns(HierarchyDepthOutOfBoundsWarning),
)  # type: ignore
def test_multiple_theme_type_pairs(
    test_release_version: str,
    wide_form_working_directory: Path,
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
            working_directory=wide_form_working_directory,
            hierarchy_depth=hierarchy_value,
            pyarrow_filters=pyarrow_filters,
            verbosity_mode="verbose",
            ignore_cache=False,
            include_all_possible_columns=False,
        )
        prepared_theme_type_prefixes = tuple("|".join(t) for t in theme_type_pairs)
        feature_columns = [
            column_name for column_name in gdf.columns if column_name not in ("id", GEOMETRY_COLUMN)
        ]
        assert all(
            column_name.startswith(prepared_theme_type_prefixes) for column_name in feature_columns
        )
        assert (gdf.dtypes.loc[feature_columns] == "bool").all()


@P.parameters("hierarchy_value", "theme_type_pairs", "expectation")  # type: ignore
@P.case("Empty value", None, [("base", "water")], does_not_raise())  # type: ignore
@P.case("Zero", 0, [("base", "water")], does_not_raise())  # type: ignore
@P.case("Negative", -1, [("base", "water")], pytest.raises(NegativeHierarchyDepthError))  # type: ignore
@P.case("First value", 1, [("base", "water")], does_not_raise())  # type: ignore
@P.case("Second value", 2, [("base", "water")], does_not_raise())  # type: ignore
@P.case("Third value", 3, [("base", "water")], pytest.warns(HierarchyDepthOutOfBoundsWarning))  # type: ignore
@P.case("Single value as list", [0], [("base", "water")], does_not_raise())  # type: ignore
@P.case("Multiple values wrong length", [0, 0], [("base", "water")], pytest.raises(ValueError))  # type: ignore
@P.case(
    "Multiple values same length",
    [0, 0],
    [("base", "water"), ("base", "infrastructure")],
    does_not_raise(),
)  # type: ignore
@P.case(
    "Multiple values negative wrong length",
    [-1, -1],
    [("base", "water")],
    pytest.raises(ValueError),
)  # type: ignore
@P.case(
    "Multiple values negative same length",
    [-1, -1],
    [("base", "water"), ("base", "infrastructure")],
    pytest.raises(NegativeHierarchyDepthError),
)  # type: ignore
@P.case(
    "Multiple values same length with warning",
    [99, 99],
    [("base", "water"), ("base", "infrastructure")],
    pytest.warns(HierarchyDepthOutOfBoundsWarning),
)  # type: ignore
def test_hierarchy_values(
    test_release_version: str,
    wide_form_working_directory: Path,
    hierarchy_value: Optional[int],
    theme_type_pairs: list[tuple[str, str]],
    expectation: Any,
) -> None:
    """Test if hierarchy values are parsed correctly."""
    with expectation:
        gdf = convert_bounding_box_to_wide_form_geodataframe_for_multiple_types(
            theme_type_pairs=theme_type_pairs,
            bbox=bbox(),
            hierarchy_depth=hierarchy_value,
            release=test_release_version,
            working_directory=wide_form_working_directory,
            verbosity_mode="verbose",
            ignore_cache=False,
            include_all_possible_columns=False,
        )
        assert GEOMETRY_COLUMN in gdf.columns
        feature_columns = [
            column_name for column_name in gdf.columns if column_name not in ("id", GEOMETRY_COLUMN)
        ]
        assert all("|" in column_name for column_name in feature_columns)
        assert (gdf.dtypes.loc[feature_columns] == "bool").all()


@pytest.mark.parametrize(
    "theme_type_pair",
    [("places", "place"), ("base", "water")],
)  # type: ignore
def test_include_all_possible_columns_parameter(
    test_release_version: str, wide_form_working_directory: Path, theme_type_pair: tuple[str, str]
) -> None:
    """Check if include_all_possible_columns works as intended."""
    theme_value, type_value = theme_type_pair
    pruned_dataset = convert_bounding_box_to_wide_form_geodataframe(
        theme=theme_value,
        type=type_value,
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
    ).drop(columns=GEOMETRY_COLUMN)

    full_dataset = convert_bounding_box_to_wide_form_geodataframe(
        theme=theme_value,
        type=type_value,
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=True,
    ).drop(columns=GEOMETRY_COLUMN)

    all_possible_columns = set(
        get_all_possible_column_names(
            theme=theme_value, type=type_value, release=test_release_version
        )
    )
    pruned_columns = set(pruned_dataset.columns)
    full_columns = set(full_dataset.columns)

    assert all_possible_columns == full_columns
    assert len(pruned_columns) < len(full_columns)

    columns_difference = full_columns.difference(pruned_columns)

    assert full_dataset.sum().loc[list(columns_difference)].sum() == 0


def test_empty_region(
    test_release_version: str,
    wide_form_working_directory: Path,
) -> None:
    """Test if regions without data are properly parsed."""
    gdf = convert_bounding_box_to_wide_form_geodataframe(
        "places",
        "place",
        (-10, -10, -9.9, -9.9),
        release=test_release_version,
        working_directory=wide_form_working_directory,
    ).drop(columns=GEOMETRY_COLUMN)

    all_possible_columns = set(
        get_all_possible_column_names(theme="places", type="place", release=test_release_version)
    )

    assert gdf.empty
    assert set(gdf.columns) == all_possible_columns


def test_confidence_parameter(
    test_release_version: str,
    wide_form_working_directory: Path,
) -> None:
    """Test if confidence parameter for places is working."""
    default_confidence = convert_bounding_box_to_wide_form_geodataframe(
        theme="places",
        type="place",
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
    )
    higher_confidence = convert_bounding_box_to_wide_form_geodataframe(
        theme="places",
        type="place",
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
        places_minimal_confidence=0.95,
    )
    lower_confidence = convert_bounding_box_to_wide_form_geodataframe(
        theme="places",
        type="place",
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
        places_minimal_confidence=0.05,
    )

    assert len(default_confidence) > len(higher_confidence)
    assert len(default_confidence) < len(lower_confidence)


def test_places_use_primary_category_only_parameter(
    test_release_version: str,
    wide_form_working_directory: Path,
) -> None:
    """Test if places_use_primary_category_only parameter is working."""
    primary_only = convert_bounding_box_to_wide_form_geodataframe(
        theme="places",
        type="place",
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
        places_use_primary_category_only=True,
    ).drop(columns=GEOMETRY_COLUMN)

    all_categories = convert_bounding_box_to_wide_form_geodataframe(
        theme="places",
        type="place",
        bbox=bbox(),
        release=test_release_version,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
        places_use_primary_category_only=False,
    ).drop(columns=GEOMETRY_COLUMN)

    assert (primary_only.sum(axis=1) == 1).all(), "Not all rows have exactly one primary category."
    assert (
        primary_only.sum().sum() < all_categories.sum().sum()
    ), "Primary only has more categories than all categories."


def test_old_version(
    wide_form_working_directory: Path,
) -> None:
    """Test if oldest supported version is working."""
    convert_bounding_box_to_wide_form_geodataframe_for_all_types(
        bbox=bbox(),
        release=MINIMAL_SUPPORTED_RELEASE_VERSION,
        working_directory=wide_form_working_directory,
        verbosity_mode="verbose",
        ignore_cache=False,
        include_all_possible_columns=False,
    )


def test_generate_result_file_name_order(
    test_release_version: str,
) -> None:
    """Test if result file name is generated correctly."""
    theme_type_pairs = [
        ("base", "water"),
        ("base", "land_cover"),
        ("base", "infrastructure"),
        ("places", "place"),
    ]
    pyarrow_filters = [None, None, [("class", "==", "bridge")], [("confidence", ">", 0.95)]]
    hierarchy_depths = [None, 1, 2, 3]
    result = _generate_result_file_path(
        release=test_release_version,
        theme_type_pairs=theme_type_pairs,
        geometry_filter=box(*bbox()),
        pyarrow_filters=pyarrow_filters,
        hierarchy_depth=hierarchy_depths,
        include_all_possible_columns=False,
        sort_result=True,
    )

    reverse_order_result = _generate_result_file_path(
        release=test_release_version,
        theme_type_pairs=theme_type_pairs[::-1],
        geometry_filter=box(*bbox()),
        pyarrow_filters=pyarrow_filters[::-1],
        hierarchy_depth=hierarchy_depths[::-1],
        include_all_possible_columns=False,
        sort_result=True,
    )

    assert result == reverse_order_result
