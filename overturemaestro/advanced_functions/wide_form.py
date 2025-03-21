"""Functions for retrieving Overture Maps features in a wide form."""

import json
import tempfile
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, overload

import numpy as np
import pandas as pd
from fsspec.implementations.http import HTTPFileSystem
from pooch import file_hash, retrieve
from pooch import get_logger as get_pooch_logger
from requests import HTTPError
from rich import print as rprint
from shapely.geometry.base import BaseGeometry

from overturemaestro._constants import (
    GEOMETRY_COLUMN,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from overturemaestro._duckdb import _set_up_duckdb_connection, _sql_escape
from overturemaestro._exceptions import (
    HierarchyDepthOutOfBoundsWarning,
    NegativeHierarchyDepthError,
)
from overturemaestro._geometry_sorting import sort_geoparquet_file_by_geometry
from overturemaestro._rich_progress import VERBOSITY_MODE, TrackProgressBar, TrackProgressSpinner
from overturemaestro.cache import (
    _get_global_wide_form_release_cache_directory,
    _get_local_wide_form_release_cache_directory,
)
from overturemaestro.data_downloader import (
    PYARROW_FILTER,
    _generate_geometry_hash,
    download_data_for_multiple_types,
)
from overturemaestro.elapsed_time_decorator import show_total_elapsed_time_decorator
from overturemaestro.release_index import (
    LFS_DIRECTORY_URL,
    _check_release_version,
    get_newest_release_version,
)

__all__ = [
    "convert_geometry_to_wide_form_parquet_for_all_types",
    "convert_geometry_to_wide_form_parquet_for_multiple_types",
    "get_all_possible_column_names",
]

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame
    from pyarrow.compute import Expression


def _check_depth_for_wide_form(
    theme: str, type: str, hierarchy_columns: list[str], depth: Optional[int] = None, **kwargs: Any
) -> int:
    depth = depth if depth is not None else len(hierarchy_columns)

    if depth < 0:
        raise NegativeHierarchyDepthError("Hierarchy depth cannot be negative")
    elif depth > len(hierarchy_columns):
        warnings.warn(
            (
                f"Provided hierarchy depth is out of bounds"
                f" (valid for {theme}/{type}: 0 - {len(hierarchy_columns)})."
                f" Value will be clipped to {len(hierarchy_columns)}."
            ),
            HierarchyDepthOutOfBoundsWarning,
            stacklevel=0,
        )
        depth = len(hierarchy_columns)

    return depth


def _transform_to_wide_form(
    theme: str,
    type: str,
    release_version: str,
    parquet_file: Path,
    output_path: Path,
    include_all_possible_columns: bool,
    hierarchy_columns: list[str],
    working_directory: Union[str, Path],
    verbosity_mode: VERBOSITY_MODE,
    **kwargs: Any,
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

    if include_all_possible_columns:
        wide_column_definitions = _get_wide_column_definitions(
            theme=theme,
            type=type,
            release_version=release_version,
            hierarchy_columns=hierarchy_columns,
            verbosity_mode="silent",
        ).to_dict(orient="records")
    else:
        joined_hierarchy_columns = ",".join(hierarchy_columns)

        available_colums_sql_query = f"""
        SELECT DISTINCT
            {joined_hierarchy_columns},
            concat_ws('|', '{theme}', '{type}', {joined_hierarchy_columns}) as column_name
        FROM read_parquet(
            '{parquet_file}',
            hive_partitioning=false
        )
        ORDER BY column_name
        """

        wide_column_definitions = (
            connection.sql(available_colums_sql_query).fetchdf().to_dict(orient="records")
        )

    case_clauses = []

    for wide_column_definition in wide_column_definitions:
        column_name = wide_column_definition["column_name"] or f"{theme}|{type}"
        conditions = []
        for condition_column in hierarchy_columns:
            if wide_column_definition[condition_column] is None:
                conditions.append(f"{condition_column} IS NULL")
            else:
                escaped_value = _sql_escape(wide_column_definition[condition_column])
                conditions.append(f"{condition_column} = '{escaped_value}'")

        joined_conditions = " AND ".join(conditions)
        case_clauses.append(f'COALESCE(({joined_conditions}), False) AS "{column_name}"')

    query = f"""
    COPY (
        SELECT
            id,
            geometry,
            {", ".join(case_clauses)}
        FROM read_parquet(
            '{parquet_file}',
            hive_partitioning=false
        )
    ) TO '{output_path}' (
        FORMAT 'parquet',
        ROW_GROUP_SIZE {PARQUET_ROW_GROUP_SIZE},
        COMPRESSION {PARQUET_COMPRESSION},
        COMPRESSION_LEVEL {PARQUET_COMPRESSION_LEVEL},
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)
    connection.close()

    return output_path


def _transform_to_wide_form_without_hierarchy(
    theme: str,
    type: str,
    parquet_file: Path,
    output_path: Path,
    working_directory: Union[str, Path],
    **kwargs: Any,
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

    query = f"""
    COPY (
        SELECT
            id,
            geometry,
            true as "{theme}|{type}"
        FROM read_parquet(
            '{parquet_file}',
            hive_partitioning=false
        )
    ) TO '{output_path}' (
        FORMAT 'parquet',
        ROW_GROUP_SIZE {PARQUET_ROW_GROUP_SIZE},
        COMPRESSION {PARQUET_COMPRESSION},
        COMPRESSION_LEVEL {PARQUET_COMPRESSION_LEVEL},
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)
    connection.close()

    return output_path


def _prepare_download_parameters_for_poi(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    hierachy_columns: list[str],
    pyarrow_filter: Optional["Expression"] = None,
    **kwargs: Any,
) -> tuple[list[str], Optional["Expression"]]:
    # TODO: swap to dedicated function?
    import pyarrow.compute as pc

    category_not_null_filter = pc.invert(pc.field("categories").is_null())
    minimal_confidence_filter = pc.field("confidence") >= pc.scalar(
        kwargs.get("places_minimal_confidence", 0.75)
    )
    if pyarrow_filter is not None:
        pyarrow_filter = pyarrow_filter & category_not_null_filter & minimal_confidence_filter
    else:
        pyarrow_filter = category_not_null_filter & minimal_confidence_filter

    return (["categories"] if hierachy_columns else [], pyarrow_filter)


def _transform_poi_to_wide_form(
    theme: str,
    type: str,
    release_version: str,
    parquet_file: Path,
    output_path: Path,
    include_all_possible_columns: bool,
    hierarchy_columns: list[str],
    working_directory: Union[str, Path],
    verbosity_mode: VERBOSITY_MODE,
    **kwargs: Any,
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

    primary_category_only = kwargs.get("places_use_primary_category_only", False)

    primary_category_name = "primary"
    alternate_category_name = "alternate"
    if release_version < "2024-07-22.0":
        primary_category_name = "main"

    wide_column_definitions = _get_wide_column_definitions_for_poi(
        theme=theme,
        type=type,
        release_version=release_version,
        hierarchy_columns=hierarchy_columns,
        verbosity_mode="silent",
    )

    if not include_all_possible_columns:
        if primary_category_only:
            available_colums_sql_query = f"""
            SELECT DISTINCT
                categories.{primary_category_name} as category
            FROM read_parquet(
                '{parquet_file}',
                hive_partitioning=false
            )
            """
        else:
            available_colums_sql_query = f"""
            SELECT DISTINCT
                categories.{primary_category_name} as category
            FROM read_parquet(
                '{parquet_file}',
                hive_partitioning=false
            )
            UNION
            SELECT DISTINCT
                UNNEST(categories.{alternate_category_name}) as category
            FROM read_parquet(
                '{parquet_file}',
                hive_partitioning=false
            )
            """

        existing_category_names = (
            connection.sql(available_colums_sql_query).fetchdf()["category"].sort_values()
        )
        wide_column_definitions = wide_column_definitions[
            wide_column_definitions["category"].isin(existing_category_names)
        ]

    case_clauses = []

    for column_name, categories_list in (
        wide_column_definitions.groupby("column_name")["category"].agg(list).to_dict().items()
    ):
        conditions = []
        for category_name in categories_list:
            escaped_value = _sql_escape(category_name)
            conditions.append(f"categories.{primary_category_name} = '{escaped_value}'")

            if not primary_category_only:
                conditions.append(f"'{escaped_value}' IN categories.{alternate_category_name}")

        joined_conditions = " OR ".join(conditions)
        case_clauses.append(f'COALESCE(({joined_conditions}), False) AS "{column_name}"')

    query = f"""
    COPY (
        SELECT
            id,
            geometry,
            {", ".join(case_clauses)}
        FROM read_parquet(
            '{parquet_file}',
            hive_partitioning=false
        )
    ) TO '{output_path}' (
        FORMAT 'parquet',
        ROW_GROUP_SIZE {PARQUET_ROW_GROUP_SIZE},
        COMPRESSION {PARQUET_COMPRESSION},
        COMPRESSION_LEVEL {PARQUET_COMPRESSION_LEVEL},
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)
    connection.close()

    return output_path


def _get_all_possible_column_names(
    theme: str, type: str, release_version: str, hierarchy_columns: list[str], **kwargs: Any
) -> "DataFrame":
    import duckdb

    connection = duckdb.connect()

    for extension in ("spatial", "httpfs"):
        connection.install_extension(extension)
        connection.load_extension(extension)

    dataset_path = (
        f"s3://overturemaps-us-west-2/release/{release_version}/theme={theme}/type={type}/*"
    )
    joined_hierarchy_columns = ",".join(hierarchy_columns)
    df = duckdb.sql(
        f"""
        SELECT DISTINCT
            {joined_hierarchy_columns}
        FROM read_parquet(
            '{dataset_path}',
            hive_partitioning=false
        )
        ORDER BY ALL
        """
    ).to_df()

    return df


def _get_all_possible_column_names_for_poi(
    theme: str, type: str, release_version: str, hierarchy_columns: list[str], **kwargs: Any
) -> "DataFrame":
    import duckdb

    connection = duckdb.connect()

    for extension in ("spatial", "httpfs"):
        connection.install_extension(extension)
        connection.load_extension(extension)

    dataset_path = (
        f"s3://overturemaps-us-west-2/release/{release_version}/theme={theme}/type={type}/*"
    )

    primary_category_name = "primary"
    alternate_category_name = "alternate"
    if release_version < "2024-07-22.0":
        primary_category_name = "main"

    df = (
        duckdb.sql(
            f"""
        SELECT DISTINCT
            categories.{primary_category_name} as column_name
        FROM read_parquet(
            '{dataset_path}',
            hive_partitioning=false
        )
        UNION
        SELECT DISTINCT
            UNNEST(categories.{alternate_category_name}) as column_name
        FROM read_parquet(
            '{dataset_path}',
            hive_partitioning=false
        )
        """
        )
        .to_df()
        .dropna()
        .sort_values(by="column_name")
        .reset_index(drop=True)
    )

    hierarchy_data = pd.read_csv(
        # List of all possible places values on CC-BY-SA 4.0 license
        # provided by Overture Maps Foundation
        "https://raw.githubusercontent.com/OvertureMaps/schema/refs/heads/main/docs/schema/concepts/by-theme/places/overture_categories.csv",
        sep=";",
        names=["category", "hierarchy"],
        skiprows=1,
    )
    hierarchy_split = (
        hierarchy_data["hierarchy"].str.strip().str[1:-1].str.split(",").apply(pd.Series)
    )

    hierarchy_split.columns = [str(i + 1) for i in range(hierarchy_split.shape[1])]

    # Concatenate the original dataframe with the new hierarchy columns
    data_split = pd.concat([hierarchy_data[["category"]], hierarchy_split], axis=1)

    rows = data_split.to_dict(orient="records")

    for category_name in df["column_name"]:
        if category_name not in data_split["category"].values:
            rows.append({"category": category_name, "1": category_name})

    return pd.DataFrame(rows).replace({np.nan: None})


def _get_wide_column_definitions(
    theme: str,
    type: str,
    release_version: str,
    hierarchy_columns: list[str],
    verbosity_mode: VERBOSITY_MODE = "transient",
    **kwargs: Any,
) -> "DataFrame":
    if not hierarchy_columns:
        return pd.DataFrame(dict(column_name=[f"{theme}|{type}"]))

    all_columns_names = load_wide_form_all_column_names_release_index(
        theme=theme, type=type, release=release_version, verbosity_mode=verbosity_mode
    )
    columns_not_in_hierarchy = [c for c in all_columns_names.columns if c not in hierarchy_columns]
    if columns_not_in_hierarchy:
        all_columns_names = all_columns_names.drop(
            columns=columns_not_in_hierarchy
        ).drop_duplicates()
    combine_columns_fn = partial(
        _combine_columns, theme=theme, type=type, hierarchy_columns=hierarchy_columns
    )
    all_columns_names["column_name"] = all_columns_names.apply(combine_columns_fn, axis=1)
    return all_columns_names.sort_values(by="column_name")


def _combine_columns(row: pd.Series, theme: str, type: str, hierarchy_columns: list[str]) -> str:
    result = f"{theme}|{type}"
    for column_name in hierarchy_columns:
        value = row[column_name]
        if value is None:
            break
        result += f"|{value}"
    return result


def _get_wide_column_definitions_for_poi(
    theme: str,
    type: str,
    release_version: str,
    hierarchy_columns: list[str],
    verbosity_mode: VERBOSITY_MODE = "transient",
    **kwargs: Any,
) -> "DataFrame":
    if not hierarchy_columns:
        return pd.DataFrame(dict(column_name=[f"{theme}|{type}"]))

    all_columns_names = load_wide_form_all_column_names_release_index(
        theme=theme, type=type, release=release_version, verbosity_mode=verbosity_mode
    )
    columns_not_in_hierarchy = [
        c for c in all_columns_names.columns if c != "category" and c not in hierarchy_columns
    ]
    if columns_not_in_hierarchy:
        all_columns_names = all_columns_names.drop(
            columns=columns_not_in_hierarchy
        ).drop_duplicates()
    combine_columns_fn = partial(
        _combine_columns, theme=theme, type=type, hierarchy_columns=hierarchy_columns
    )
    all_columns_names["column_name"] = all_columns_names.apply(combine_columns_fn, axis=1)
    return all_columns_names.sort_values(by="column_name")


class DownloadParametersPreparationCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        geometry_filter: BaseGeometry,
        hierachy_columns: list[str],
        pyarrow_filter: Optional["Expression"],
        **kwargs: Any,
    ) -> tuple[list[str], Optional["Expression"]]: ...


class DepthCheckCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        hierarchy_columns: list[str],
        depth: Optional[int],
        **kwargs: Any,
    ) -> int: ...


class DataTransformationCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        release_version: str,
        parquet_file: Path,
        output_path: Path,
        include_all_possible_columns: bool,
        hierarchy_columns: list[str],
        working_directory: Union[str, Path],
        verbosity_mode: VERBOSITY_MODE,
        **kwargs: Any,
    ) -> Path: ...


class GetAllPossibleColumnNamesCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        release_version: str,
        hierarchy_columns: list[str],
        **kwargs: Any,
    ) -> "DataFrame": ...


class GetWideColumnDefinitionsCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        release_version: str,
        hierarchy_columns: list[str],
        verbosity_mode: VERBOSITY_MODE,
        **kwargs: Any,
    ) -> "DataFrame": ...


@dataclass
class WideFormDefinition:
    """Helper object for theme type wide form logic definition."""

    hierachy_columns: list[str]
    download_parameters_preparation_function: Optional[DownloadParametersPreparationCallable] = None
    depth_check_function: DepthCheckCallable = _check_depth_for_wide_form
    data_transform_function: DataTransformationCallable = _transform_to_wide_form
    get_all_possible_column_names_function: GetAllPossibleColumnNamesCallable = (
        _get_all_possible_column_names
    )
    get_wide_column_definitions_function: GetWideColumnDefinitionsCallable = (
        _get_wide_column_definitions
    )


THEME_TYPE_CLASSIFICATION: dict[tuple[str, str], WideFormDefinition] = {
    ("base", "infrastructure"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("base", "land"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("base", "land_cover"): WideFormDefinition(hierachy_columns=["subtype"]),
    ("base", "land_use"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("base", "water"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("transportation", "segment"): WideFormDefinition(
        hierachy_columns=["subtype", "class", "subclass"]
    ),
    ("places", "place"): WideFormDefinition(
        hierachy_columns=["1", "2", "3", "4", "5", "6"],  # 6 IS A MAX DEPTH
        download_parameters_preparation_function=_prepare_download_parameters_for_poi,
        data_transform_function=_transform_poi_to_wide_form,
        get_all_possible_column_names_function=_get_all_possible_column_names_for_poi,
        get_wide_column_definitions_function=_get_wide_column_definitions_for_poi,
    ),
    ("buildings", "building"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
}


def get_theme_type_classification(release: str) -> dict[tuple[str, str], WideFormDefinition]:
    classification = THEME_TYPE_CLASSIFICATION.copy()
    # start from the newest release

    if release < "2024-08-20.0":
        classification[("transportation", "segment")] = WideFormDefinition(
            hierachy_columns=["subtype", "class"]
        )

    if release < "2024-05-16-beta.0":
        classification[("buildings", "building")] = WideFormDefinition(hierachy_columns=["class"])
        classification.pop(("base", "land_cover"))

    return classification


@overload
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


@show_total_elapsed_time_decorator
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path:
    """
    Get GeoParquet file for a given geometry in a wide format for multiple types.

    Automatically downloads Overture Maps dataset for a given release and theme/type pairs
    in a concurrent manner and returns a single file as a result with multiple columns based
    on dataset schema.

    Args:
        theme_type_pairs (list[tuple[str, str]]): Pairs of themes and types of the dataset.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        include_all_possible_columns (bool, optional): Whether to have always the same list of
            columns in the resulting file. This ensures that always the same set of columns is
            returned for a given release for different regions. This also means, that some columns
            might be all filled with a False value. Defaults to True.
        hierarchy_depth (Optional[Union[int, list[Optional[int]]]], optional): Depth used to
            calculate how many hierarchy columns should be used to generate the wide form of
            the data. Can be a single integer or a list of integers. If None, will use all
            available columns. Defaults to None.
        pyarrow_filters (Optional[list[Optional[PYARROW_FILTER]]], optional): A list of pyarrow
            expressions used to filter specific theme type pair. Must be the same length as the list
            of theme type pairs. Defaults to None.
        result_file_path (Union[str, Path], optional): Where to save
            the geoparquet file. If not provided, will be generated based on hashes
            from filters. Defaults to None.
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.
        sort_result (bool, optional): Whether to sort the result by geometry or not.
            Defaults to True.
        places_use_primary_category_only (bool, optional): Whether to use only primary category
            from the places dataset. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    if pyarrow_filters is not None and len(theme_type_pairs) != len(pyarrow_filters):
        raise ValueError("Pyarrow filters length doesn't match length of theme type pairs.")

    if isinstance(hierarchy_depth, list) and len(theme_type_pairs) != len(hierarchy_depth):
        raise ValueError("Hierarchy depth list length doesn't match length of theme type pairs.")

    if not release:
        release = get_newest_release_version()

    pyarrow_filters_list = []
    for idx in range(len(theme_type_pairs)):
        _pyarrow_filter = pyarrow_filters[idx] if pyarrow_filters else None

        if _pyarrow_filter is not None:
            from pyarrow.parquet import filters_to_expression

            _pyarrow_filter = filters_to_expression(_pyarrow_filter)

        pyarrow_filters_list.append(_pyarrow_filter)

    if result_file_path is None:
        result_file_path = working_directory / _generate_result_file_path(
            release=release,
            theme_type_pairs=theme_type_pairs,
            geometry_filter=geometry_filter,
            include_all_possible_columns=include_all_possible_columns,
            hierarchy_depth=hierarchy_depth,
            pyarrow_filters=pyarrow_filters_list,
            sort_result=sort_result,
            places_use_primary_category_only=places_use_primary_category_only,
            places_minimal_confidence=places_minimal_confidence,
        )

    result_file_path = Path(result_file_path)

    if not result_file_path.exists() or ignore_cache:
        result_file_path.parent.mkdir(exist_ok=True, parents=True)

        prepared_download_parameters = _prepare_download_parameters_for_all_theme_type_pairs(
            release=release,
            theme_type_pairs=theme_type_pairs,
            geometry_filter=geometry_filter,
            hierarchy_depth=hierarchy_depth,
            pyarrow_filters=pyarrow_filters_list,
            verbosity_mode=verbosity_mode,
            places_minimal_confidence=places_minimal_confidence,
        )

        hierachy_columns_list, columns_to_download_list, pyarrow_filter_list = zip(
            *prepared_download_parameters
        )

        downloaded_parquet_files = download_data_for_multiple_types(
            release=release,
            theme_type_pairs=theme_type_pairs,
            geometry_filter=geometry_filter,
            pyarrow_filters=pyarrow_filter_list,
            columns_to_download=[
                ["id", GEOMETRY_COLUMN, *columns_to_download]
                for columns_to_download in columns_to_download_list
            ],
            ignore_cache=ignore_cache,
            working_directory=working_directory,
            verbosity_mode=verbosity_mode,
            max_workers=max_workers,
            sort_result=False,
        )

        with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)

            merged_parquet_path = (
                tmp_dir_path / f"{result_file_path.stem}_merged{result_file_path.suffix}"
                if sort_result
                else result_file_path
            )

            transformed_wide_form_directory_output = tmp_dir_path / "wide_form_files"
            transformed_wide_form_directory_output.mkdir(parents=True, exist_ok=True)

            with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
                for (
                    (theme_value, type_value),
                    hierachy_columns,
                    downloaded_parquet_file,
                ) in progress.track(
                    zip(theme_type_pairs, hierachy_columns_list, downloaded_parquet_files),
                    total=len(theme_type_pairs),
                    description="Transforming data into wide form",
                ):
                    wide_form_definition = get_theme_type_classification(release=release)[
                        (theme_value, type_value)
                    ]

                    output_path = (
                        transformed_wide_form_directory_output
                        / f"{theme_value}_{type_value}.parquet"
                    )
                    if len(theme_type_pairs) == 1:
                        output_path = merged_parquet_path

                    if not hierachy_columns:
                        _transform_to_wide_form_without_hierarchy(
                            theme=theme_value,
                            type=type_value,
                            parquet_file=downloaded_parquet_file,
                            output_path=output_path,
                            working_directory=tmp_dir_path,
                        )
                    else:
                        wide_form_definition.data_transform_function(
                            theme=theme_value,
                            type=type_value,
                            release_version=release,
                            parquet_file=downloaded_parquet_file,
                            output_path=output_path,
                            include_all_possible_columns=include_all_possible_columns,
                            hierarchy_columns=hierachy_columns,
                            working_directory=tmp_dir_path,
                            verbosity_mode=verbosity_mode,
                            places_use_primary_category_only=places_use_primary_category_only,
                        )

            if len(theme_type_pairs) > 1:
                with TrackProgressSpinner(
                    "Joining results to a single file", verbosity_mode=verbosity_mode
                ):
                    _combine_multiple_wide_form_files(
                        theme_type_pairs=theme_type_pairs,
                        transformed_wide_form_directory=transformed_wide_form_directory_output,
                        output_path=merged_parquet_path,
                        working_directory=tmp_dir_path,
                    )

            if sort_result:
                with TrackProgressSpinner(
                    "Sorting result file by geometry", verbosity_mode=verbosity_mode
                ):
                    sort_geoparquet_file_by_geometry(
                        input_file_path=merged_parquet_path,
                        output_file_path=result_file_path,
                        working_directory=working_directory,
                        sort_extent=geometry_filter.bounds,
                    )

    return result_file_path


@overload
def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path:
    """
    Get GeoParquet file for a given geometry in a wide format for all types.

    Automatically downloads Overture Maps dataset for a given release and all available theme/types
    in a concurrent manner and returns a single file as a result with multiple columns based
    on dataset schema.

    Args:
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        include_all_possible_columns (bool, optional): Whether to have always the same list of
            columns in the resulting file. This ensures that always the same set of columns is
            returned for a given release for different regions. This also means, that some columns
            might be all filled with a False value. Defaults to True.
        hierarchy_depth (Optional[Union[int, list[Optional[int]]]], optional): Depth used to
            calculate how many hierarchy columns should be used to generate the wide form of
            the data. Can be a single integer or a list of integers. If None, will use all
            available columns. Defaults to None.
        pyarrow_filters (Optional[list[Optional[PYARROW_FILTER]]], optional): A list of pyarrow
            expressions used to filter specific theme type pair. Must be the same length as the list
            of theme type pairs. Defaults to None.
        result_file_path (Union[str, Path], optional): Where to save
            the geoparquet file. If not provided, will be generated based on hashes
            from filters. Defaults to None.
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.
        sort_result (bool, optional): Whether to sort the result by geometry or not.
            Defaults to True.
        places_use_primary_category_only (bool, optional): Whether to use only primary category
            from the places dataset. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    if not release:
        release = get_newest_release_version()

    return convert_geometry_to_wide_form_parquet_for_multiple_types(
        theme_type_pairs=list(get_theme_type_classification(release=release).keys()),
        geometry_filter=geometry_filter,
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )


def get_all_possible_column_names(
    release: Optional[str] = None,
    theme: Optional[str] = None,
    type: Optional[str] = None,
    hierarchy_depth: Optional[int] = None,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> list[str]:
    """
    Get a list of all possible columns.

    Args:
        release (Optional[str], optional): Select the release for which the list should be returned.
            If None, will return for the newest release. Defaults to None.
        theme (Optional[str], optional): Select the theme of the dataset. If None, will return
            the list for all datasets. Defaults to None.
        type (Optional[str], optional): Select the type of the dataset. If None, will return
            the list for all datasets. Defaults to None.
        hierarchy_depth (Optional[int]): Depth used to calculate how many hierarchy columns should
            be used to generate the wide form of the data. If None, will use all available columns.
            Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        list[str]: List of column names.
    """
    if (theme is None and type is not None) or (theme is not None and type is None):
        raise ValueError("Theme and type both have to be present or None.")

    if not release:
        release = get_newest_release_version()

    _check_release_version(release)

    if theme and type:
        definitions = {(theme, type): get_theme_type_classification(release=release)[(theme, type)]}
    else:
        definitions = get_theme_type_classification(release=release)

    columns = []
    for (theme_value, type_value), wide_form_definition in definitions.items():
        depth = wide_form_definition.depth_check_function(
            theme=theme_value,
            type=type_value,
            hierarchy_columns=wide_form_definition.hierachy_columns,
            depth=hierarchy_depth,
        )
        hierachy_columns = wide_form_definition.hierachy_columns[:depth]

        df = wide_form_definition.get_wide_column_definitions_function(
            theme=theme_value,
            type=type_value,
            release_version=release,
            hierarchy_columns=hierachy_columns,
            verbosity_mode=verbosity_mode,
        )
        columns.extend(df["column_name"].unique())

    return sorted(columns)


def _generate_result_file_path(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    include_all_possible_columns: bool,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]],
    pyarrow_filters: Optional[list[Union["Expression", None]]],
    sort_result: bool,
    **kwargs: Any,
) -> Path:
    import hashlib

    directory = Path(release)

    params_per_theme_type_pair = {
        theme_type_pair: (
            hierarchy_depth[idx] if isinstance(hierarchy_depth, list) else hierarchy_depth,
            pyarrow_filters[idx] if pyarrow_filters else None,
        )
        for idx, theme_type_pair in enumerate(theme_type_pairs)
    }

    sorted_theme_type_pairs = sorted(theme_type_pairs)

    if len(theme_type_pairs) == 1:
        theme_value, type_value = theme_type_pairs[0]
        directory = directory / f"theme={theme_value}" / f"type={type_value}"
    else:
        h = hashlib.new("sha256")
        h.update(str(sorted_theme_type_pairs).encode())
        directory = directory / h.hexdigest()

    clipping_geometry_hash_part = _generate_geometry_hash(geometry_filter)

    pyarrow_filter_hash_part = "nofilter"
    if pyarrow_filters is not None:
        h = hashlib.new("sha256")
        for theme_type_pair in sorted_theme_type_pairs:
            h.update(str(params_per_theme_type_pair[theme_type_pair][1]).encode())
        pyarrow_filter_hash_part = h.hexdigest()[:8]

    hierarchy_hash_part = ""
    if hierarchy_depth is not None:
        h = hashlib.new("sha256")
        for theme_type_pair in sorted_theme_type_pairs:
            h.update(str(params_per_theme_type_pair[theme_type_pair][0]).encode())
        hierarchy_hash_part = f"_h{h.hexdigest()[:8]}"

    include_all_columns_hash_part = ""
    if not include_all_possible_columns:
        include_all_columns_hash_part = "_pruned"

    h = hashlib.new("sha256")
    h.update(str(sorted(kwargs.items())).encode())
    kwargs_hash_part = h.hexdigest()[:8]

    sort_result_part = "_sorted" if sort_result else ""

    return directory / (
        f"{clipping_geometry_hash_part}_{pyarrow_filter_hash_part}_{kwargs_hash_part}"
        f"_wide_form{hierarchy_hash_part}{include_all_columns_hash_part}{sort_result_part}.parquet"
    )


def _prepare_download_parameters_for_all_theme_type_pairs(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Union["Expression", None]]] = None,
    verbosity_mode: VERBOSITY_MODE = "transient",
    **kwargs: Any,
) -> list[tuple[list[str], list[str], Optional[PYARROW_FILTER]]]:
    prepared_parameters = []
    with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
        for idx, (theme_value, type_value) in progress.track(
            enumerate(theme_type_pairs),
            total=len(theme_type_pairs),
            description="Preparing download parameters",
        ):
            single_pyarrow_filter = pyarrow_filters[idx] if pyarrow_filters else None
            single_hierarchy_depth = (
                hierarchy_depth[idx] if isinstance(hierarchy_depth, list) else hierarchy_depth
            )
            wide_form_definition = get_theme_type_classification(release=release)[
                (theme_value, type_value)
            ]

            depth = wide_form_definition.depth_check_function(
                theme=theme_value,
                type=type_value,
                hierarchy_columns=wide_form_definition.hierachy_columns,
                depth=single_hierarchy_depth,
                **kwargs,
            )
            hierachy_columns = wide_form_definition.hierachy_columns[:depth]
            columns_to_download = hierachy_columns

            if wide_form_definition.download_parameters_preparation_function is not None:
                columns_to_download, single_pyarrow_filter = (
                    wide_form_definition.download_parameters_preparation_function(
                        theme=theme_value,
                        type=type_value,
                        geometry_filter=geometry_filter,
                        hierachy_columns=hierachy_columns,
                        pyarrow_filter=single_pyarrow_filter,
                        **kwargs,
                    )
                )

            prepared_parameters.append(
                (hierachy_columns, columns_to_download, single_pyarrow_filter)
            )

    return prepared_parameters


def _combine_multiple_wide_form_files(
    theme_type_pairs: list[tuple[str, str]],
    transformed_wide_form_directory: Path,
    output_path: Path,
    working_directory: Union[str, Path],
) -> None:
    import pyarrow.parquet as pq

    all_select_columns = []
    subqueries = []

    for theme_value, type_value in theme_type_pairs:
        current_parquet_file = (
            transformed_wide_form_directory / f"{theme_value}_{type_value}.parquet"
        )
        available_columns = [
            col
            for col in pq.read_metadata(current_parquet_file).schema.names
            if col not in ("id", GEOMETRY_COLUMN)
        ]

        combined_columns_select = ", ".join(f'"{column}"' for column in available_columns)

        select_subquery = f"""
        SELECT id, geometry, {combined_columns_select}
        FROM read_parquet('{current_parquet_file}', hive_partitioning=false)
        """

        subqueries.append(select_subquery)
        all_select_columns.extend(available_columns)

    joined_all_select_columns = ", ".join(
        f'COALESCE("{col}", False) AS "{col}"' for col in sorted(all_select_columns)
    )
    joined_subqueries = " UNION ALL BY NAME ".join(subqueries)

    connection = _set_up_duckdb_connection(working_directory)
    query = f"""
    COPY (
        SELECT
            id,
            geometry,
            {joined_all_select_columns}
        FROM (
            {joined_subqueries}
        )
    ) TO '{output_path}' (
        FORMAT 'parquet',
        ROW_GROUP_SIZE {PARQUET_ROW_GROUP_SIZE},
        COMPRESSION {PARQUET_COMPRESSION},
        COMPRESSION_LEVEL {PARQUET_COMPRESSION_LEVEL},
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)
    connection.close()


@overload
def load_wide_form_all_column_names_release_index(
    theme: str,
    type: str,
    *,
    remote_index: bool = False,
    skip_index_download: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> "DataFrame": ...


@overload
def load_wide_form_all_column_names_release_index(
    theme: str,
    type: str,
    release: str,
    *,
    remote_index: bool = False,
    skip_index_download: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> "DataFrame": ...


@overload
def load_wide_form_all_column_names_release_index(
    theme: str,
    type: str,
    release: Optional[str] = None,
    *,
    remote_index: bool = False,
    skip_index_download: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> "DataFrame": ...


def load_wide_form_all_column_names_release_index(
    theme: str,
    type: str,
    release: Optional[str] = None,
    *,
    remote_index: bool = False,
    skip_index_download: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> "DataFrame":
    """
    Load release index as a GeoDataFrame.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        geometry_filter (Optional[BaseGeometry], optional): Geometry to pre-filter resulting rows.
            Defaults to None.
        remote_index (bool, optional): Avoid downloading the index and stream it from remote source.
            Defaults to False.
        skip_index_download (bool, optional): Avoid downloading the index if doesn't exist locally
            and generate it instead. Defaults to False.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        gpd.GeoDataFrame: Index with bounding boxes for each row group for each parquet file.
    """
    if not release:
        release = get_newest_release_version()

    _check_release_version(release)
    cache_directory = _get_global_wide_form_release_cache_directory(release)
    index_file_path: Union[Path, str] = cache_directory / _get_wide_form_release_index_file_name(
        theme, type
    )

    file_exists = Path(index_file_path).exists()
    filesystem = None

    if not file_exists:
        if remote_index:
            filesystem = HTTPFileSystem()
            local_cache_path = _get_local_wide_form_release_cache_directory(release)
            index_file_path = LFS_DIRECTORY_URL + str(
                local_cache_path / _get_wide_form_release_index_file_name(theme, type)
            )
        elif skip_index_download:
            # Generate the index and skip download
            generate_wide_form_all_column_names_release_index(
                release,
                verbosity_mode=verbosity_mode,
            )
        else:
            # Try to download the index or generate it if cannot be downloaded
            download_existing_wide_form_all_column_names_release_index(
                release,
                verbosity_mode=verbosity_mode,
            ) or generate_wide_form_all_column_names_release_index(
                release,
                verbosity_mode=verbosity_mode,
            )

    return pd.read_parquet(index_file_path, filesystem=filesystem)


@overload
def download_existing_wide_form_all_column_names_release_index(
    *,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool: ...


@overload
def download_existing_wide_form_all_column_names_release_index(
    release: str, *, verbosity_mode: VERBOSITY_MODE = "transient"
) -> bool: ...


def download_existing_wide_form_all_column_names_release_index(
    release: Optional[str] = None,
    *,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Download a pregenerated wide form all column names index for an Overture Maps dataset release.

    Args:
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been downloaded or not.
    """
    return _download_existing_wide_form_all_column_names_release_index(
        release=release, verbosity_mode=verbosity_mode
    )


def _download_existing_wide_form_all_column_names_release_index(
    release: Optional[str] = None,
    theme: Optional[str] = None,
    type: Optional[str] = None,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Download a pregenerated index for an Overture Maps dataset release.

    Args:
        theme (Optional[str], optional): Specify a theme to be downloaded. Defaults to None.
        type (Optional[str], optional): Specify a type to be downloaded. Defaults to None.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been downloaded or not.
    """
    if (theme is None and type is not None) or (theme is not None and type is None):
        raise ValueError("Theme and type both have to be present or None.")

    if not release:
        release = get_newest_release_version()

    _check_release_version(release)

    global_cache_directory = _get_global_wide_form_release_cache_directory(release)
    local_cache_directory = _get_local_wide_form_release_cache_directory(release)

    logger = get_pooch_logger()
    logger.setLevel("WARNING")

    try:
        index_content_file_name = "release_index_content.json"
        index_content_file_url = (
            LFS_DIRECTORY_URL + (local_cache_directory / index_content_file_name).as_posix()
        )
        retrieve(
            index_content_file_url,
            fname=index_content_file_name,
            path=global_cache_directory,
            progressbar=False,
            known_hash=None,
        )

        theme_type_tuples = json.loads(
            (global_cache_directory / index_content_file_name).read_text()
        )

        with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
            for theme_type_tuple in progress.track(
                theme_type_tuples,
                description="Downloading release indexes",
            ):
                theme_value = theme_type_tuple["theme"]
                type_value = theme_type_tuple["type"]
                sha_value = theme_type_tuple["sha"]
                file_name = _get_wide_form_release_index_file_name(theme_value, type_value)

                if theme_value == theme and type_value == type:
                    continue

                index_file_url = LFS_DIRECTORY_URL + (local_cache_directory / file_name).as_posix()
                retrieve(
                    index_file_url,
                    fname=file_name,
                    path=global_cache_directory,
                    progressbar=False,
                    known_hash=sha_value,
                )

    except HTTPError as ex:
        if ex.response.status_code == 404:
            return False

        raise

    return True


def generate_wide_form_all_column_names_release_index(
    release: str,
    ignore_cache: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Generate a wide form all column names index for an Overture Maps dataset release.

    Args:
        release (str): Release version.
        ignore_cache (bool, optional): Whether to ignore precalculated parquet files or not.
            Defaults to False.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been generated or not.
    """
    return _generate_wide_form_all_column_names_release_index(
        release=release, ignore_cache=ignore_cache, verbosity_mode=verbosity_mode
    )


def _generate_wide_form_all_column_names_release_index(
    release: str,
    index_location_path: Optional[Path] = None,
    ignore_cache: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Generate files with all possible wide form column names for an Overture Maps dataset release.

    Args:
        release (str): Release version.
        dataset_path (str, optional): Specify dataset path.
            Defaults to "overturemaps-us-west-2/release/".
        dataset_fs (Literal["local", "s3"], optional): Which filesystem use in PyArrow operations.
            Defaults to "s3".
        index_location_path (Path, optional): Specify index location path. If not, will generate to
            the global cache location. Defaults to None.
        ignore_cache (bool, optional): Whether to ignore precalculated parquet files or not.
            Defaults to False.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been generated or not.
    """
    _check_release_version(release)
    cache_directory = Path(
        index_location_path or _get_global_wide_form_release_cache_directory(release)
    )
    release_index_path = cache_directory / "release_index_content.json"

    if release_index_path.exists() and not ignore_cache:
        rprint("Cache exists. Skipping generation.")
        return False

    cache_directory.mkdir(exist_ok=True, parents=True)

    file_hashes = []

    with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
        for (theme_value, type_value), definition in progress.track(
            sorted(get_theme_type_classification(release=release).items()),
            description="Saving parquet indexes",
        ):
            file_name = _get_wide_form_release_index_file_name(theme_value, type_value)
            cache_file_path = cache_directory / file_name

            if cache_file_path.exists() and not ignore_cache:
                rprint(f"Cache for {theme_value}/{type_value} exists. Skipping generation.")
                file_hashes.append(
                    dict(theme=theme_value, type=type_value, sha=file_hash(str(cache_file_path)))
                )
                continue

            df = definition.get_all_possible_column_names_function(
                theme=theme_value,
                type=type_value,
                release_version=release,
                hierarchy_columns=definition.hierachy_columns,
            )

            df.to_parquet(cache_file_path)
            file_hashes.append(
                dict(theme=theme_value, type=type_value, sha=file_hash(str(cache_file_path)))
            )
            rprint(f"Saved index file {cache_file_path}")

    pd.DataFrame(file_hashes).to_json(release_index_path, orient="records")

    return True


def _get_wide_form_release_index_file_name(theme_value: str, type_value: str) -> str:
    return f"{theme_value}_{type_value}.parquet"
