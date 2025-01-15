"""Functions for retrieving Overture Maps features in a wide form."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Union, overload

from shapely.geometry.base import BaseGeometry

from overturemaestro._duckdb import _set_up_duckdb_connection, _sql_escape
from overturemaestro._exceptions import HierarchyDepthOutOfBoundsError
from overturemaestro._rich_progress import VERBOSITY_MODE, TrackProgressBar, TrackProgressSpinner
from overturemaestro.data_downloader import (
    PYARROW_FILTER,
    _generate_geometry_hash,
    download_data_for_multiple_types,
)
from overturemaestro.elapsed_time_decorator import show_total_elapsed_time_decorator
from overturemaestro.release_index import get_newest_release_version

if TYPE_CHECKING:  # pragma: no cover
    from pyarrow.compute import Expression


def _check_depth_for_wide_form(hierarchy_columns: list[str], depth: Optional[int] = None) -> int:
    depth = depth if depth is not None else len(hierarchy_columns)

    if depth < 1 or depth > len(hierarchy_columns):
        raise HierarchyDepthOutOfBoundsError(
            f"Provided hierarchy depth is out of bounds (valid: 1 - {len(hierarchy_columns)})"
        )

    return depth


def _transform_to_wide_form(
    theme: str,
    type: str,
    parquet_file: Path,
    output_path: Path,
    hierarchy_columns: list[str],
    working_directory: Union[str, Path],
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

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
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)

    return output_path


def _prepare_download_parameters_for_poi(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    hierachy_columns: list[str],
    pyarrow_filter: Optional["Expression"] = None,
) -> tuple[list[str], Optional["Expression"]]:
    # TODO: swap to dedicated function?
    # TODO: add option to change minimal confidence
    import pyarrow.compute as pc

    category_not_null_filter = pc.invert(pc.field("categories").is_null())
    minimal_confidence_filter = pc.field("confidence") >= pc.scalar(0.75)
    if pyarrow_filter is not None:
        pyarrow_filter = pyarrow_filter & category_not_null_filter & minimal_confidence_filter
    else:
        pyarrow_filter = category_not_null_filter & minimal_confidence_filter

    return (["categories"], pyarrow_filter)


def _transform_poi_to_wide_form(
    theme: str,
    type: str,
    parquet_file: Path,
    output_path: Path,
    hierarchy_columns: list[str],
    working_directory: Union[str, Path],
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

    primary_category_only = len(hierarchy_columns) == 1

    if primary_category_only:
        available_colums_sql_query = f"""
        SELECT DISTINCT
            categories.primary as column_name
        FROM read_parquet(
            '{parquet_file}',
            hive_partitioning=false
        )
        """
    else:
        available_colums_sql_query = f"""
        SELECT DISTINCT
            categories.primary as column_name
        FROM read_parquet(
            '{parquet_file}',
            hive_partitioning=false
        )
        UNION
        SELECT DISTINCT
            UNNEST(categories.alternate) as column_name
        FROM read_parquet(
            '{parquet_file}',
            hive_partitioning=false
        )
        """

    wide_column_definitions = (
        connection.sql(available_colums_sql_query).fetchdf()["column_name"].sort_values()
    )

    case_clauses = []

    for column_name in wide_column_definitions:
        conditions = []

        escaped_value = _sql_escape(column_name)
        conditions.append(f"categories.primary = '{escaped_value}'")

        if not primary_category_only:
            conditions.append(f"'{escaped_value}' IN categories.alternate")

        joined_conditions = " OR ".join(conditions)
        case_clauses.append(
            f'COALESCE(({joined_conditions}), False) AS "{theme}|{type}|{column_name}"'
        )

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
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)

    return output_path


class DownloadParametersPreparationCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        geometry_filter: BaseGeometry,
        hierachy_columns: list[str],
        pyarrow_filter: Optional["Expression"] = None,
    ) -> tuple[list[str], Optional["Expression"]]: ...


class DepthCheckCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, hierarchy_columns: list[str], depth: Optional[int] = None
    ) -> int: ...


class DataTransformationCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        parquet_file: Path,
        output_path: Path,
        hierarchy_columns: list[str],
        working_directory: Union[str, Path],
    ) -> Path: ...


@dataclass
class WideFormDefinition:
    """Helper object for theme type wide form logic definition."""

    hierachy_columns: list[str]
    download_parameters_preparation_function: Optional[DownloadParametersPreparationCallable] = None
    depth_check_function: DepthCheckCallable = _check_depth_for_wide_form
    data_transform_function: DataTransformationCallable = _transform_to_wide_form


THEME_TYPE_CLASSIFICATION = {
    ("base", "infrastructure"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("base", "land"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("base", "land_cover"): WideFormDefinition(hierachy_columns=["subtype"]),
    ("base", "land_use"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("base", "water"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
    ("transportation", "segment"): WideFormDefinition(
        hierachy_columns=["subtype", "class", "subclass"]
    ),
    ("places", "place"): WideFormDefinition(
        hierachy_columns=["primary", "alternate"],
        download_parameters_preparation_function=_prepare_download_parameters_for_poi,
        data_transform_function=_transform_poi_to_wide_form,
    ),
    ("buildings", "building"): WideFormDefinition(hierachy_columns=["subtype", "class"]),
}


@overload
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: str,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@show_total_elapsed_time_decorator
def convert_geometry_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
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
        hierarchy_depth (Optional[int]): Depth used to calculate how many hierarchy columns should
            be used to generate the wide form of the data. If None, will use all available columns.
            Defaults to None.
        pyarrow_filters (Optional[list[Optional[PYARROW_FILTER]]], optional): A list of pyarrow
            expressions used to filter specific theme type pair. Must be the same length as the list
            of theme type pairs. Defaults to None.
        result_file_path (Union[str, Path], optional): Where to save
            the geoparquet file. If not provided, will be generated based on hashes
            from filters. Defaults to `None`.
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

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    if pyarrow_filters is not None and len(theme_type_pairs) != len(pyarrow_filters):
        raise ValueError("Pyarrow filters length doesn't match length of theme type pairs.")

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
            hierarchy_depth=hierarchy_depth,
            pyarrow_filters=pyarrow_filters_list,
        )

    result_file_path = Path(result_file_path)

    if not result_file_path.exists() or ignore_cache:
        result_file_path.parent.mkdir(exist_ok=True, parents=True)

        prepared_download_parameters = _prepare_download_parameters_for_all_theme_type_pairs(
            theme_type_pairs=theme_type_pairs,
            geometry_filter=geometry_filter,
            hierarchy_depth=hierarchy_depth,
            pyarrow_filters=pyarrow_filters_list,
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
                ["id", "geometry", *columns_to_download]
                for columns_to_download in columns_to_download_list
            ],
            ignore_cache=ignore_cache,
            working_directory=working_directory,
            verbosity_mode=verbosity_mode,
            max_workers=max_workers,
        )

        with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)

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
                    wide_form_definition = THEME_TYPE_CLASSIFICATION[(theme_value, type_value)]

                    output_path = (
                        transformed_wide_form_directory_output
                        / f"{theme_value}_{type_value}.parquet"
                    )
                    if len(theme_type_pairs) == 1:
                        output_path = result_file_path

                    wide_form_definition.data_transform_function(
                        theme=theme_value,
                        type=type_value,
                        parquet_file=downloaded_parquet_file,
                        output_path=output_path,
                        hierarchy_columns=hierachy_columns,
                        working_directory=tmp_dir_path,
                    )

            if len(theme_type_pairs) > 1:
                with TrackProgressSpinner(
                    "Joining results to a single file", verbosity_mode=verbosity_mode
                ):
                    _combine_multiple_wide_form_files(
                        theme_type_pairs=theme_type_pairs,
                        transformed_wide_form_directory=transformed_wide_form_directory_output,
                        output_path=result_file_path,
                        working_directory=tmp_dir_path,
                    )

    return result_file_path


@overload
def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    release: str,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


def convert_geometry_to_wide_form_parquet_for_all_types(
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
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
        hierarchy_depth (Optional[int]): Depth used to calculate how many hierarchy columns should
            be used to generate the wide form of the data. If None, will use all available columns.
            Defaults to None.
        pyarrow_filters (Optional[list[Optional[PYARROW_FILTER]]], optional): A list of pyarrow
            expressions used to filter specific theme type pair. Must be the same length as the list
            of theme type pairs. Defaults to None.
        result_file_path (Union[str, Path], optional): Where to save
            the geoparquet file. If not provided, will be generated based on hashes
            from filters. Defaults to `None`.
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

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    return convert_geometry_to_wide_form_parquet_for_multiple_types(
        theme_type_pairs=list(THEME_TYPE_CLASSIFICATION.keys()),
        geometry_filter=geometry_filter,
        release=release,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
    )


def _generate_result_file_path(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    hierarchy_depth: Optional[int],
    pyarrow_filters: Optional[list[Union["Expression", None]]],
) -> Path:
    import hashlib

    directory = Path(release)

    if len(theme_type_pairs) == 1:
        theme_value, type_value = theme_type_pairs[0]
        directory = directory / f"theme={theme_value}" / f"type={type_value}"
    else:
        h = hashlib.new("sha256")
        h.update(str(sorted(theme_type_pairs)).encode())
        directory = directory / h.hexdigest()

    clipping_geometry_hash_part = _generate_geometry_hash(geometry_filter)

    pyarrow_filter_hash_part = "nofilter"
    if pyarrow_filters is not None:
        h = hashlib.new("sha256")
        for single_pyarrow_filter in pyarrow_filters:
            h.update(str(single_pyarrow_filter).encode())
        pyarrow_filter_hash_part = h.hexdigest()

    hierarchy_hash_part = ""
    if hierarchy_depth is not None:
        hierarchy_hash_part = f"_h{hierarchy_depth}"

    return directory / (
        f"{clipping_geometry_hash_part}_{pyarrow_filter_hash_part}"
        f"_wide_form{hierarchy_hash_part}.parquet"
    )


def _prepare_download_parameters_for_all_theme_type_pairs(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[list[Union["Expression", None]]] = None,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> list[tuple[list[str], list[str], Optional[PYARROW_FILTER]]]:
    prepared_parameters = []
    with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
        for idx, (theme_value, type_value) in progress.track(
            enumerate(theme_type_pairs),
            total=len(theme_type_pairs),
            description="Preparing download parameters",
        ):
            single_pyarrow_filter = pyarrow_filters[idx] if pyarrow_filters else None
            wide_form_definition = THEME_TYPE_CLASSIFICATION[(theme_value, type_value)]

            depth = wide_form_definition.depth_check_function(
                wide_form_definition.hierachy_columns, hierarchy_depth
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
            if col not in ("id", "geometry")
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
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)
