"""Functions for retrieving Overture Maps features in a wide form."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Union, overload

from shapely.geometry.base import BaseGeometry

from overturemaestro._duckdb import _set_up_duckdb_connection, _sql_escape
from overturemaestro._exceptions import HierarchyDepthOutOfBoundsError
from overturemaestro._rich_progress import VERBOSITY_MODE, TrackProgressBar
from overturemaestro.data_downloader import _download_data, _generate_geometry_hash, pyarrow_filters
from overturemaestro.elapsed_time_decorator import show_total_elapsed_time_decorator
from overturemaestro.release_index import get_newest_release_version

if TYPE_CHECKING:
    from pyarrow.compute import Expression


def _prepare_download_parameters_for_poi(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    hierachy_columns: list[str],
    pyarrow_filter: Optional[pyarrow_filters] = None,
) -> tuple[list[str], Optional[pyarrow_filters]]:
    # TODO: swap to dedicated function?
    import pyarrow.compute as pc

    category_not_null_filter = pc.invert(pc.field("categories").is_null(nan_is_null=True))
    if pyarrow_filter is not None:
        from pyarrow.parquet import filters_to_expression

        pyarrow_filter = filters_to_expression(pyarrow_filter) & category_not_null_filter
    else:
        pyarrow_filter = category_not_null_filter

    return (["categories"], pyarrow_filter)


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
    parquet_paths: list[Path],
    output_path: Path,
    hierarchy_columns: list[str],
    working_directory: Union[str, Path],
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

    joined_hierarchy_columns = ",".join(hierarchy_columns)

    joined_parquet_paths = ",".join(f"'{p}'" for p in parquet_paths)

    available_colums_sql_query = f"""
    SELECT DISTINCT
        {joined_hierarchy_columns},
        concat_ws('|', {joined_hierarchy_columns}) as column_name
    FROM read_parquet(
        [{joined_parquet_paths}],
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
            {", ".join(case_clauses)},
            geometry
        FROM read_parquet(
            [{joined_parquet_paths}],
            hive_partitioning=false
        )
    ) TO '{output_path}' (
        FORMAT 'parquet',
        PER_THREAD_OUTPUT false
    )
    """

    connection.execute(query)

    return output_path


def _transform_poi_to_wide_form(
    theme: str,
    type: str,
    parquet_paths: list[Path],
    output_path: Path,
    hierarchy_columns: list[str],
    working_directory: Union[str, Path],
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

    primary_category_only = len(hierarchy_columns) == 1

    joined_parquet_paths = ",".join(f"'{p}'" for p in parquet_paths)

    if primary_category_only:
        available_colums_sql_query = f"""
        SELECT DISTINCT
            categories.primary as column_name
        FROM read_parquet(
            [{joined_parquet_paths}],
            hive_partitioning=false
        )
        """
    else:
        available_colums_sql_query = f"""
        SELECT DISTINCT
            categories.primary as column_name
        FROM read_parquet(
            [{joined_parquet_paths}],
            hive_partitioning=false
        )
        UNION
        SELECT DISTINCT
            UNNEST(categories.alternate) as column_name
        FROM read_parquet(
            [{joined_parquet_paths}],
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
        case_clauses.append(f'COALESCE(({joined_conditions}), False) AS "{column_name}"')

    query = f"""
    COPY (
        SELECT
            id,
            {", ".join(case_clauses)},
            geometry
        FROM read_parquet(
            [{joined_parquet_paths}],
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
        pyarrow_filter: Optional[pyarrow_filters] = None,
    ) -> tuple[list[str], Optional[pyarrow_filters]]: ...


class DepthCheckCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, hierarchy_columns: list[str], depth: Optional[int] = None
    ) -> int: ...


class DataTransformationCallable(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self,
        theme: str,
        type: str,
        parquet_paths: list[Path],
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

# convert_bounding_box_to_wide_form_geodataframe,
# convert_bounding_box_to_wide_form_parquet,
# convert_geometry_to_wide_form_geodataframe,
# convert_geometry_to_wide_form_parquet,


@overload
def convert_geometry_to_wide_form_parquet(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: str,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@show_total_elapsed_time_decorator
def convert_geometry_to_wide_form_parquet(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> Path:
    """
    Get a GeoParquet file with Overture Maps data within given geometry in a wide form.

    Automatically downloads Overture Maps dataset for a given release and theme/type
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
        pyarrow_filter (Optional[pyarrow_filters], optional): Filters to apply on a pyarrow dataset.
            Can be pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]]. Defaults to None.
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
    with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)

        if not release:
            release = get_newest_release_version()

        if result_file_path is None:
            result_file_path = working_directory / _generate_result_file_path(
                release=release,
                theme_type_pairs=theme_type_pairs,
                geometry_filter=geometry_filter,
                pyarrow_filter=pyarrow_filter,
            )

        result_file_path = Path(result_file_path)

        if not result_file_path.exists() or ignore_cache:
            result_file_path.parent.mkdir(exist_ok=True, parents=True)

            prepared_download_parameters = _prepare_download_parameters_for_all_theme_type_pairs(
                theme_type_pairs=theme_type_pairs,
                geometry_filter=geometry_filter,
                hierarchy_depth=hierarchy_depth,
                pyarrow_filter=pyarrow_filter,
            )

            hierachy_columns_list, pyarrow_filter_list = zip(*prepared_download_parameters)

            all_downloaded_parquet_files = _download_data(
                release=release,
                theme_type_pairs=theme_type_pairs,
                geometry_filter=geometry_filter,
                pyarrow_filter=pyarrow_filter_list,
                columns_to_download=[
                    ["id", "geometry", *hierachy_columns]
                    for hierachy_columns in hierachy_columns_list
                ],
                work_directory=tmp_dir_path,
                verbosity_mode=verbosity_mode,
                max_workers=max_workers,
            )

            transformed_wide_form_directory_output = tmp_dir_path / "wide_form_files"
            transformed_wide_form_directory_output.mkdir(parents=True, exist_ok=True)
            with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
                for (
                    (theme_value, type_value),
                    hierachy_columns,
                    downloaded_parquet_files,
                ) in progress.track(
                    zip(theme_type_pairs, hierachy_columns_list, all_downloaded_parquet_files),
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
                        parquet_paths=downloaded_parquet_files,
                        output_path=output_path,
                        hierarchy_columns=hierachy_columns,
                        working_directory=tmp_dir_path,
                    )

            if len(theme_type_pairs) > 1:
                _combine_multiple_wide_form_files(
                    theme_type_pairs=theme_type_pairs,
                    transformed_wide_form_directory=transformed_wide_form_directory_output,
                    output_path=result_file_path,
                    working_directory=tmp_dir_path,
                )

        return result_file_path


def _generate_result_file_path(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    pyarrow_filter: Optional["Expression"],
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
    if pyarrow_filter is not None:
        h = hashlib.new("sha256")
        h.update(str(pyarrow_filter).encode())
        pyarrow_filter_hash_part = h.hexdigest()

    return directory / f"{clipping_geometry_hash_part}_{pyarrow_filter_hash_part}_wide_form.parquet"


def _prepare_download_parameters_for_all_theme_type_pairs(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
) -> list[tuple[list[str], Optional[pyarrow_filters]]]:
    prepared_parameters = []
    for theme_value, type_value in theme_type_pairs:
        wide_form_definition = THEME_TYPE_CLASSIFICATION[(theme_value, type_value)]

        depth = wide_form_definition.depth_check_function(
            wide_form_definition.hierachy_columns, hierarchy_depth
        )
        hierachy_columns = wide_form_definition.hierachy_columns[:depth]
        copied_pyarrow_filter = pyarrow_filter

        if wide_form_definition.download_parameters_preparation_function is not None:
            hierachy_columns, copied_pyarrow_filter = (
                wide_form_definition.download_parameters_preparation_function(
                    theme=theme_value,
                    type=type_value,
                    geometry_filter=geometry_filter,
                    hierachy_columns=hierachy_columns,
                    pyarrow_filter=copied_pyarrow_filter,
                )
            )

        prepared_parameters.append((hierachy_columns, copied_pyarrow_filter))

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

        modified_column_aliases = [f"{theme_value}|{type_value}|{col}" for col in available_columns]

        combined_columns_select = ", ".join(
            f'"{column}" as "{alias}"'
            for column, alias in zip(available_columns, modified_column_aliases)
        )

        select_subquery = f"""
        SELECT id, geometry, {combined_columns_select}
        FROM read_parquet('{current_parquet_file}', hive_partitioning=false)
        """

        subqueries.append(select_subquery)
        all_select_columns.extend(modified_column_aliases)

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