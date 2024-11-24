"""Functions for retrieving Overture Maps features in a wide form."""

import tempfile
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, overload

from shapely.geometry.base import BaseGeometry

from overturemaestro._duckdb import _set_up_duckdb_connection, _sql_escape
from overturemaestro._exceptions import HierarchyDepthOutOfBoundsError
from overturemaestro._rich_progress import VERBOSITY_MODE
from overturemaestro.data_downloader import _generate_geometry_hash, download_data, pyarrow_filters
from overturemaestro.release_index import get_newest_release_version

if TYPE_CHECKING:
    from pyarrow.compute import Expression

# TODO: add dedicated function for downloading raw data (download_data / buildings / etc)


def _check_depth_for_wide_form(hierarchy_columns: list[str], depth: Optional[int] = None) -> int:
    depth = depth if depth is not None else len(hierarchy_columns)

    if depth < 1 or depth > len(hierarchy_columns):
        raise HierarchyDepthOutOfBoundsError(
            f"Provided hierarchy depth is out of bounds (valid: 1 - {len(hierarchy_columns)})"
        )

    return depth


def _transform_to_wide_form(
    parquet_path: Path,
    output_path: Path,
    hierarchy_columns: list[str],
    working_directory: Union[str, Path] = "files",
) -> Path:
    connection = _set_up_duckdb_connection(working_directory)

    joined_hierarchy_columns = ",".join(hierarchy_columns)

    available_colums_sql_query = f"""
    SELECT DISTINCT
        {joined_hierarchy_columns},
        concat_ws('|', {joined_hierarchy_columns}) as column_name
    FROM '{parquet_path}'
    ORDER BY column_name
    """

    wide_column_definitions = (
        connection.sql(available_colums_sql_query).fetchdf().to_dict(orient="records")
    )

    case_clauses = []

    for wide_column_definition in wide_column_definitions:
        column_name = wide_column_definition["column_name"]
        conditions = []
        for condition_column in hierarchy_columns:
            if wide_column_definition[condition_column] is None:
                conditions.append(f"{condition_column} IS NULL")
            else:
                escaped_value = _sql_escape(wide_column_definition[condition_column])
                conditions.append(f"{condition_column} = '{escaped_value}'")

        joined_conditions = " AND ".join(conditions)
        case_clauses.append(f'({joined_conditions}) AS "{column_name}"')

    query = f"""
    COPY (
        SELECT
            id,
            {", ".join(case_clauses)},
            geometry
        FROM '{parquet_path}'
    ) TO '{output_path}' (
        FORMAT 'parquet',
        PER_THREAD_OUTPUT false
        -- ROW_GROUP_SIZE 25000,
    )
    """

    connection.execute(query)

    return output_path


WideFormDefinition = namedtuple(
    "WideFormDefinition", ["download_columns", "depth_check_function", "data_transform_function"]
)

THEME_TYPE_CLASSIFICATION = {
    ("base", "infrastructure"): WideFormDefinition(
        ["subtype", "class"], _check_depth_for_wide_form, _transform_to_wide_form
    ),
    ("base", "land"): WideFormDefinition(
        ["subtype", "class"], _check_depth_for_wide_form, _transform_to_wide_form
    ),
    ("base", "land_cover"): WideFormDefinition(
        ["subtype"], _check_depth_for_wide_form, _transform_to_wide_form
    ),
    ("base", "land_use"): WideFormDefinition(
        ["subtype", "class"], _check_depth_for_wide_form, _transform_to_wide_form
    ),
    ("base", "water"): WideFormDefinition(
        ["subtype", "class"], _check_depth_for_wide_form, _transform_to_wide_form
    ),
    ("transportation", "segment"): WideFormDefinition(
        ["subtype", "class", "subclass"],
        _check_depth_for_wide_form,
        _transform_to_wide_form,
    ),
    # ("places", "place"): ["categories.primary", "categories.primary"],
    ("buildings", "building"): WideFormDefinition(
        ["subtype", "class"], _check_depth_for_wide_form, _transform_to_wide_form
    ),
}

# convert_bounding_box_to_wide_form_geodataframe,
# convert_bounding_box_to_wide_form_parquet,
# convert_geometry_to_wide_form_geodataframe,
# convert_geometry_to_wide_form_parquet,


@overload
def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: str,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> Path: ...


@overload
def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> Path: ...


def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> Path:
    """
    Get a GeoParquet file with Overture Maps data within given geometry in a wide form.

    Automatically downloads Overture Maps dataset for a given release and theme/type
    in a concurrent manner and returns a single file as a result with multiple columns based
    on dataset schema.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
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

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)

        if not release:
            release = get_newest_release_version()

        wide_form_definition = THEME_TYPE_CLASSIFICATION[(theme, type)]

        depth = wide_form_definition.depth_check_function(
            wide_form_definition.download_columns, hierarchy_depth
        )
        columns_to_download = wide_form_definition.download_columns[:depth]

        downloaded_parquet_path = download_data(
            release=release,
            theme=theme,
            type=type,
            geometry_filter=geometry_filter,
            pyarrow_filter=pyarrow_filter,
            columns_to_download=["id", *columns_to_download, "geometry"],
            ignore_cache=ignore_cache,
            working_directory=tmp_dir_path,
            verbosity_mode=verbosity_mode,
        )

        if result_file_path is None:
            result_file_path = working_directory / _generate_result_file_path(
                release=release,
                theme=theme,
                type=type,
                geometry_filter=geometry_filter,
                pyarrow_filter=pyarrow_filter,
            )

        result_file_path = Path(result_file_path)

        wide_form_definition.data_transform_function(
            parquet_path=downloaded_parquet_path,
            output_path=result_file_path,
            hierarchy_columns=columns_to_download,
            working_directory=tmp_dir_path,
        )

        return result_file_path


# wide_column_definitions


# osm_tag_keys = set()
# found_tag_keys = [
#     row[0]
#     for row in self.connection.sql(
#         f"""
#         SELECT DISTINCT UNNEST(map_keys(tags)) tag_key
#         FROM ({parsed_geometries.sql_query()})
#         """
#     ).fetchall()
# ]
# osm_tag_keys.update(found_tag_keys)
# osm_tag_keys_select_clauses = [
#     f"list_extract(map_extract(tags, '{osm_tag_key}'), 1) as \"{osm_tag_key}\""
#     for osm_tag_key in sorted(list(osm_tag_keys))
# ]


def _generate_result_file_path(
    release: str,
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    pyarrow_filter: Optional["Expression"],
) -> Path:
    import hashlib

    clipping_geometry_hash_part = _generate_geometry_hash(geometry_filter)

    pyarrow_filter_hash_part = "nofilter"
    if pyarrow_filter is not None:
        h = hashlib.new("sha256")
        h.update(str(pyarrow_filter).encode())
        pyarrow_filter_hash_part = h.hexdigest()

    return (
        Path(release)
        / f"theme={theme}"
        / f"type={type}"
        / f"{clipping_geometry_hash_part}_{pyarrow_filter_hash_part}_wide_form.parquet"
    )
