"""Functions for retrieving Overture Maps features in a wide form."""

# TODO: replace with dedicated DuckDB functions - POI can have multiple values at once
import tempfile
from pathlib import Path
from typing import Optional, Union, overload

import duckdb
from shapely.geometry.base import BaseGeometry

from overturemaestro._rich_progress import VERBOSITY_MODE
from overturemaestro.data_downloader import (
    download_data,
    pyarrow_filters,
)

THEME_TYPE_CLASSIFICATION = {
    ("base", "infrastructure"): ["subtype", "class"],
    ("base", "land"): ["subtype", "class"],
    ("base", "land_cover"): ["subtype", "class"],
    ("base", "land_use"): ["subtype", "class"],
    ("base", "water"): ["subtype", "class"],
    ("transportation", "segment"): ["subtype", "class", "subclass"],
    ("places", "place"): ["categories.primary", "categories.primary"],
    ("buildings", "building"): ["subtype", "class"],
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
    pyarrow_filter: Optional[pyarrow_filters] = None,
    columns_to_download: Optional[list[str]] = None,
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
    pyarrow_filter: Optional[pyarrow_filters] = None,
    columns_to_download: Optional[list[str]] = None,
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
    pyarrow_filter: Optional[pyarrow_filters] = None,
    columns_to_download: Optional[list[str]] = None,
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
    pyarrow_filter: Optional[pyarrow_filters] = None,
    columns_to_download: Optional[list[str]] = None,
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
        pyarrow_filter (Optional[pyarrow_filters], optional): Filters to apply on a pyarrow dataset.
            Can be pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]]. Defaults to None.
        columns_to_download (Optional[list[str]], optional): List of columns to download.
            Automatically adds geometry column to the list. If None, will download all columns.
            Defaults to None.
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
    return download_data(
        release=release,
        theme=theme,
        type=type,
        geometry_filter=geometry_filter,
        pyarrow_filter=pyarrow_filter,
        columns_to_download=columns_to_download,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
    )

def _get_transform_to_wide_form_function():
    hierarchy_columns = ["subtype", "class", "subclass"]

    def _transform_to_wide_form(
        parquet_path: Path,
        working_directory: Union[str, Path] = "files",
    ) -> Path:
        with tempfile.TemporaryDirectory(dir=Path(Path(working_directory)).resolve()) as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)

            connection = set_up_duckdb_connection(tmp_dir_path)

            joined_hierarchy_columns = ",".join(hierarchy_columns)

            wide_column_definitions = (
                connection.sql(
                    f"""
                    SELECT DISTINCT
                        {joined_hierarchy_columns},
                        concat_ws(
                            '|',
                            {joined_hierarchy_columns}
                        ) as column_name
                    FROM '{parquet_path}'
                    ORDER BY column_name
                    """
                )
                .fetchdf()
                .to_dict(orient="records")
            )

            case_clauses = []

            for wide_column_definition in wide_column_definitions:
                column_name = wide_column_definition["column_name"]
                conditions = []
                for condition_column in hierarchy_columns:
                    if wide_column_definition[condition_column] is None:
                        conditions.append(f"{condition_column} IS NULL")
                    else:
                        # TODO: escape sql value
                        escaped_value = wide_column_definition[condition_column]
                        conditions.append(f"{condition_column} = '{escaped_value}'")
                case_clauses.append(
                    f'CASE WHEN {" AND ".join(conditions)} THEN 1 ELSE 0 END AS "{column_name}"'
                )
            # case_clauses

    return _transform_to_wide_form


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


def set_up_duckdb_connection(tmp_dir_path: Path) -> "duckdb.DuckDBPyConnection":
    local_db_file = "db.duckdb"
    connection = duckdb.connect(
        database=str(tmp_dir_path / local_db_file),
        config=dict(preserve_insertion_order=False),
    )
    connection.sql("SET enable_progress_bar = false;")
    connection.sql("SET enable_progress_bar_print = false;")

    connection.install_extension("spatial")
    connection.load_extension("spatial")

    return connection
