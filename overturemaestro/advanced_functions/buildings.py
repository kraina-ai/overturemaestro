"""Functions for retrieving Overture Maps buildings with parts as a single geometry."""

#     convert_bounding_box_to_buildings_geodataframe,
#     convert_bounding_box_to_buildings_parquet,
#     convert_geometry_to_buildings_geodataframe,
#     convert_geometry_to_buildings_parquet,

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, overload

import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from overturemaestro._duckdb import _set_up_duckdb_connection
from overturemaestro._rich_progress import VERBOSITY_MODE
from overturemaestro.data_downloader import (
    _download_data,
    _generate_geometry_hash,
    pyarrow_filters,
)
from overturemaestro.elapsed_time_decorator import show_total_elapsed_time_decorator
from overturemaestro.release_index import get_newest_release_version

if TYPE_CHECKING:
    from pyarrow.compute import Expression

# TODO: write test for checking if geometries of buildings without parts are untouched


@overload
def convert_geometry_to_buildings_parquet(
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
def convert_geometry_to_buildings_parquet(
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
def convert_geometry_to_buildings_parquet(
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


@show_total_elapsed_time_decorator
def convert_geometry_to_buildings_parquet(
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
    Get a GeoParquet file with Overture Maps buildings data with parts as a single geometry.

    Automatically downloads Overture Maps buildings and building_parts dataset for a given release
    in a concurrent manner and returns a single file as a result.

    Args:
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
    with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)

        if not release:
            release = get_newest_release_version()

        all_building_parquet_files = _download_data(
            release=release,
            theme_type_pairs=[("buildings", "building"), ("buildings", "building_part")],
            geometry_filter=geometry_filter,
            pyarrow_filter=[pyarrow_filter, None],
            columns_to_download=[columns_to_download, ["building_id", "geometry"]],
            work_directory=tmp_dir_path,
            verbosity_mode=verbosity_mode,
        )

        downloaded_buildings_parquet_paths = []
        downloaded_building_parts_parquet_paths = []

        for downloaded_file_path in all_building_parquet_files:
            schema = pq.read_schema(downloaded_file_path)
            if "building_id" in schema.names:
                downloaded_building_parts_parquet_paths.append(downloaded_file_path)
            else:
                downloaded_buildings_parquet_paths.append(downloaded_file_path)

        if result_file_path is None:
            result_file_path = working_directory / _generate_result_file_path(
                release=release,
                geometry_filter=geometry_filter,
                pyarrow_filter=pyarrow_filter,
            )

        result_file_path = Path(result_file_path)

        result_file_path.parent.mkdir(exist_ok=True, parents=True)

        joined_building_paths = ",".join(f"'{p}'" for p in downloaded_buildings_parquet_paths)
        joined_building_part_paths = ",".join(
            f"'{p}'" for p in downloaded_building_parts_parquet_paths
        )

        join_sql = f"""
        WITH buildings_properties AS (
            SELECT buildings.* EXCLUDE (geometry)
            FROM read_parquet(
                [{joined_building_paths}],
                hive_partitioning=false
            ) buildings
        ), buildings_with_parts AS (
            SELECT
                buildings.id,
                ST_Union_Agg(
                    CASE WHEN building_parts.geometry IS NULL
                    THEN buildings.geometry
                    ELSE ST_Union(buildings.geometry, building_parts.geometry)
                    END
                ) AS geometry
            FROM read_parquet(
                [{joined_building_paths}],
                hive_partitioning=false
            ) buildings
            LEFT JOIN read_parquet(
                [{joined_building_part_paths}],
                hive_partitioning=false
            ) building_parts
            ON buildings.id = building_parts.building_id
            GROUP BY id
        )
        SELECT
            *
        FROM buildings_with_parts
        JOIN buildings_properties
        USING (id)
        """

        copy_sql = f"""
        COPY ({join_sql}) TO '{result_file_path}' (
            FORMAT 'parquet',
            PER_THREAD_OUTPUT false
        )
        """

        connection = _set_up_duckdb_connection(tmp_dir_path)
        connection.execute(copy_sql)

        return result_file_path


def _generate_result_file_path(
    release: str,
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
        / "buildings_with_parts"
        / f"{clipping_geometry_hash_part}_{pyarrow_filter_hash_part}.parquet"
    )
