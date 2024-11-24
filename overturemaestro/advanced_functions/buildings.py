"""Functions for retrieving Overture Maps buildings with parts as a single geometry."""

#     convert_bounding_box_to_buildings_geodataframe,
#     convert_bounding_box_to_buildings_parquet,
#     convert_geometry_to_buildings_geodataframe,
#     convert_geometry_to_buildings_parquet,

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, overload

from shapely.geometry.base import BaseGeometry

from overturemaestro._duckdb import _set_up_duckdb_connection
from overturemaestro._rich_progress import VERBOSITY_MODE
from overturemaestro.data_downloader import _generate_geometry_hash, download_data, pyarrow_filters
from overturemaestro.elapsed_time_decorator import show_total_elapsed_time_decorator
from overturemaestro.release_index import get_newest_release_version

if TYPE_CHECKING:
    from pyarrow.compute import Expression

# TODO: write test for checking if geometries of buildings without parts are untouched
# TODO: remove hive partitioning from reading parquet


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
    # download buildings
    # download building_parts

    # combine geometries using duckdb
    with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)

        if not release:
            release = get_newest_release_version()

        downloaded_buildings_parquet_path = download_data(
            release=release,
            theme="buildings",
            type="building",
            geometry_filter=geometry_filter,
            pyarrow_filter=pyarrow_filter,
            columns_to_download=columns_to_download,
            ignore_cache=ignore_cache,
            working_directory=tmp_dir_path,
            verbosity_mode=verbosity_mode,
        )

        downloaded_building_parts_parquet_path = download_data(
            release=release,
            theme="buildings",
            type="building_part",
            geometry_filter=geometry_filter,
            columns_to_download=["building_id", "geometry"],
            ignore_cache=ignore_cache,
            working_directory=tmp_dir_path,
            verbosity_mode=verbosity_mode,
        )

        if result_file_path is None:
            result_file_path = working_directory / _generate_result_file_path(
                release=release,
                geometry_filter=geometry_filter,
                pyarrow_filter=pyarrow_filter,
            )

        result_file_path = Path(result_file_path)

        result_file_path.parent.mkdir(exist_ok=True, parents=True)

        join_sql = f"""
        WITH buildings_properties AS (
            SELECT buildings.* EXCLUDE (geometry)
            FROM read_parquet(
                '{downloaded_buildings_parquet_path}',
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
                '{downloaded_buildings_parquet_path}',
                hive_partitioning=false
            ) buildings
            LEFT JOIN read_parquet(
                '{downloaded_building_parts_parquet_path}',
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
            -- ROW_GROUP_SIZE 25000,
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
