"""Functions for retrieving Overture Maps divisions with their hierarchy."""

import tempfile
from pathlib import Path
from typing import Literal, Optional, Union, get_args

import geopandas as gpd
import pyarrow.parquet as pq
from rq_geo_toolkit.duckdb import set_up_duckdb_connection, sql_escape
from rq_geo_toolkit.file_utils import atomic_output_path
from shapely import box
from shapely.geometry.base import BaseGeometry

from overturemaestro._constants import (
    GEOMETRY_COLUMN,
    INDEX_COLUMN,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from overturemaestro._rich_progress import VERBOSITY_MODE
from overturemaestro.functions import convert_geometry_to_parquet
from overturemaestro.geopandas_io import geodataframe_from_parquet
from overturemaestro.release_index import get_newest_release_version

__all__ = [
    "DIVISION_SUBTYPE",
    "convert_bounding_box_to_divisions_geodataframe",
    "convert_bounding_box_to_divisions_parquet",
    "convert_geometry_to_divisions_geodataframe",
    "convert_geometry_to_divisions_parquet",
]

DIVISION_SUBTYPE = Literal[
    "country",
    "dependency",
    "region",
    "county",
    "localadmin",
    "locality",
    "macrohood",
    "neighborhood",
    "microhood",
]

HIERARCHY_SUBTYPES = ("country", "region", "county", "localadmin", "locality")

# Points of some of the divisions are placed outside of their polygons
DIVISION_POINTS_BUFFER = 0.1


def convert_geometry_to_divisions_parquet(
    subtype: DIVISION_SUBTYPE,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
) -> Path:
    """
    Get a GeoParquet file with divisions of a given subtype intersecting given geometry.

    Downloads land polygons from the `division_area` type and joins them with the hierarchy
    from the `division` type. Full, not clipped polygons are returned.

    Only divisions of a given subtype are returned. Not every country has all of the subtypes and
    the same subtype can mean a different thing in different countries. Polygons can be missing for
    levels below the county. Localities can be nested, so their polygons can overlap.

    Result columns:
        - `id`: id of the division, it's the same as `division_id` in the Overture Maps data.
        - `division_area_id`: id of the polygon.
        - `subtype`, `name`, `name_en`, `country` and `region` (ISO codes).
        - `parent_division_id`, `wikidata` and `population`.
        - `country_id`, `country_name` and the same for the region, county, localadmin
          and locality. Outermost one for the nested localities, empty if not in a hierarchy.
        - `hierarchies`: list of all divisions starting from the country.
        - `geometry`.

    A few divisions with a point far away from a polygon have empty hierarchy columns.

    Args:
        subtype (DIVISION_SUBTYPE): Subtype of divisions to download. One of: country, dependency,
            region, county, localadmin, locality, macrohood, neighborhood, microhood.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
        result_file_path (Optional[Union[str, Path]], optional): Where to save
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

    Raises:
        ValueError: If subtype is unknown.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    if subtype not in get_args(DIVISION_SUBTYPE):
        raise ValueError(
            f"Unknown division subtype: {subtype}."
            f" Available subtypes: {', '.join(get_args(DIVISION_SUBTYPE))}."
        )

    release = release or get_newest_release_version()

    areas_path = convert_geometry_to_parquet(
        theme="divisions",
        type="division_area",
        geometry_filter=geometry_filter,
        release=release,
        pyarrow_filter=[("subtype", "=", subtype), ("class", "=", "land")],
        columns_to_download=[
            "id",
            "division_id",
            "subtype",
            "names",
            "country",
            "region",
            "bbox",
            "geometry",
        ],
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
    )

    result_file_path = Path(
        result_file_path
        or Path(working_directory) / release / "divisions_with_hierarchy" / areas_path.name
    )
    if result_file_path.exists() and not ignore_cache:
        return result_file_path

    # Points can be outside of the geometry filter, but close to their polygons
    xmin, ymin, xmax, ymax = _get_bounds(areas_path) or geometry_filter.bounds
    divisions_path = convert_geometry_to_parquet(
        theme="divisions",
        type="division",
        geometry_filter=box(
            xmin - DIVISION_POINTS_BUFFER,
            ymin - DIVISION_POINTS_BUFFER,
            xmax + DIVISION_POINTS_BUFFER,
            ymax + DIVISION_POINTS_BUFFER,
        ),
        release=release,
        pyarrow_filter=[("subtype", "=", subtype)],
        columns_to_download=[
            "id",
            "hierarchies",
            "parent_division_id",
            "wikidata",
            "population",
            "geometry",
        ],
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=False,
    )

    result_file_path.parent.mkdir(parents=True, exist_ok=True)
    _join_areas_with_hierarchy(
        areas_path=areas_path,
        divisions_path=divisions_path,
        output_path=result_file_path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
    )

    return result_file_path


def convert_geometry_to_divisions_geodataframe(
    subtype: DIVISION_SUBTYPE,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
) -> gpd.GeoDataFrame:
    """
    Get a GeoDataFrame with divisions of a given subtype intersecting given geometry.

    See `convert_geometry_to_divisions_parquet` for the description of arguments and columns.

    Args:
        subtype (DIVISION_SUBTYPE): Subtype of divisions to download.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. Defaults to None (newest available).
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.
        sort_result (bool, optional): Whether to sort the result by geometry or not.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with divisions, indexed by the division id.
    """
    return geodataframe_from_parquet(
        convert_geometry_to_divisions_parquet(
            subtype=subtype,
            geometry_filter=geometry_filter,
            release=release,
            ignore_cache=ignore_cache,
            working_directory=working_directory,
            verbosity_mode=verbosity_mode,
            max_workers=max_workers,
            sort_result=sort_result,
        )
    )


def convert_bounding_box_to_divisions_parquet(
    subtype: DIVISION_SUBTYPE,
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
) -> Path:
    """
    Get a GeoParquet file with divisions of a given subtype intersecting given bounding box.

    See `convert_geometry_to_divisions_parquet` for the full description of arguments and columns.

    Args:
        subtype (DIVISION_SUBTYPE): Subtype of divisions to download.
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
        release (Optional[str], optional): Release version. Defaults to None (newest available).
        compression (str, optional): Compression of the final parquet file. Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
        result_file_path (Optional[Union[str, Path]], optional): Where to save
            the geoparquet file. Defaults to None (generated from filters).
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.
        sort_result (bool, optional): Whether to sort the result by geometry or not.
            Defaults to True.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    return convert_geometry_to_divisions_parquet(
        subtype=subtype,
        geometry_filter=box(*bbox),
        release=release,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
    )


def convert_bounding_box_to_divisions_geodataframe(
    subtype: DIVISION_SUBTYPE,
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
) -> gpd.GeoDataFrame:
    """
    Get a GeoDataFrame with divisions of a given subtype intersecting given bounding box.

    See `convert_geometry_to_divisions_parquet` for the full description of arguments and columns.

    Args:
        subtype (DIVISION_SUBTYPE): Subtype of divisions to download.
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
        release (Optional[str], optional): Release version. Defaults to None (newest available).
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.
        sort_result (bool, optional): Whether to sort the result by geometry or not.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with divisions, indexed by the division id.
    """
    return convert_geometry_to_divisions_geodataframe(
        subtype=subtype,
        geometry_filter=box(*bbox),
        release=release,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
    )


def _get_bounds(parquet_path: Path) -> Optional[tuple[float, float, float, float]]:
    """Get total bounds of features, None if there are no features."""
    bboxes = pq.read_table(parquet_path, columns=["bbox"])["bbox"].to_pylist()
    if not bboxes:
        return None
    return (
        min(bbox["xmin"] for bbox in bboxes),
        min(bbox["ymin"] for bbox in bboxes),
        max(bbox["xmax"] for bbox in bboxes),
        max(bbox["ymax"] for bbox in bboxes),
    )


def _join_areas_with_hierarchy(
    areas_path: Path,
    divisions_path: Path,
    output_path: Path,
    compression: str,
    compression_level: int,
    row_group_size: int,
) -> None:
    """Join polygons with the flattened hierarchy of their divisions."""
    hierarchy_columns = ", ".join(
        f"[h for h in d.hierarchies[1] if h.subtype = '{subtype}'][1].{source}"
        f" AS {subtype}_{target}"
        for subtype in HIERARCHY_SUBTYPES
        for source, target in (("division_id", "id"), ("name", "name"))
    )
    with (
        tempfile.TemporaryDirectory(dir=output_path.parent) as tmp_dir,
        atomic_output_path(output_path) as tmp_output_path,
        set_up_duckdb_connection(tmp_dir, preserve_insertion_order=True) as connection,
    ):
        connection.execute(f"""
    COPY (
        SELECT
            a.division_id AS {INDEX_COLUMN},
            a.id AS division_area_id,
            a.subtype,
            a.names."primary" AS name,
            map_extract(a.names.common, 'en')[1] AS name_en,
            a.country,
            a.region,
            d.parent_division_id,
            d.wikidata,
            d.population,
            {hierarchy_columns},
            d.hierarchies,
            a.{GEOMETRY_COLUMN}
        FROM read_parquet(
            '{sql_escape(str(areas_path))}', hive_partitioning=false, file_row_number=true
        ) a
        LEFT JOIN read_parquet(
            '{sql_escape(str(divisions_path))}', hive_partitioning=false
        ) d ON a.division_id = d.id
        ORDER BY a.file_row_number
    ) TO '{sql_escape(str(tmp_output_path))}' (
        FORMAT 'parquet',
        ROW_GROUP_SIZE {row_group_size},
        COMPRESSION {compression},
        COMPRESSION_LEVEL {compression_level}
    )
    """)
