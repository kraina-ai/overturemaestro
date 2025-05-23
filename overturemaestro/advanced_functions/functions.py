"""
Advanced Functions.

This module contains helper functions to simplify the API.
"""

from pathlib import Path
from typing import Optional, Union, overload

import geopandas as gpd
from shapely import box
from shapely.geometry.base import BaseGeometry

from overturemaestro._constants import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from overturemaestro._rich_progress import VERBOSITY_MODE
from overturemaestro.advanced_functions.wide_form import (
    convert_geometry_to_wide_form_parquet_for_all_types,
    convert_geometry_to_wide_form_parquet_for_multiple_types,
)
from overturemaestro.data_downloader import PYARROW_FILTER
from overturemaestro.geopandas_io import geodataframe_from_parquet

# TODO: add examples

__all__ = [
    "convert_bounding_box_to_wide_form_geodataframe",
    "convert_bounding_box_to_wide_form_geodataframe_for_all_types",
    "convert_bounding_box_to_wide_form_geodataframe_for_multiple_types",
    "convert_bounding_box_to_wide_form_parquet",
    "convert_bounding_box_to_wide_form_parquet_for_all_types",
    "convert_bounding_box_to_wide_form_parquet_for_multiple_types",
    "convert_geometry_to_wide_form_geodataframe",
    "convert_geometry_to_wide_form_geodataframe_for_all_types",
    "convert_geometry_to_wide_form_geodataframe_for_multiple_types",
    "convert_geometry_to_wide_form_parquet",
    "convert_geometry_to_wide_form_parquet_for_all_types",
    "convert_geometry_to_wide_form_parquet_for_multiple_types",
]


@overload
def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


def convert_geometry_to_wide_form_parquet(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
    Get GeoParquet file for a given geometry in a wide format.

    Automatically downloads Overture Maps dataset for a given release and theme/type
    in a concurrent manner and returns a single file as a result with multiple columns based
    on dataset schema.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        include_all_possible_columns (bool, optional): Whether to have always the same list of
            columns in the resulting file. This ensures that always the same set of columns is
            returned for a given release for different regions. This also means, that some columns
            might be all filled with a False value. Defaults to True.
        hierarchy_depth (Optional[int], optional): Depth used to calculate how many hierarchy
            columns should be used to generate the wide form of the data. If None, will use all
            available columns. Defaults to None.
        pyarrow_filters (Optional[PYARROW_FILTER], optional): A pyarrow expression used to filter
            specific theme type pair. Defaults to None.
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    return convert_geometry_to_wide_form_parquet_for_multiple_types(
        theme_type_pairs=[(theme, type)],
        geometry_filter=geometry_filter,
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=[pyarrow_filters],
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )


@overload
def convert_geometry_to_wide_form_geodataframe(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_geometry_to_wide_form_geodataframe(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_geometry_to_wide_form_geodataframe(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


def convert_geometry_to_wide_form_geodataframe(
    theme: str,
    type: str,
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Get GeoDataFrame for a given geometry in a wide format.

    Automatically downloads Overture Maps dataset for a given release and theme/type
    in a concurrent manner and returns a single GeoDataFrame as a result with multiple columns based
    on dataset schema.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        include_all_possible_columns (bool, optional): Whether to have always the same list of
            columns in the resulting file. This ensures that always the same set of columns is
            returned for a given release for different regions. This also means, that some columns
            might be all filled with a False value. Defaults to True.
        hierarchy_depth (Optional[int], optional): Depth used to calculate how many hierarchy
            columns should be used to generate the wide form of the data. If None, will use all
            available columns. Defaults to None.
        pyarrow_filters (Optional[PYARROW_FILTER], optional): A pyarrow expression used to filter
            specific theme type pair. Defaults to None.
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Overture Maps features.
    """
    parsed_geoparquet_file = convert_geometry_to_wide_form_parquet(
        theme=theme,
        type=type,
        geometry_filter=geometry_filter,
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )
    return geodataframe_from_parquet(parsed_geoparquet_file)


@overload
def convert_geometry_to_wide_form_geodataframe_for_all_types(
    geometry_filter: BaseGeometry,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_geometry_to_wide_form_geodataframe_for_all_types(
    geometry_filter: BaseGeometry,
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_geometry_to_wide_form_geodataframe_for_all_types(
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


def convert_geometry_to_wide_form_geodataframe_for_all_types(
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Get GeoDataFrame for a given geometry in a wide format for all types.

    Automatically downloads Overture Maps dataset for a given release and all available theme/types
    in a concurrent manner and returns a single GeoDataFrame as a result with multiple columns based
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
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Overture Maps features.
    """
    parsed_geoparquet_file = convert_geometry_to_wide_form_parquet_for_all_types(
        geometry_filter=geometry_filter,
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )
    return geodataframe_from_parquet(parsed_geoparquet_file)


@overload
def convert_geometry_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_geometry_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_geometry_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


def convert_geometry_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: BaseGeometry,
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Get GeoDataFrame for a given geometry in a wide format for multiple types.

    Automatically downloads Overture Maps dataset for a given release and theme/type pairs
    in a concurrent manner and returns a single GeoDataFrame as a result with multiple columns based
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
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Overture Maps features.
    """
    parsed_geoparquet_file = convert_geometry_to_wide_form_parquet_for_multiple_types(
        theme_type_pairs=theme_type_pairs,
        geometry_filter=geometry_filter,
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )
    return geodataframe_from_parquet(parsed_geoparquet_file)


@overload
def convert_bounding_box_to_wide_form_parquet(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_bounding_box_to_wide_form_parquet(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_bounding_box_to_wide_form_parquet(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


def convert_bounding_box_to_wide_form_parquet(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
    Get GeoParquet file for a given bounding box in a wide format.

    Automatically downloads Overture Maps dataset for a given release and theme/type
    in a concurrent manner and returns a single file as a result with multiple columns based
    on dataset schema.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        include_all_possible_columns (bool, optional): Whether to have always the same list of
            columns in the resulting file. This ensures that always the same set of columns is
            returned for a given release for different regions. This also means, that some columns
            might be all filled with a False value. Defaults to True.
        hierarchy_depth (Optional[int], optional): Depth used to calculate how many hierarchy
            columns should be used to generate the wide form of the data. If None, will use all
            available columns. Defaults to None.
        pyarrow_filters (Optional[PYARROW_FILTER], optional): A pyarrow
            expression used to filter specific theme type pair. Defaults to None.
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    return convert_geometry_to_wide_form_parquet(
        theme=theme,
        type=type,
        geometry_filter=box(*bbox),
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )


@overload
def convert_bounding_box_to_wide_form_parquet_for_all_types(
    bbox: tuple[float, float, float, float],
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_bounding_box_to_wide_form_parquet_for_all_types(
    bbox: tuple[float, float, float, float],
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_bounding_box_to_wide_form_parquet_for_all_types(
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


def convert_bounding_box_to_wide_form_parquet_for_all_types(
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
    Get GeoParquet file for a given bounding box in a wide format for all types.

    Automatically downloads Overture Maps dataset for a given release and all available theme/types
    in a concurrent manner and returns a single file as a result with multiple columns based
    on dataset schema.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
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
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    return convert_geometry_to_wide_form_parquet_for_all_types(
        geometry_filter=box(*bbox),
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )


@overload
def convert_bounding_box_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_bounding_box_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
def convert_bounding_box_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> Path: ...


def convert_bounding_box_to_wide_form_parquet_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
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
    Get GeoParquet file for a given bounding box in a wide format for multiple types.

    Automatically downloads Overture Maps dataset for a given release and theme/type pairs
    in a concurrent manner and returns a single file as a result with multiple columns based
    on dataset schema.

    Args:
        theme_type_pairs (list[tuple[str, str]]): Pairs of themes and types of the dataset.
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
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
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        Path: Path to the generated GeoParquet file.
    """
    return convert_geometry_to_wide_form_parquet_for_multiple_types(
        theme_type_pairs=theme_type_pairs,
        geometry_filter=box(*bbox),
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        result_file_path=result_file_path,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )


@overload
def convert_bounding_box_to_wide_form_geodataframe(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_bounding_box_to_wide_form_geodataframe(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_bounding_box_to_wide_form_geodataframe(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


def convert_bounding_box_to_wide_form_geodataframe(
    theme: str,
    type: str,
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[int] = None,
    pyarrow_filters: Optional[PYARROW_FILTER] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Get GeoDataFrame for a given bounding box in a wide format.

    Automatically downloads Overture Maps dataset for a given release and theme/type
    in a concurrent manner and returns a single GeoDataFrame as a result with multiple columns based
    on dataset schema.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        include_all_possible_columns (bool, optional): Whether to have always the same list of
            columns in the resulting file. This ensures that always the same set of columns is
            returned for a given release for different regions. This also means, that some columns
            might be all filled with a False value. Defaults to True.
        hierarchy_depth (Optional[int], optional): Depth used to calculate how many hierarchy
            columns should be used to generate the wide form of the data. If None, will use all
            available columns. Defaults to None.
        pyarrow_filters (Optional[PYARROW_FILTER], optional): A pyarrow expression used to filter
            specific theme type pair. Defaults to None.
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Overture Maps features.
    """
    return convert_geometry_to_wide_form_geodataframe(
        theme=theme,
        type=type,
        geometry_filter=box(*bbox),
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )


@overload
def convert_bounding_box_to_wide_form_geodataframe_for_all_types(
    bbox: tuple[float, float, float, float],
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_bounding_box_to_wide_form_geodataframe_for_all_types(
    bbox: tuple[float, float, float, float],
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_bounding_box_to_wide_form_geodataframe_for_all_types(
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


def convert_bounding_box_to_wide_form_geodataframe_for_all_types(
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Get GeoDataFrame for a given bounding box in a wide format for all types.

    Automatically downloads Overture Maps dataset for a given release and all available theme/types
    in a concurrent manner and returns a single GeoDataFrame as a result with multiple columns based
    on dataset schema.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
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
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Overture Maps features.
    """
    return convert_geometry_to_wide_form_geodataframe_for_all_types(
        geometry_filter=box(*bbox),
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )


@overload
def convert_bounding_box_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_bounding_box_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    release: str,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


@overload
def convert_bounding_box_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame: ...


def convert_bounding_box_to_wide_form_geodataframe_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    bbox: tuple[float, float, float, float],
    release: Optional[str] = None,
    *,
    include_all_possible_columns: bool = True,
    hierarchy_depth: Optional[Union[int, list[Optional[int]]]] = None,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
    sort_result: bool = True,
    places_use_primary_category_only: bool = False,
    places_minimal_confidence: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Get GeoDataFrame for a given bounding box in a wide format for multiple types.

    Automatically downloads Overture Maps dataset for a given release and theme/type pairs
    in a concurrent manner and returns a single GeoDataFrame as a result with multiple columns based
    on dataset schema.

    Args:
        theme_type_pairs (list[tuple[str, str]]): Pairs of themes and types of the dataset.
        bbox (tuple[float, float, float, float]): Bounding box used to filter data.
            Order of values: xmin, ymin, xmax, ymax.
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
        compression (str, optional): Compression of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Remember to change compression level together with this parameter.
            Defaults to "zstd".
        compression_level (int, optional): Compression level of the final parquet file.
            Check https://duckdb.org/docs/sql/statements/copy#parquet-options for more info.
            Defaults to 3.
        row_group_size (int, optional): Approximate number of rows per row group in the final
            parquet file. Defaults to 100_000.
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
        places_use_primary_category_only (bool, optional): Whether to use only the primary category
            for places. Defaults to False.
        places_minimal_confidence (float, optional): Minimal confidence level for the places
            dataset. Defaults to 0.75.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Overture Maps features.
    """
    return convert_geometry_to_wide_form_geodataframe_for_multiple_types(
        theme_type_pairs=theme_type_pairs,
        geometry_filter=box(*bbox),
        release=release,
        include_all_possible_columns=include_all_possible_columns,
        hierarchy_depth=hierarchy_depth,
        pyarrow_filters=pyarrow_filters,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
        sort_result=sort_result,
        places_use_primary_category_only=places_use_primary_category_only,
        places_minimal_confidence=places_minimal_confidence,
    )
