"""Module for sorting GeoParquet files."""

import multiprocessing
import tempfile
from collections.abc import Callable
from functools import partial
from math import ceil
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Optional, Union

import duckdb
import psutil
import pyarrow.parquet as pq
from rich import print as rprint

from overturemaestro._constants import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from overturemaestro._duckdb import _set_up_duckdb_connection

# from overturemaestro._parquet_compression import compress_parquet_with_duckdb

if TYPE_CHECKING:  # pragma: no cover
    from overturemaestro._rich_progress import VERBOSITY_MODE

MEMORY_1GB = 1024**3


def compress_parquet_with_duckdb(
    input_file_path: Path,
    output_file_path: Path,
    working_directory: Union[str, Path] = "files",
    parquet_metadata: Optional[pq.FileMetaData] = None,
    verbosity_mode: "VERBOSITY_MODE" = "transient",
) -> Path:
    """
    Compresses a GeoParquet file while keeping its metadata.

    Args:
        input_file_path (Path): Input GeoParquet file path.
        output_file_path (Path): Output GeoParquet file path.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        parquet_metadata (Optional[pq.FileMetaData], optional): GeoParquet file metadata used to
            copy. If not provided, will load the metadata from the input file. Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".
    """
    assert input_file_path.resolve().as_posix() != output_file_path.resolve().as_posix()

    Path(working_directory).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)

        connection = _set_up_duckdb_connection(tmp_dir_path, preserve_insertion_order=True)

        connection.execute("SET enable_geoparquet_conversion = false;")

        original_metadata_string = _parquet_schema_metadata_to_duckdb_kv_metadata(
            parquet_metadata or pq.read_metadata(input_file_path)
        )

        _run_query_with_memory_limit(
            tmp_dir_path=tmp_dir_path,
            verbosity_mode=verbosity_mode,
            current_memory_limit=None,
            function=_compress_with_memory_limit,
            args=(input_file_path, output_file_path, original_metadata_string),
        )

    return output_file_path


def sort_geoparquet_file_by_geometry(
    input_file_path: Path,
    output_file_path: Optional[Path] = None,
    sort_extent: Optional[tuple[float, float, float, float]] = None,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
) -> Path:
    """
    Sorts a GeoParquet file by the geometry column.

    Args:
        input_file_path (Path): Input GeoParquet file path.
        output_file_path (Optional[Path], optional): Output GeoParquet file path.
            If not provided, will generate file name based on input file name with
            `_sorted` suffix. Defaults to None.
        sort_extent (Optional[tuple[float, float, float, float]], optional): Extent to use
            in the ST_Hilbert function. If not, will calculate extent from the
            geometries in the file. Defaults to None.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".
    """
    if output_file_path is None:
        output_file_path = (
            input_file_path.parent / f"{input_file_path.stem}_sorted{input_file_path.suffix}"
        )

    assert input_file_path.resolve().as_posix() != output_file_path.resolve().as_posix()

    Path(working_directory).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)
        order_file_path = tmp_dir_path / "order_file.parquet"

        current_memory_limit = _run_query_with_memory_limit(
            tmp_dir_path=tmp_dir_path,
            verbosity_mode=verbosity_mode,
            current_memory_limit=None,
            function=_sort_with_memory_limit,
            args=(input_file_path, order_file_path, sort_extent),
        )

        original_metadata_string = _parquet_schema_metadata_to_duckdb_kv_metadata(
            pq.read_metadata(input_file_path)
        )

        _run_query_with_memory_limit(
            tmp_dir_path=tmp_dir_path,
            verbosity_mode=verbosity_mode,
            current_memory_limit=current_memory_limit,
            function=_compress_with_memory_limit,
            args=(order_file_path, output_file_path, original_metadata_string),
        )

    return output_file_path


def _compress_with_memory_limit(
    input_file_path: Path,
    output_file_path: Path,
    original_metadata_string: str,
    current_memory_limit: int,
    tmp_dir_path: Path,
) -> None:
    connection = _set_up_duckdb_connection(tmp_dir_path, preserve_insertion_order=True)

    connection.execute("SET enable_geoparquet_conversion = false;")
    connection.execute(f"SET memory_limit = '{current_memory_limit}GB';")

    connection.execute(
        f"""
        COPY (
            SELECT original_data.*
            FROM read_parquet('{input_file_path}', hive_partitioning=false) original_data
        ) TO '{output_file_path}' (
            FORMAT parquet,
            COMPRESSION {PARQUET_COMPRESSION},
            COMPRESSION_LEVEL {PARQUET_COMPRESSION_LEVEL},
            ROW_GROUP_SIZE {PARQUET_ROW_GROUP_SIZE},
            KV_METADATA {original_metadata_string}
        );
        """
    )

    connection.close()

    # db_path = Path(tmp_dir_path) / "db.duckdb"
    # while db_path.exists():
    #     try:
    #         db_path.unlink(missing_ok=True)
    #     except PermissionError:
    #         sleep(0.1)


def _sort_with_memory_limit(
    input_file_path: Path,
    output_file_path: Path,
    sort_extent: Optional[tuple[float, float, float, float]],
    current_memory_limit: int,
    tmp_dir_path: Path,
) -> None:
    connection = _set_up_duckdb_connection(tmp_dir_path, preserve_insertion_order=True)

    struct_type = "::STRUCT(min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE)"
    connection.sql(
        f"""
        CREATE OR REPLACE MACRO bbox_within(a, b) AS
        (
            (a{struct_type}).min_x >= (b{struct_type}).min_x and
            (a{struct_type}).max_x <= (b{struct_type}).max_x
        )
        and
        (
            (a{struct_type}).min_y >= (b{struct_type}).min_y and
            (a{struct_type}).max_y <= (b{struct_type}).max_y
        );
        """
    )

    # https://medium.com/radiant-earth-insights/using-duckdbs-hilbert-function-with-geop-8ebc9137fb8a
    if sort_extent is None:
        # Calculate extent from the geometries in the file
        order_clause = f"""
        ST_Hilbert(
            geometry,
            (
                SELECT ST_Extent(ST_Extent_Agg(geometry))::BOX_2D
                FROM read_parquet('{input_file_path}', hive_partitioning=false)
            )
        )
        """
    else:
        extent_box_clause = f"""
        {{
            min_x: {sort_extent[0]},
            min_y: {sort_extent[1]},
            max_x: {sort_extent[2]},
            max_y: {sort_extent[3]}
        }}::BOX_2D
        """
        # Keep geometries within the extent first,
        # and geometries that are bigger than the extent last (like administrative boundaries)

        # Then sort by Hilbert curve but readjust the extent to all geometries that
        # are not fully within the extent, but also not bigger than the extent overall.
        order_clause = f"""
        bbox_within(({extent_box_clause}), ST_Extent(geometry)),
        ST_Hilbert(
            geometry,
            (
                SELECT ST_Extent(ST_Extent_Agg(geometry))::BOX_2D
                FROM read_parquet('{input_file_path}', hive_partitioning=false)
                WHERE NOT bbox_within(
                    ({extent_box_clause}),
                    ST_Extent(geometry)
                )
            )
        )
        """

    connection.execute(f"SET memory_limit = '{current_memory_limit}GB';")

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{input_file_path}', hive_partitioning=false)
            ORDER BY {order_clause}
        ) TO '{output_file_path}' (
            FORMAT parquet
        );
        """
    )

    connection.close()

    # db_path = Path(tmp_dir_path) / "db.duckdb"
    # while db_path.exists():
    #     try:
    #         db_path.unlink(missing_ok=True)
    #     except PermissionError:
    #         sleep(0.1)


def _run_query_with_memory_limit(
    tmp_dir_path: Path,
    verbosity_mode: "VERBOSITY_MODE",
    current_memory_limit: Optional[int],
    function: Callable[..., None],
    args: Any,
) -> int:
    current_memory_limit = current_memory_limit or ceil(
        psutil.virtual_memory().total * 0.8 / MEMORY_1GB
    )

    while current_memory_limit >= 1:
        try:
            with (
                multiprocessing.get_context("spawn").Pool() as pool,
                tempfile.TemporaryDirectory(dir=Path(tmp_dir_path).resolve()) as tmp_dir_name,
            ):
                nested_tmp_dir_path = Path(tmp_dir_name)
                r = pool.apply_async(
                    func=partial(
                        function,
                        current_memory_limit=current_memory_limit,
                        tmp_dir_path=nested_tmp_dir_path,
                    ),
                    args=args,
                )
                actual_memory = psutil.virtual_memory()
                percentage_threshold = 95
                if (actual_memory.total * 0.05) > MEMORY_1GB:
                    percentage_threshold = (
                        100 * (actual_memory.total - MEMORY_1GB) / actual_memory.total
                    )
                while not r.ready():
                    actual_memory = psutil.virtual_memory()
                    if actual_memory.percent > percentage_threshold:
                        raise MemoryError()

                    sleep(0.5)
                r.get()
            return current_memory_limit
        except (duckdb.OutOfMemoryException, MemoryError) as ex:
            if current_memory_limit == 1:
                raise RuntimeError(
                    "Not enough memory to run the ordering query. Please rerun without sorting."
                ) from ex

            current_memory_limit = ceil(current_memory_limit / 2)

            if not verbosity_mode == "silent":
                rprint(
                    f"Encountered {ex.__class__.__name__} during operation."
                    " Retrying with lower memory limit"
                    f" ({current_memory_limit}GB)."
                )

    raise RuntimeError("Not enough memory to run the ordering query. Please rerun without sorting.")


def _parquet_schema_metadata_to_duckdb_kv_metadata(parquet_file_metadata: pq.FileMetaData) -> str:
    def escape_single_quotes(s: str) -> str:
        return s.replace("'", "''")

    kv_pairs = []
    for key, value in parquet_file_metadata.metadata.items():
        escaped_key = escape_single_quotes(key.decode())
        escaped_value = escape_single_quotes(value.decode())
        kv_pairs.append(f"'{escaped_key}': '{escaped_value}'")

    return "{ " + ", ".join(kv_pairs) + " }"
