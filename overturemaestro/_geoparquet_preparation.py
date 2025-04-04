"""Module for sorting GeoParquet files."""

import multiprocessing
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import ceil
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Optional, Union

import duckdb
import polars as pl
import psutil
import pyarrow.parquet as pq
from rich import print as rprint

from overturemaestro._constants import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from overturemaestro._duckdb import _set_up_duckdb_connection

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

        original_metadata_string = _parquet_schema_metadata_to_duckdb_kv_metadata(
            parquet_metadata or pq.read_metadata(input_file_path)
        )

        _run_query_with_memory_limit(
            tmp_dir_path=tmp_dir_path,
            verbosity_mode=verbosity_mode,
            current_memory_gb_limit=None,
            current_threads_limit=None,
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
        order_dir_path = tmp_dir_path / "ordered"
        order_dir_path.mkdir(parents=True, exist_ok=True)

        _sort_with_multiprocessing(
            input_file_path=input_file_path,
            output_dir_path=order_dir_path,
            sort_extent=sort_extent,
            tmp_dir_path=tmp_dir_path,
        )

        original_metadata_string = _parquet_schema_metadata_to_duckdb_kv_metadata(
            pq.read_metadata(input_file_path)
        )

        input_file_path.unlink()

        order_files = sorted(order_dir_path.glob("*.parquet"), key=lambda x: int(x.stem))

        _run_query_with_memory_limit(
            tmp_dir_path=tmp_dir_path,
            verbosity_mode=verbosity_mode,
            current_memory_gb_limit=None,
            current_threads_limit=None,
            function=_compress_with_memory_limit,
            args=(order_files, output_file_path, original_metadata_string),
        )

    return output_file_path


def _compress_with_memory_limit(
    input_file_path: Union[list[Path], Path],
    output_file_path: Path,
    original_metadata_string: str,
    current_memory_gb_limit: float,
    current_threads_limit: int,
    tmp_dir_path: Path,
) -> None:
    connection = _set_up_duckdb_connection(tmp_dir_path, preserve_insertion_order=True)

    connection.execute("SET enable_geoparquet_conversion = false;")
    connection.execute(f"SET memory_limit = '{current_memory_gb_limit}GB';")
    connection.execute(f"SET threads = {current_threads_limit};")

    if isinstance(input_file_path, Path):
        sql_input_str = f"'{input_file_path}'"
    else:
        sql_input_str = f"[{', '.join([f"'{path}'" for path in input_file_path])}]"

    connection.execute(
        f"""
        COPY (
            SELECT original_data.*
            FROM read_parquet({sql_input_str}, hive_partitioning=false) original_data
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


def _sort_with_multiprocessing(
    input_file_path: Path,
    output_dir_path: Path,
    sort_extent: Optional[tuple[float, float, float, float]],
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
                WHERE NOT bbox_within(({extent_box_clause}), ST_Extent(geometry))
            )
        )
        """

    relation = connection.sql(
        f"""
        SELECT file_row_number, row_number() OVER (ORDER BY {order_clause}) as order_id
        FROM read_parquet('{input_file_path}', hive_partitioning=false, file_row_number=true)
        """
    )

    order_file_path = tmp_dir_path / "order_index.parquet"

    relation.to_parquet(
        str(order_file_path),
        row_group_size=100_000,
        compression=PARQUET_COMPRESSION,
    )

    connection.close()

    # Calculate mapping of ranges which file row ids exist in each row group
    original_file_row_group_mapping = _calculate_row_group_mapping(input_file_path)

    # Order each row group from the ordered index in separate processes by reading
    # selected row groups from the original file
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as ex:
        fn = partial(
            _order_single_row_group,
            output_dir_path=output_dir_path,
            order_file_path=order_file_path,
            original_file_path=input_file_path,
            original_file_row_group_mapping=original_file_row_group_mapping,
        )
        ex.map(fn, list(range(pq.read_metadata(order_file_path).num_row_groups)), chunksize=1)


def _run_query_with_memory_limit(
    tmp_dir_path: Path,
    verbosity_mode: "VERBOSITY_MODE",
    current_memory_gb_limit: Optional[float],
    current_threads_limit: Optional[int],
    function: Callable[..., None],
    args: Any,
) -> tuple[float, int]:
    current_memory_gb_limit = current_memory_gb_limit or ceil(
        psutil.virtual_memory().total / MEMORY_1GB
    )
    current_threads_limit = current_threads_limit or multiprocessing.cpu_count()

    while current_memory_gb_limit > 0:
        try:
            with (
                tempfile.TemporaryDirectory(dir=Path(tmp_dir_path).resolve()) as tmp_dir_name,
                multiprocessing.get_context("spawn").Pool() as pool,
            ):
                nested_tmp_dir_path = Path(tmp_dir_name)
                r = pool.apply_async(
                    func=partial(
                        function,
                        current_memory_gb_limit=current_memory_gb_limit,
                        current_threads_limit=current_threads_limit,
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
            return current_memory_gb_limit, current_threads_limit
        except (duckdb.OutOfMemoryException, MemoryError) as ex:
            if current_memory_gb_limit < 1:
                raise RuntimeError(
                    "Not enough memory to run the ordering query. Please rerun without sorting."
                ) from ex

            if current_memory_gb_limit == 1:
                current_memory_gb_limit /= 2
            else:
                current_memory_gb_limit = ceil(current_memory_gb_limit / 2)

            current_threads_limit = ceil(current_threads_limit / 2)

            if not verbosity_mode == "silent":
                rprint(
                    f"Encountered {ex.__class__.__name__} during operation."
                    " Retrying with lower number of resources"
                    f" ({current_memory_gb_limit:.2f}GB, {current_threads_limit} threads)."
                )

    raise RuntimeError("Not enough memory to run the ordering query. Please rerun without sorting.")


def _order_single_row_group(
    row_group_id: int,
    output_dir_path: Path,
    order_file_path: Path,
    original_file_path: Path,
    original_file_row_group_mapping: dict[int, tuple[int, int]],
) -> None:
    # Calculate row_groups and local indexes withing those row_groups
    ordering_row_group_extended = (
        pl.from_arrow(pq.ParquetFile(order_file_path).read_row_group(row_group_id))
        .with_columns(
            # Assign row group based on file row number using original file mapping
            pl.col("file_row_number")
            .map_elements(
                lambda row_number: next(
                    row_group_id
                    for row_group_id, (
                        start_row_number,
                        end_row_number,
                    ) in original_file_row_group_mapping.items()
                    if start_row_number <= row_number <= end_row_number
                ),
                return_dtype=pl.Int64(),
            )
            .alias("row_group_id")
        )
        .with_columns(
            # Assign local row index within the row group using
            # original file mapping and total row number
            pl.struct(pl.col("file_row_number"), pl.col("row_group_id"))
            .map_elements(
                lambda struct: struct["file_row_number"]
                - original_file_row_group_mapping[struct["row_group_id"]][0],
                return_dtype=pl.Int64(),
            )
            .alias("local_index")
        )
    )

    # Example of ordered file mapping:
    # order_id, file_row_number, row_group_id, local_index
    # 1,        1,               1,            0
    # 2,        5,               1,            5
    # 3,        15,              2,            1
    # 4,        3,               1,            3

    # Read all expected rows from each row group at once to avoid multiple reads
    # Group matching consecutive row group and save local indexes
    # indexes_to_read_per_row_group = {1: [0, 5, 3], 2: [1]}
    # reshuffled_indexes_to_read = [(1, [0, 5]), (2, [1]), (1, [3])]

    # Dictionary with row group id and a list of local indices to read from each row group.
    indexes_to_read_per_row_group: dict[int, list[int]] = {}
    # Grouped list of local indexes to read per row group in order
    reshuffled_indexes_to_read: list[tuple[int, list[int]]] = []
    # Cache objects to keep track of each group withing multiple row groups
    current_index_per_row_group: dict[int, int] = {}
    current_reshuffled_indexes_group: list[int] = []
    current_rg_id = ordering_row_group_extended["row_group_id"][0]

    # Iterate rows in order
    for rg_id, local_index in ordering_row_group_extended[
        ["row_group_id", "local_index"]
    ].iter_rows():
        if rg_id not in indexes_to_read_per_row_group:
            indexes_to_read_per_row_group[rg_id] = []
            current_index_per_row_group[rg_id] = 0

        indexes_to_read_per_row_group[rg_id].append(local_index)

        if rg_id != current_rg_id:
            reshuffled_indexes_to_read.append((current_rg_id, current_reshuffled_indexes_group))
            current_rg_id = rg_id
            current_reshuffled_indexes_group = [current_index_per_row_group[rg_id]]
        else:
            current_reshuffled_indexes_group.append(current_index_per_row_group[rg_id])

        current_index_per_row_group[rg_id] += 1

    if current_reshuffled_indexes_group:
        reshuffled_indexes_to_read.append((current_rg_id, current_reshuffled_indexes_group))

    # Read expected rows per row group
    read_tables_per_row_group = {
        rg_id: pq.ParquetFile(original_file_path).read_row_group(rg_id).take(local_rows_ids)
        for rg_id, local_rows_ids in indexes_to_read_per_row_group.items()
    }

    schema = pq.read_schema(original_file_path)
    with pq.ParquetWriter(output_dir_path / f"{row_group_id}.parquet", schema=schema) as writer:
        # Read rows from each read row group using reshuffled local indexes
        for rg_id, reshuffled_indexes in reshuffled_indexes_to_read:
            writer.write(read_tables_per_row_group[rg_id].take(reshuffled_indexes))


def _parquet_schema_metadata_to_duckdb_kv_metadata(parquet_file_metadata: pq.FileMetaData) -> str:
    def escape_single_quotes(s: str) -> str:
        return s.replace("'", "''")

    kv_pairs = []
    for key, value in parquet_file_metadata.metadata.items():
        escaped_key = escape_single_quotes(key.decode())
        escaped_value = escape_single_quotes(value.decode())
        kv_pairs.append(f"'{escaped_key}': '{escaped_value}'")

    return "{ " + ", ".join(kv_pairs) + " }"


def _calculate_row_group_mapping(file_path: Path) -> dict[int, tuple[int, int]]:
    pq_f = pq.ParquetFile(file_path)

    mapping = {}
    total_rows = 0
    for i in range(pq_f.num_row_groups):
        start_index = total_rows
        rows_in_row_group = pq_f.metadata.row_group(i).num_rows
        total_rows += rows_in_row_group
        end_index = total_rows - 1
        mapping[i] = (start_index, end_index)

    return mapping
