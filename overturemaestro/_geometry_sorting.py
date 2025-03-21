"""Module for sorting GeoParquet files."""

import tempfile
from pathlib import Path
from typing import Optional, Union

import pyarrow.parquet as pq

from overturemaestro._constants import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from overturemaestro._duckdb import _set_up_duckdb_connection


def sort_geoparquet_file_by_geometry(
    input_file_path: Path,
    output_file_path: Optional[Path] = None,
    sort_extent: Optional[tuple[float, float, float, float]] = None,
    working_directory: Union[str, Path] = "files",
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
    """
    if output_file_path is None:
        output_file_path = (
            input_file_path.parent / f"{input_file_path.stem}_sorted{input_file_path.suffix}"
        )

    assert input_file_path.resolve().as_posix() != output_file_path.resolve().as_posix()

    Path(working_directory).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)

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

        original_metadata_string = _parquet_schema_metadata_to_duckdb_kv_metadata(input_file_path)

        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{input_file_path}', hive_partitioning=false)
                ORDER BY {order_clause}
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

    return output_file_path


def _parquet_schema_metadata_to_duckdb_kv_metadata(parquet_file_path: Path) -> str:
    def escape_single_quotes(s: str) -> str:
        return s.replace("'", "''")

    kv_pairs = []
    for key, value in pq.read_metadata(parquet_file_path).metadata.items():
        escaped_key = escape_single_quotes(key.decode())
        escaped_value = escape_single_quotes(value.decode())
        kv_pairs.append(f"'{escaped_key}': '{escaped_value}'")

    return "{ " + ", ".join(kv_pairs) + " }"
