"""Module for sorting GeoParquet files."""

from pathlib import Path
from typing import Optional

import duckdb

from overturemaestro.data_downloader import PARQUET_COMPRESSION, PARQUET_ROW_GROUP_SIZE


def sort_geoparquet_file_by_geometry(
    input_file_path: Path,
    output_file_path: Optional[Path] = None,
    sort_extent: Optional[tuple[float, float, float, float]] = None,
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
    """
    if output_file_path is None:
        output_file_path = (
            input_file_path.parent / f"{input_file_path.stem}_sorted{input_file_path.suffix}"
        )

    # https://medium.com/radiant-earth-insights/using-duckdbs-hilbert-function-with-geop-8ebc9137fb8a
    if sort_extent is None:
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
        order_clause = f"""
        ST_Within(ST_Extent(geometry), {extent_box_clause}),
        ST_Hilbert(
            geometry,
            {extent_box_clause}
        )
        """

    relation = duckdb.sql(
        f"""
        SELECT *
        FROM read_parquet('{input_file_path}', hive_partitioning=false)
        ORDER BY {order_clause}
        """
    )

    print(relation.sql_query())

    relation.to_parquet(
        str(output_file_path),
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        compression=PARQUET_COMPRESSION,
    )

    return output_file_path
