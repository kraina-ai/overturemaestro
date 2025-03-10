"""Module for sorting GeoParquet files."""

import tempfile
from pathlib import Path
from typing import Optional, Union

from overturemaestro._constants import PARQUET_COMPRESSION, PARQUET_ROW_GROUP_SIZE
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

        relation = connection.sql(
            f"""
            SELECT *
            FROM read_parquet('{input_file_path}', hive_partitioning=false)
            ORDER BY {order_clause}
            """
        )

        relation.to_parquet(
            str(output_file_path),
            row_group_size=PARQUET_ROW_GROUP_SIZE,
            compression=PARQUET_COMPRESSION,
        )

        connection.close()

    return output_file_path
