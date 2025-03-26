"""Helper functions for GeoDataFrame IO operations."""

from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq

from overturemaestro._constants import INDEX_COLUMN


def geodataframe_from_parquet(parquet_path: Path) -> gpd.GeoDataFrame:
    """
    Read parquet file as GeoDataframe.

    Loads the file without partitioning columns. Creates an index based on the id column.
    """
    schema = pq.ParquetDataset(parquet_path, partitioning=None).schema
    gdf = gpd.read_parquet(parquet_path, partitioning=None, columns=schema.names)
    if INDEX_COLUMN in gdf.columns:
        gdf.set_index(INDEX_COLUMN, inplace=True)

    return gdf
