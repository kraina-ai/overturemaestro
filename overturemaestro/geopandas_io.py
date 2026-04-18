"""Helper functions for GeoDataFrame IO operations."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from overturemaestro._constants import GEOMETRY_COLUMN, INDEX_COLUMN


def geodataframe_from_parquet(parquet_path: Path) -> gpd.GeoDataFrame:
    """
    Read parquet file as GeoDataframe.

    Loads the file without partitioning columns. Creates an index based on the id column.
    """
    schema = pq.ParquetDataset(parquet_path, partitioning=None).schema
    try:
        gdf = gpd.read_parquet(parquet_path, partitioning=None, columns=schema.names)
    except ValueError:
        df = pd.read_parquet(parquet_path, partitioning=None, columns=schema.names)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df[GEOMETRY_COLUMN], crs=4326))

    if INDEX_COLUMN in gdf.columns:
        gdf.set_index(INDEX_COLUMN, inplace=True)

    return gdf
