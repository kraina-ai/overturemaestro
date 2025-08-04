"""Constants used across the project."""


from rq_geo_toolkit.constants import (
    GEOMETRY_COLUMN,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)

INDEX_COLUMN = "id"

__all__ = [
    "GEOMETRY_COLUMN",
    "INDEX_COLUMN",
    "PARQUET_COMPRESSION",
    "PARQUET_COMPRESSION_LEVEL",
    "PARQUET_ROW_GROUP_SIZE",
]
