"""
OvertureMaestro.

OvertureMaestro is a Python library used for downloading Overture Maps datasets with quality of life
features.
"""

from overturemaestro.functions import (
    convert_bounding_box_to_geodataframe,
    convert_bounding_box_to_geodataframe_for_multiple_types,
    convert_bounding_box_to_parquet,
    convert_bounding_box_to_parquet_for_multiple_types,
    convert_geometry_to_geodataframe,
    convert_geometry_to_geodataframe_for_multiple_types,
    convert_geometry_to_parquet,
    convert_geometry_to_parquet_for_multiple_types,
)
from overturemaestro.release_index import get_available_theme_type_pairs

__app_name__ = "OvertureMaestro"
__version__ = "0.0.3"

__all__ = [
    "convert_bounding_box_to_geodataframe",
    "convert_bounding_box_to_geodataframe_for_multiple_types",
    "convert_bounding_box_to_parquet",
    "convert_bounding_box_to_parquet_for_multiple_types",
    "convert_geometry_to_geodataframe",
    "convert_geometry_to_geodataframe_for_multiple_types",
    "convert_geometry_to_parquet",
    "convert_geometry_to_parquet_for_multiple_types",
    "get_available_theme_type_pairs",
]
