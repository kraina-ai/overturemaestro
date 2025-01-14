"""
Advanced functions.

This module contains dedicated functions for specific use cases.
"""

# from overturemaestro.advanced_functions.poi import (
#     convert_bounding_box_to_pois_geodataframe,
#     convert_bounding_box_to_pois_parquet,
#     convert_geometry_to_pois_geodataframe,
#     convert_geometry_to_pois_parquet,
# )
# from overturemaestro.advanced_functions.transportation import (
#     convert_bounding_box_to_roads_geodataframe,
#     convert_bounding_box_to_roads_parquet,
#     convert_geometry_to_roads_geodataframe,
#     convert_geometry_to_roads_parquet,
# )
from overturemaestro.advanced_functions.functions import (
    convert_bounding_box_to_wide_form_geodataframe,
    convert_bounding_box_to_wide_form_geodataframe_for_all_types,
    convert_bounding_box_to_wide_form_geodataframe_for_multiple_types,
    convert_bounding_box_to_wide_form_parquet,
    convert_bounding_box_to_wide_form_parquet_for_all_types,
    convert_bounding_box_to_wide_form_parquet_for_multiple_types,
    convert_geometry_to_wide_form_geodataframe,
    convert_geometry_to_wide_form_geodataframe_for_all_types,
    convert_geometry_to_wide_form_geodataframe_for_multiple_types,
    convert_geometry_to_wide_form_parquet,
    convert_geometry_to_wide_form_parquet_for_all_types,
    convert_geometry_to_wide_form_parquet_for_multiple_types,
)

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

# __all__ = [
#     "convert_bounding_box_to_buildings_geodataframe",
#     "convert_bounding_box_to_buildings_parquet",
#     "convert_bounding_box_to_pois_geodataframe",
#     "convert_bounding_box_to_pois_parquet",
#     "convert_bounding_box_to_roads_geodataframe",
#     "convert_bounding_box_to_roads_parquet",
#     "convert_bounding_box_to_wide_form_geodataframe",
#     "convert_bounding_box_to_wide_form_parquet",
#     "convert_geometry_to_buildings_geodataframe",
#     "convert_geometry_to_buildings_parquet",
#     "convert_geometry_to_pois_geodataframe",
#     "convert_geometry_to_pois_parquet",
#     "convert_geometry_to_roads_geodataframe",
#     "convert_geometry_to_roads_parquet",
#     "convert_geometry_to_wide_form_geodataframe",
#     "convert_geometry_to_wide_form_parquet",
# ]
