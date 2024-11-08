"""Functions for retrieving Overture Maps features in a wide form."""

# TODO: replace with dedicated DuckDB functions - POI can have multiple values at once
THEME_TYPE_CLASSIFICATION = {
    ("base", "infrastructure"): ["subtype", "class"],
    ("base", "land"): ["subtype", "class"],
    ("base", "land_cover"): ["subtype", "class"],
    ("base", "land_use"): ["subtype", "class"],
    ("base", "water"): ["subtype", "class"],
    ("transportation", "segment"): ["subtype"],
    ("places", "place"): ["categories.primary", "categories.primary"],
    ("buildings", "building"): ["subtype", "class"],
}

# convert_bounding_box_to_wide_form_geodataframe,
# convert_bounding_box_to_wide_form_parquet,
# convert_geometry_to_wide_form_geodataframe,
# convert_geometry_to_wide_form_parquet,
