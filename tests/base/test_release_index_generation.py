"""Tests related to generating release indexes."""

import shutil
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq

from overturemaestro import convert_bounding_box_to_parquet_for_multiple_types
from overturemaestro._geometry_clustering import decompress_ranges
from overturemaestro.release_index import (
    _generate_release_index,
    download_existing_release_index,
    get_available_theme_type_pairs,
)

SELECTED_CITIES_BOUNDING_BOXES = [
    (16.8073393, 51.0426686, 17.1762192, 51.2100604),  # Wrocław, Poland
    (13.088345, 52.3382448, 13.7611609, 52.6755087),  # Berlin, Germany
    (126.7644328, 37.4285424, 127.1837702, 37.7014794),  # Seoul, South Korea
    (31.2200331, 29.7483062, 31.9090054, 30.3209168),  # Cairo, Egypt
    (-123.2249611, 49.1989306, -123.0232419, 49.3161714),  # Vancouver, Canada
]


def test_generate_release_indexes(test_release_version: str) -> None:
    """Test if generated indexes are working properly."""
    working_directory = "files/local_overture_maps_dataset"

    # Load data from existing Overture dataset using existing index
    download_existing_release_index(release=test_release_version)
    selected_pairs = [
        pair
        for pair in get_available_theme_type_pairs(release=test_release_version)
        if pair[0] in ("base", "buildings", "places", "transportation")
    ]
    for bounding_box in SELECTED_CITIES_BOUNDING_BOXES:
        convert_bounding_box_to_parquet_for_multiple_types(
            theme_type_pairs=selected_pairs,
            bbox=bounding_box,
            release=test_release_version,
            working_directory=working_directory,
        )

    # Generate local index
    test_index_path = Path("test_index")
    if test_index_path.exists():
        shutil.rmtree(test_index_path)

    _generate_release_index(
        release=test_release_version,
        dataset_path=working_directory,
        dataset_fs="local",
        index_location_path=test_index_path,
    )

    # Check if all row_groups are mapped and if all rows are covered by the index
    for theme_value, type_value in selected_pairs:
        index_gdf = gpd.read_parquet(f"test_index/{theme_value}_{type_value}.parquet")
        unique_files = index_gdf["filename"].unique()

        for unique_file in unique_files:
            assert Path(unique_file).exists()

        dataset_index = (
            index_gdf.explode("row_indexes_ranges")
            .groupby(["filename", "row_group"])["row_indexes_ranges"]
            .apply(list)
            .reset_index()
        )

        row_groups_to_check = dataset_index[
            ["filename", "row_group", "row_indexes_ranges"]
        ].to_dict(orient="records")

        for row_group_to_check in row_groups_to_check:
            local_row_group = pq.ParquetFile(row_group_to_check["filename"]).read_row_group(
                row_group_to_check["row_group"]
            )
            data_index = local_row_group["id"].to_pandas().index
            decompressed_range = decompress_ranges(row_group_to_check["row_indexes_ranges"])

            assert data_index.difference(decompressed_range).empty
