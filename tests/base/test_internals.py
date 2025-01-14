"""Tests internals code for proper coverage in multiprocessing."""

from pathlib import Path

import pyarrow.compute as pc

from overturemaestro.data_downloader import _download_single_parquet_row_group_multiprocessing

# TODO: add test for checking if metadata is ok


def test_download_single_parquet_row_group() -> None:
    """Test if downloading single parquet row group is working."""
    _download_single_parquet_row_group_multiprocessing(
        params={
            "filename": "overturemaps-us-west-2/release/2024-08-20.0/theme=places/type=place/part-00002-93118862-ebe9-4b31-8277-1a87d792bd5d-c000.zstd.parquet",  # noqa: E501
            "row_group": 53,
            "row_indexes_ranges": [
                [0, 1904],
                [1906, 2085],
                [2200, 2200],
                [2202, 2212],
                [2224, 2246],
                [2251, 2252],
                [2260, 2260],
                [2305, 2305],
                [12192, 32481],
                [32484, 32485],
                [32511, 33159],
                [33162, 33188],
                [33197, 37371],
                [37375, 37460],
                [37462, 37462],
                [37554, 37562],
                [37592, 37618],
                [37621, 37622],
                [38789, 38791],
                [38793, 38793],
                [38796, 38796],
                [38798, 38798],
                [40055, 40056],
                [40058, 40060],
                [40070, 40073],
            ],
            "theme": "places",
            "type": "place",
            "user_defined_pyarrow_filter": pc.field("confidence") > 0.95,
            "columns_to_download": ["id", "geometry", "categories"],
        },
        bbox=(7.416486207767861, 43.7310867041912, 7.421931388477276, 43.73370705597216),
        working_directory=Path("files"),
    )
