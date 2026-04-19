"""Tests internals code for proper coverage in multiprocessing."""

from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from overturemaestro._constants import GEOMETRY_COLUMN, INDEX_COLUMN
from overturemaestro.data_downloader import _download_single_parquet_row_group_multiprocessing
from overturemaestro.functions import convert_geometry_to_parquet
from overturemaestro.geocode import geocode_to_geometry


def test_download_single_parquet_row_group(test_release_version: str) -> None:
    """Test if downloading single parquet row group is working."""
    # load random file from stac catalog
    stac_catalog_response = requests.get(
        f"https://stac.overturemaps.org/{test_release_version}/places/place/collection.json",
        allow_redirects=True,
    ).json()

    first_file_catalog = next(
        link["href"].split("/", 1)[1]
        for link in stac_catalog_response["links"]
        if link["rel"] == "item"
    )

    file_details_response = requests.get(
        f"https://stac.overturemaps.org/{test_release_version}/places/place/{first_file_catalog}",
        allow_redirects=True,
    ).json()

    s3_url = file_details_response["assets"]["aws"]["alternate"]["s3"]["href"][5:]
    print(s3_url)

    result_path = _download_single_parquet_row_group_multiprocessing(
        params={
            "filename": s3_url,  # noqa: E501
            "row_group": 53,
            "theme": "places",
            "type": "place",
            "user_defined_pyarrow_filter": pc.field("confidence") > 0.95,
            "columns_to_download": [INDEX_COLUMN, GEOMETRY_COLUMN, "categories"],
        },
        bbox=(-180, -90, 180, 90),
        working_directory=Path("files"),
    )
    print(result_path)

    assert result_path.exists()
    assert pq.ParquetFile(result_path).num_row_groups > 0
    assert pq.read_metadata(result_path).num_rows > 0


def test_compression(test_release_version: str) -> None:
    """Test if compression works."""
    geometry = geocode_to_geometry("City of London")

    pq_zstd_3 = convert_geometry_to_parquet(
        theme="buildings",
        type="building",
        geometry_filter=geometry,
        compression="zstd",
        compression_level=3,
        ignore_cache=True,
        release=test_release_version,
        result_file_path="files/zstd_3.parquet",
    )
    pq_zstd_22 = convert_geometry_to_parquet(
        theme="buildings",
        type="building",
        geometry_filter=geometry,
        compression="zstd",
        compression_level=22,
        ignore_cache=True,
        release=test_release_version,
        result_file_path="files/zstd_22.parquet",
    )

    assert pq_zstd_3.stat().st_size > pq_zstd_22.stat().st_size


def test_sorting(test_release_version: str) -> None:
    """Test if sorted file is smaller and metadata in both files is equal."""
    geometry = geocode_to_geometry("Monaco")

    unsorted_pq = convert_geometry_to_parquet(
        theme="buildings",
        type="building",
        geometry_filter=geometry,
        sort_result=False,
        ignore_cache=True,
        release=test_release_version,
    )
    sorted_pq = convert_geometry_to_parquet(
        theme="buildings",
        type="building",
        geometry_filter=geometry,
        sort_result=True,
        ignore_cache=True,
        release=test_release_version,
    )

    print(unsorted_pq)
    print(sorted_pq)

    assert pq.read_schema(unsorted_pq).equals(pq.read_schema(sorted_pq))
    assert pq.read_metadata(unsorted_pq).num_rows == pq.read_metadata(sorted_pq).num_rows

    assert unsorted_pq.stat().st_size > sorted_pq.stat().st_size
