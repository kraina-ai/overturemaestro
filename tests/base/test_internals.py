"""Tests internals code for proper coverage in multiprocessing."""

from pathlib import Path
from urllib.parse import urljoin

import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from overturemaestro._constants import GEOMETRY_COLUMN, INDEX_COLUMN
from overturemaestro.data_downloader import _download_single_parquet_row_group_multiprocessing


def test_download_single_parquet_row_group(test_release_version: str) -> None:
    """Test if downloading single parquet row group is working."""
    # load random file from stac catalog
    collection_url = (
        f"https://stac.overturemaps.org/{test_release_version}/places/place/collection.json"
    )
    stac_catalog_response = requests.get(collection_url, allow_redirects=True, timeout=30).json()

    # Item links can be relative or absolute depending on the release
    first_file_catalog_url = next(
        urljoin(collection_url, link["href"])
        for link in stac_catalog_response["links"]
        if link["rel"] == "item"
    )

    file_details_response = requests.get(
        first_file_catalog_url, allow_redirects=True, timeout=30
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
            "columns_to_download": [INDEX_COLUMN, GEOMETRY_COLUMN, "taxonomy"],
        },
        bbox=(-180, -90, 180, 90),
        working_directory=Path("files"),
    )
    print(result_path)

    assert result_path.exists()
    assert pq.ParquetFile(result_path).num_row_groups > 0
    assert pq.read_metadata(result_path).num_rows > 0
