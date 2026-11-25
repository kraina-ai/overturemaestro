"""Tests internals code for proper coverage in multiprocessing."""

from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import requests
from shapely import MultiPolygon, Point, box
from shapely.geometry.base import BaseGeometry

from overturemaestro._constants import GEOMETRY_COLUMN, INDEX_COLUMN
from overturemaestro.data_downloader import (
    _download_single_parquet_row_group_multiprocessing,
    _filter_downloaded_files,
    _generate_empty_file,
)


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


def test_generate_empty_file_with_selected_columns(
    test_release_version: str, tmp_path: Path
) -> None:
    """Test if an empty file has the same columns and metadata as downloaded files."""
    result_path = _generate_empty_file(
        theme="places",
        type="place",
        release=test_release_version,
        columns_to_download=["confidence"],
        working_directory=tmp_path,
        verbosity_mode="silent",
    )

    schema = pq.read_schema(result_path)
    assert set(schema.names) == {INDEX_COLUMN, GEOMETRY_COLUMN, "confidence"}
    assert schema.metadata[b"_theme"] == b"places"
    assert schema.metadata[b"_type"] == b"place"
    assert pq.read_metadata(result_path).num_rows == 0


@pytest.mark.parametrize(
    "geometry_filter", [box(1, 1, 2, 2), Point(1.5, 1.5)], ids=["rectangle", "point"]
)  # type: ignore
def test_filter_by_geometry_not_bounds(geometry_filter: BaseGeometry, tmp_path: Path) -> None:
    """Test if features intersecting the filter only with their bounds are removed."""
    features = gpd.GeoDataFrame(
        {INDEX_COLUMN: ["intersecting", "around"]},
        geometry=[box(0, 0, 3, 3), MultiPolygon([box(-180, 0, -179, 3), box(179, 0, 180, 3)])],
        crs=4326,
    )
    features_path = tmp_path / "features.parquet"
    pq.write_table(pa.table(features.to_arrow(index=False)), features_path)

    result_paths = _filter_downloaded_files(
        downloaded_parquet_files=[features_path],
        theme_type_pairs=[("divisions", "division_area")],
        geometry_filter=geometry_filter,
        working_directory=tmp_path,
        verbosity_mode="silent",
        max_workers=1,
    )

    assert [pq.read_table(path)[INDEX_COLUMN].to_pylist() for path in result_paths] == [
        ["intersecting"]
    ]


def test_filter_without_downloaded_files(tmp_path: Path) -> None:
    """Test if lack of downloaded files is properly handled for any geometry."""
    result_paths = _filter_downloaded_files(
        downloaded_parquet_files=[],
        theme_type_pairs=[("divisions", "division_area")],
        geometry_filter=Point(0, 0).buffer(1),
        working_directory=tmp_path,
        verbosity_mode="silent",
        max_workers=1,
    )

    assert not result_paths
