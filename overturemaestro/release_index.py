"""
Release index cache utilities.

Functions used to load cached release index, download it from a dedicated repository or to generate
it on demand.
"""

import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Union

import fsspec.implementations.http
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.fs as fs
from pooch import file_hash, retrieve
from pooch import get_logger as get_pooch_logger
from rich import print as rprint
from shapely import box
from shapely.geometry.base import BaseGeometry

from overturemaestro._geometry_clustering import calculate_row_group_bounding_box
from overturemaestro._parquet_multiprocessing import map_parquet_dataset
from overturemaestro._rich_progress import TrackProgressBar

__all__ = [
    "download_existing_release_index",
    "generate_release_index",
    "get_available_theme_type_pairs",
    "load_release_index",
    "load_release_indexes",
]

# This is the release with bbox with proper naming convention
MINIMAL_SUPPORTED_RELEASE_VERSION = "2024-04-16-beta.0"

# Dedicated GitHub repository with precalculated indexes
LFS_DIRECTORY_URL = (
    "https://raw.githubusercontent.com/kraina-ai/overturemaps-releases-indexes/main/"
)


class ReleaseVersionNotSupportedError(ValueError): ...  # noqa: D101


def load_release_indexes(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load multiple release indexes as a GeoDataFrame.

    Args:
        release (str): Release version.
        theme_type_pairs (list[tuple[str, str]]): Pairs of themes and types of the dataset.
        geometry_filter (Optional[BaseGeometry], optional): Geometry to pre-filter resulting rows.
            Defaults to None.
        remote_index (bool, optional): Avoid downloading the index and stream it from remote source.
            Defaults to False.

    Returns:
        gpd.GeoDataFrame: Index with bounding boxes for each row group for each parquet file.
    """
    return gpd.pd.concat(
        load_release_index(release, theme_value, type_value, geometry_filter, remote_index)
        for theme_value, type_value in theme_type_pairs
    )


def load_release_index(
    release: str,
    theme: str,
    type: str,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load release index as a GeoDataFrame.

    Args:
        release (str): Release version.
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        geometry_filter (Optional[BaseGeometry], optional): Geometry to pre-filter resulting rows.
            Defaults to None.
        remote_index (bool, optional): Avoid downloading the index and stream it from remote source.
            Defaults to False.

    Returns:
        gpd.GeoDataFrame: Index with bounding boxes for each row group for each parquet file.
    """
    _check_release_version(release)
    cache_directory = _get_release_cache_directory(release)
    index_file_path: Union[Path, str] = cache_directory / _get_index_file_name(theme, type)

    file_exists = Path(index_file_path).exists()
    filesystem = None

    if not file_exists:
        if remote_index:
            filesystem = fsspec.implementations.http.HTTPFileSystem()
            index_file_path = LFS_DIRECTORY_URL + str(index_file_path)
        else:
            # Download or generate the index if cannot be downloaded
            download_existing_release_index(release) or generate_release_index(release)

    if geometry_filter is None:
        return gpd.read_parquet(path=index_file_path, filesystem=filesystem)

    xmin, ymin, xmax, ymax = geometry_filter.bounds

    df = gpd.read_parquet(
        path=index_file_path, bbox=(xmin, ymin, xmax, ymax), filesystem=filesystem
    )
    df = df[df.intersects(geometry_filter)]
    return df


def get_available_theme_type_pairs(release: str) -> list[tuple[str, str]]:
    """
    Get a list of available theme and type objects for a given release.

    Args:
        release (str): Release version.

    Returns:
        list[tuple[str, str]]: List of theme and type pairs.
    """
    _check_release_version(release)

    cache_directory = _get_release_cache_directory(release)
    release_index_path = cache_directory / "release_index_content.json"
    if not release_index_path.exists():
        raise FileNotFoundError(
            f"Index for release {release} isn't cached locally. "
            "Please download or generate the index first using "
            "download_existing_release_index or generate_release_index function."
        )
    theme_type_tuples = json.loads(release_index_path.read_text())
    return sorted(
        (theme_type_tuple["theme"], theme_type_tuple["type"])
        for theme_type_tuple in theme_type_tuples
    )


def download_existing_release_index(release: str) -> bool:
    """
    Download a pregenerated index for an Overture Maps dataset release.

    Args:
        release (str): Release version.

    Returns:
        bool: Information whether index have been downloaded or not.
    """
    return _download_existing_release_index(release=release)


def _download_existing_release_index(
    release: str, theme: Optional[str] = None, type: Optional[str] = None
) -> bool:
    """
    Download a pregenerated index for an Overture Maps dataset release.

    Args:
        release (str): Release version.
        theme (Optional[str], optional): Specify a theme to be downloaded. Defaults to None.
        type (Optional[str], optional): Specify a type to be downloaded. Defaults to None.

    Returns:
        bool: Information whether index have been downloaded or not.
    """
    _check_release_version(release)
    if (theme is None and type is not None) or (theme is not None and type is None):
        raise ValueError("Theme and type both have to be present or None.")

    cache_directory = _get_release_cache_directory(release)

    logger = get_pooch_logger()
    logger.setLevel("WARNING")

    try:
        index_content_file_name = "release_index_content.json"
        index_content_file_url = (
            LFS_DIRECTORY_URL + (cache_directory / index_content_file_name).as_posix()
        )
        retrieve(
            index_content_file_url,
            fname=index_content_file_name,
            path=cache_directory,
            progressbar=False,
            known_hash=None,
        )
        rprint("Downloaded index metadata file")

        theme_type_tuples = json.loads((cache_directory / index_content_file_name).read_text())

        with TrackProgressBar() as progress:
            for theme_type_tuple in progress.track(
                theme_type_tuples,
                description="Downloading release indexes",
            ):
                theme_value = theme_type_tuple["theme"]
                type_value = theme_type_tuple["type"]
                sha_value = theme_type_tuple["sha"]
                file_name = _get_index_file_name(theme_value, type_value)

                if theme_value == theme and type_value == type:
                    continue

                index_file_url = LFS_DIRECTORY_URL + (cache_directory / file_name).as_posix()
                retrieve(
                    index_file_url,
                    fname=file_name,
                    path=cache_directory,
                    progressbar=False,
                    known_hash=sha_value,
                )
                rprint(f"Downloaded index file {release}/{file_name}")

    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            return False

        raise

    return True


def generate_release_index(release: str) -> bool:
    """
    Generate an index for an Overture Maps dataset release.

    Args:
        release (str): Release version.

    Returns:
        bool: Information whether index have been generated or not.
    """
    return _generate_release_index(release=release)


def _generate_release_index(
    release: str, theme: Optional[str] = None, type: Optional[str] = None
) -> bool:
    """
    Generate an index for an Overture Maps dataset release.

    Args:
        release (str): Release version.
        theme (Optional[str], optional): Specify a theme to be generated. Defaults to None.
        type (Optional[str], optional): Specify a type to be generated. Defaults to None.

    Returns:
        bool: Information whether index have been generated or not.
    """
    _check_release_version(release)
    if (theme is None and type is not None) or (theme is not None and type is None):
        raise ValueError("Theme and type both have to be present or None.")

    cache_directory = _get_release_cache_directory(release)
    release_index_path = cache_directory / "release_index_content.json"
    if theme is not None and type is not None:
        release_index_path = cache_directory / f"release_index_content_{theme}_{type}.json"

    if release_index_path.exists():
        rprint("Cache exists. Skipping generation.")
        return False

    cache_directory.mkdir(exist_ok=True, parents=True)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir_name:
        tmp_dir_path = Path(tmp_dir_name)
        bounding_boxes_path = tmp_dir_path / "overture_bounding_boxes"

        dataset_path = f"overturemaps-us-west-2/release/{release}"
        if theme is not None and type is not None:
            dataset_path += f"/theme={theme}/type={type}"

        map_parquet_dataset(
            dataset_path=dataset_path,
            destination_path=bounding_boxes_path,
            function=calculate_row_group_bounding_box,
            progress_description="Generating Overture Maps release cache index",
            columns=["bbox"],
            filesystem=fs.S3FileSystem(
                anonymous=True, region="us-west-2", request_timeout=30, connect_timeout=10
            ),
        )

        df = pd.read_parquet(bounding_boxes_path)
        df["split_filename"] = df["filename"].str.split("/")
        df["theme"] = df["split_filename"].apply(lambda x: x[3].split("=")[1])
        df["type"] = df["split_filename"].apply(lambda x: x[4].split("=")[1])
        df = gpd.GeoDataFrame(
            df,
            geometry=np.apply_along_axis(
                lambda row: box(*row), 1, df[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
            ),
            crs=4326,
        )

        with TrackProgressBar() as progress:
            theme_type_tuples = df[["theme", "type"]].drop_duplicates().reset_index(drop=True)
            file_hashes = []
            for value_tuple in progress.track(
                theme_type_tuples.to_dict(orient="records"),
                description="Saving geoparquet indexes",
            ):
                theme_value = value_tuple["theme"]
                type_value = value_tuple["type"]
                file_name = _get_index_file_name(theme_value, type_value)

                df_subset = df[
                    (df["theme"] == theme_value) & (df["type"] == type_value)
                ].sort_values(["filename", "row_group"])
                cache_file_path = cache_directory / file_name

                df_subset[
                    [
                        "filename",
                        "row_group",
                        "row_indexes_ranges",
                        "geometry",
                    ]
                ].to_parquet(
                    cache_file_path,
                    geometry_encoding="geoarrow",
                    schema_version="1.1.0",
                    write_covering_bbox=True,
                )
                file_hashes.append(file_hash(str(cache_file_path)))
                rprint(f"Saved index file {cache_file_path}")

        theme_type_tuples["sha"] = pd.Series(file_hashes)
        theme_type_tuples.to_json(release_index_path, orient="records")

    return True


def _check_release_version(release: str) -> None:
    if release < MINIMAL_SUPPORTED_RELEASE_VERSION:
        raise ReleaseVersionNotSupportedError(
            f"Release version {release} is not supported."
            f" Minimal supported version is {MINIMAL_SUPPORTED_RELEASE_VERSION}."
        )


def _get_release_cache_directory(release: str) -> Path:
    return Path(f"release_indexes/{release}")


def _get_index_file_name(theme_value: str, type_value: str) -> str:
    return f"{theme_value}_{type_value}.parquet"


def _consolidate_release_index_files(release: str, remove_other_files: bool = False) -> bool:
    _check_release_version(release)

    cache_directory = _get_release_cache_directory(release)
    release_index_path = cache_directory / "release_index_content.json"
    if release_index_path.exists():
        rprint("Cache exists. Skipping generation.")
        return False

    index_content_json_files = list(cache_directory.glob("*.json"))

    all_indexes = [
        index_metadata
        for index_file in index_content_json_files
        for index_metadata in json.loads(index_file.read_text())
    ]

    release_index_path.write_text(json.dumps(all_indexes))

    if remove_other_files:
        for index_file in index_content_json_files:
            index_file.unlink()

    return True
