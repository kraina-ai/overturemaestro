"""
Release index cache utilities.

Functions used to load cached release index, download it from a dedicated repository or to generate
it on demand.
"""

import json
import tempfile
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Literal, Optional, Union, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import platformdirs
import pyarrow.fs as fs
from fsspec.implementations.github import GithubFileSystem
from fsspec.implementations.http import HTTPFileSystem
from pooch import file_hash, retrieve
from pooch import get_logger as get_pooch_logger
from rich import print as rprint
from shapely import box
from shapely.geometry.base import BaseGeometry

from overturemaestro._geometry_clustering import calculate_row_group_bounding_box
from overturemaestro._parquet_multiprocessing import map_parquet_dataset
from overturemaestro._rich_progress import VERBOSITY_MODE, TrackProgressBar

__all__ = [
    "download_existing_release_index",
    "generate_release_index",
    "get_available_theme_type_pairs",
    "get_global_release_cache_directory",
    "get_newest_release_version",
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


def get_newest_release_version() -> str:
    """
    Get newest available OvertureMaps release version.

    Checks available precalculated release indexes in the GitHub repository
    and returns the newest available version.

    Returns:
        str: Release version.
    """
    release_versions = _load_all_available_release_versions_from_github()
    return sorted(release_versions)[-1]


def get_available_release_versions() -> list[str]:
    """
    Get available OvertureMaps release versions.

    Checks available precalculated release indexes in the GitHub repository
    and returns them.

    Returns:
        list[str]: Release versions.
    """
    return sorted(_load_all_available_release_versions_from_github())[::-1]


@overload
def load_release_indexes(
    theme_type_pairs: list[tuple[str, str]],
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame: ...


@overload
def load_release_indexes(
    theme_type_pairs: list[tuple[str, str]],
    release: str,
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame: ...


@overload
def load_release_indexes(
    theme_type_pairs: list[tuple[str, str]],
    release: Optional[str] = None,
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame: ...


def load_release_indexes(
    theme_type_pairs: list[tuple[str, str]],
    release: Optional[str] = None,
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame:
    """
    Load multiple release indexes as a GeoDataFrame.

    Args:
        theme_type_pairs (list[tuple[str, str]]): Pairs of themes and types of the dataset.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        geometry_filter (Optional[BaseGeometry], optional): Geometry to pre-filter resulting rows.
            Defaults to None.
        remote_index (bool, optional): Avoid downloading the index and stream it from remote source.
            Defaults to False.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        gpd.GeoDataFrame: Index with bounding boxes for each row group for each parquet file.
    """
    return gpd.pd.concat(
        load_release_index(
            theme=theme_value,
            type=type_value,
            release=release,
            geometry_filter=geometry_filter,
            remote_index=remote_index,
            verbosity_mode=verbosity_mode,
        )
        for theme_value, type_value in theme_type_pairs
    )


@overload
def load_release_index(
    theme: str,
    type: str,
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame: ...


@overload
def load_release_index(
    theme: str,
    type: str,
    release: str,
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame: ...


@overload
def load_release_index(
    theme: str,
    type: str,
    release: Optional[str] = None,
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame: ...


def load_release_index(
    theme: str,
    type: str,
    release: Optional[str] = None,
    *,
    geometry_filter: Optional[BaseGeometry] = None,
    remote_index: bool = False,
    skip_index_download: bool = False,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> gpd.GeoDataFrame:
    """
    Load release index as a GeoDataFrame.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        geometry_filter (Optional[BaseGeometry], optional): Geometry to pre-filter resulting rows.
            Defaults to None.
        remote_index (bool, optional): Avoid downloading the index and stream it from remote source.
            Defaults to False.
        skip_index_download (bool, optional): Avoid downloading the index if doesn't exist locally
            and generate it instead. Defaults to False.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        gpd.GeoDataFrame: Index with bounding boxes for each row group for each parquet file.
    """
    if not release:
        release = get_newest_release_version()

    _check_release_version(release)
    cache_directory = _get_global_release_cache_directory(release)
    index_file_path: Union[Path, str] = cache_directory / _get_index_file_name(theme, type)

    file_exists = Path(index_file_path).exists()
    filesystem = None

    if not file_exists:
        if remote_index:
            filesystem = HTTPFileSystem()
            local_cache_path = _get_local_release_cache_directory(release)
            index_file_path = LFS_DIRECTORY_URL + str(
                local_cache_path / _get_index_file_name(theme, type)
            )
        elif skip_index_download:
            # Generate the index and skip download
            generate_release_index(
                release,
                verbosity_mode=verbosity_mode,
            )
        else:
            # Try to download the index or generate it if cannot be downloaded
            download_existing_release_index(
                release,
                verbosity_mode=verbosity_mode,
            ) or generate_release_index(
                release,
                verbosity_mode=verbosity_mode,
            )

    if geometry_filter is None:
        return gpd.read_parquet(path=index_file_path, filesystem=filesystem)

    xmin, ymin, xmax, ymax = geometry_filter.bounds

    df = gpd.read_parquet(
        path=index_file_path, bbox=(xmin, ymin, xmax, ymax), filesystem=filesystem
    )
    df = df[df.intersects(geometry_filter)]
    return df


@overload
def get_available_theme_type_pairs() -> list[tuple[str, str]]: ...


@overload
def get_available_theme_type_pairs(release: str) -> list[tuple[str, str]]: ...


def get_available_theme_type_pairs(release: Optional[str] = None) -> list[tuple[str, str]]:
    """
    Get a list of available theme and type objects for a given release.

    Args:
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.

    Returns:
        list[tuple[str, str]]: List of theme and type pairs.
    """
    if not release:
        release = get_newest_release_version()

    _check_release_version(release)

    cache_directory = _get_global_release_cache_directory(release)
    release_index_path = cache_directory / "release_index_content.json"

    if release_index_path.exists():
        index_content = pd.read_json(release_index_path)
    else:
        local_cache_directory = _get_local_release_cache_directory(release)
        index_content_file_name = "release_index_content.json"
        index_content_file_url = (
            LFS_DIRECTORY_URL + (local_cache_directory / index_content_file_name).as_posix()
        )
        index_content = pd.read_json(index_content_file_url)

    return sorted(index_content[["theme", "type"]].itertuples(index=False, name=None))


@overload
def download_existing_release_index(
    *,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool: ...


@overload
def download_existing_release_index(
    release: str, *, verbosity_mode: VERBOSITY_MODE = "transient"
) -> bool: ...


def download_existing_release_index(
    release: Optional[str] = None,
    *,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Download a pregenerated index for an Overture Maps dataset release.

    Args:
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been downloaded or not.
    """
    return _download_existing_release_index(release=release, verbosity_mode=verbosity_mode)


def _download_existing_release_index(
    release: Optional[str] = None,
    theme: Optional[str] = None,
    type: Optional[str] = None,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Download a pregenerated index for an Overture Maps dataset release.

    Args:
        theme (Optional[str], optional): Specify a theme to be downloaded. Defaults to None.
        type (Optional[str], optional): Specify a type to be downloaded. Defaults to None.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been downloaded or not.
    """
    if not release:
        release = get_newest_release_version()

    _check_release_version(release)
    if (theme is None and type is not None) or (theme is not None and type is None):
        raise ValueError("Theme and type both have to be present or None.")

    global_cache_directory = _get_global_release_cache_directory(release)
    local_cache_directory = _get_local_release_cache_directory(release)

    logger = get_pooch_logger()
    logger.setLevel("WARNING")

    try:
        index_content_file_name = "release_index_content.json"
        index_content_file_url = (
            LFS_DIRECTORY_URL + (local_cache_directory / index_content_file_name).as_posix()
        )
        retrieve(
            index_content_file_url,
            fname=index_content_file_name,
            path=global_cache_directory,
            progressbar=False,
            known_hash=None,
        )

        theme_type_tuples = json.loads(
            (global_cache_directory / index_content_file_name).read_text()
        )

        with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
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

                index_file_url = LFS_DIRECTORY_URL + (local_cache_directory / file_name).as_posix()
                retrieve(
                    index_file_url,
                    fname=file_name,
                    path=global_cache_directory,
                    progressbar=False,
                    known_hash=sha_value,
                )

    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            return False

        raise

    return True


def generate_release_index(
    release: str,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Generate an index for an Overture Maps dataset release.

    Args:
        release (str): Release version.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been generated or not.
    """
    return _generate_release_index(release=release, verbosity_mode=verbosity_mode)


def _generate_release_index(
    release: str,
    theme: Optional[str] = None,
    type: Optional[str] = None,
    dataset_path: str = "overturemaps-us-west-2/release",
    dataset_fs: Literal["local", "s3"] = "s3",
    index_location_path: Optional[Path] = None,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> bool:
    """
    Generate an index for an Overture Maps dataset release.

    Args:
        release (str): Release version.
        theme (Optional[str], optional): Specify a theme to be generated. Defaults to None.
        type (Optional[str], optional): Specify a type to be generated. Defaults to None.
        dataset_path (str, optional): Specify dataset path.
            Defaults to "overturemaps-us-west-2/release/".
        dataset_fs (Literal["local", "s3"], optional): Which filesystem use in PyArrow operations.
            Defaults to "s3".
        index_location_path (Path, optional): Specify index location path. If not, will generate to
            the global cache location. Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        bool: Information whether index have been generated or not.
    """
    _check_release_version(release)
    if (theme is None and type is not None) or (theme is not None and type is None):
        raise ValueError("Theme and type both have to be present or None.")

    cache_directory = Path(index_location_path or _get_global_release_cache_directory(release))
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

        dataset_path = f"{dataset_path}/{release}"
        if theme is not None and type is not None:
            dataset_path += f"/theme={theme}/type={type}"

        map_parquet_dataset(
            dataset_path=dataset_path,
            destination_path=bounding_boxes_path,
            function=calculate_row_group_bounding_box,
            progress_description="Generating Overture Maps release cache index",
            columns=["bbox"],
            filesystem=(
                fs.S3FileSystem(
                    anonymous=True, region="us-west-2", request_timeout=30, connect_timeout=10
                )
                if dataset_fs == "s3"
                else None
            ),
            verbosity_mode=verbosity_mode,
        )

        df = pd.read_parquet(bounding_boxes_path)
        df["split_filename"] = df["filename"].str.split("/")
        df["theme"] = df["split_filename"].apply(
            lambda path_parts: next(filter(lambda x: "theme=" in x, path_parts)).split("=")[1]
        )
        df["type"] = df["split_filename"].apply(
            lambda path_parts: next(filter(lambda x: "type=" in x, path_parts)).split("=")[1]
        )
        df = gpd.GeoDataFrame(
            df,
            geometry=np.apply_along_axis(
                lambda row: box(*row), 1, df[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
            ),
            crs=4326,
        )

        with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
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


def get_global_release_cache_directory() -> Path:
    """Get global index cache location path."""
    return Path(platformdirs.user_cache_dir("OvertureMaestro")) / "release_indexes"


def _get_global_release_cache_directory(release: str) -> Path:
    return get_global_release_cache_directory() / release


def _get_local_release_cache_directory(release: str) -> Path:
    return Path("release_indexes") / release


def _get_index_file_name(theme_value: str, type_value: str) -> str:
    return f"{theme_value}_{type_value}.parquet"


def _load_all_available_release_versions_from_github() -> list[str]:  # pragma: no cover
    release_versions_cache_file = (
        Path(platformdirs.user_cache_dir("OvertureMaestro"))
        / "release_indexes"
        / "_available_release_versions.json"
    )

    current_date = date.today()
    if release_versions_cache_file.exists():
        cache_value = json.loads(release_versions_cache_file.read_text())
        if date.fromisoformat(cache_value["date"]) >= current_date:
            return cast(list[str], cache_value["release_versions"])

    release_versions_cache_file.parent.mkdir(parents=True, exist_ok=True)
    gh_fs = GithubFileSystem(org="kraina-ai", repo="overturemaps-releases-indexes", sha="main")
    release_versions = [file_path.split("/")[1] for file_path in gh_fs.ls("release_indexes")]
    release_versions_cache_file.write_text(
        json.dumps(dict(date=current_date.isoformat(), release_versions=release_versions))
    )
    return release_versions


def _consolidate_release_index_files(
    release: str,
    remove_other_files: bool = False,
    index_location_path: Optional[Path] = None,
) -> bool:  # pragma: no cover
    _check_release_version(release)

    cache_directory = Path(index_location_path or _get_global_release_cache_directory(release))
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
