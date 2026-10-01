"""Functions for retrieving Overture Maps places data."""

import warnings
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.error import HTTPError as urllib_HTTPError

import pandas as pd
from fsspec.implementations.github import GithubFileSystem

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame

# Oldest release with the schema v2.0 taxonomy published in the docs repository
TAXONOMY_MIN_RELEASE_VERSION = "2026-09-23.0"
TAXONOMY_GITHUB_ORG = "OvertureMaps"
TAXONOMY_GITHUB_REPO = "docs"
TAXONOMY_GITHUB_REF = "main"
TAXONOMY_GITHUB_PATH = "static/taxonomy"
TAXONOMY_FILE_NAME = "taxonomy.csv"
TAXONOMY_GITHUB_RAW_URL = (
    f"https://raw.githubusercontent.com/{TAXONOMY_GITHUB_ORG}/{TAXONOMY_GITHUB_REPO}"
    f"/refs/heads/{TAXONOMY_GITHUB_REF}/{TAXONOMY_GITHUB_PATH}"
)

def _get_taxonomy_github_filesystem() -> GithubFileSystem:
    return GithubFileSystem(
        org=TAXONOMY_GITHUB_ORG, repo=TAXONOMY_GITHUB_REPO, sha=TAXONOMY_GITHUB_REF
    )


def _get_taxonomy_release_versions_from_github() -> list[str]:
    available_paths = _get_taxonomy_github_filesystem().ls(TAXONOMY_GITHUB_PATH)

    return sorted(
        PurePosixPath(available_path).name
        for available_path in available_paths
        if PurePosixPath(available_path).name >= TAXONOMY_MIN_RELEASE_VERSION
    )


def _get_closest_taxonomy_release_version(release_version: str) -> str:
    taxonomy_release_versions = _get_taxonomy_release_versions_from_github()

    if not taxonomy_release_versions:
        return TAXONOMY_MIN_RELEASE_VERSION

    return max(
        (version for version in taxonomy_release_versions if version <= release_version),
        default=TAXONOMY_MIN_RELEASE_VERSION,
    )


def _download_taxonomy_for_release(release_version: str) -> "DataFrame":
    try:
        return pd.read_csv(
            f"https://docs.overturemaps.org/taxonomy/{release_version}/taxonomy.csv",
        )
    except urllib_HTTPError:
        taxonomy_release_version = _get_closest_taxonomy_release_version(release_version)

        warnings.warn(
            (
                f"Couldn't download taxonomy for release {release_version} from the docs website."
                f" Downloading taxonomy for the closest available release"
                f" ({taxonomy_release_version}) from GitHub."
            ),
            stacklevel=0,
        )

        taxonomy_url = (
            f"{TAXONOMY_GITHUB_RAW_URL}/{taxonomy_release_version}/{TAXONOMY_FILE_NAME}"
        )

        return pd.read_csv(taxonomy_url)
