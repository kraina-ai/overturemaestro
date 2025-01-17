"""Functions for global cache."""

import shutil
from pathlib import Path

import platformdirs


def clear_release_indexes_cache() -> None:
    """Clear library release indexes cache."""
    for directory in (
        get_global_release_cache_directory(),
        get_global_wide_form_release_cache_directory(),
    ):
        shutil.rmtree(directory)


def get_global_release_cache_directory() -> Path:
    """Get global index cache location path."""
    return Path(platformdirs.user_cache_dir("OvertureMaestro")) / "release_indexes"


def _get_global_release_cache_directory(release: str) -> Path:
    return get_global_release_cache_directory() / release


def _get_local_release_cache_directory(release: str) -> Path:
    return Path("release_indexes") / release


def get_global_wide_form_release_cache_directory() -> Path:
    """Get global index cache location path."""
    return Path(platformdirs.user_cache_dir("OvertureMaestro")) / "wide_form_release_indexes"


def _get_global_wide_form_release_cache_directory(release: str) -> Path:
    return get_global_wide_form_release_cache_directory() / release


def _get_local_wide_form_release_cache_directory(release: str) -> Path:
    return Path("wide_form_release_indexes") / release
