"""Common components for tests."""

import pytest

from overturemaestro.release_index import download_existing_release_index

TEST_RELEASE_VERSION = "2024-08-20.0"


@pytest.fixture(scope="session")  # type: ignore
def test_release_version() -> str:
    """Release version for tests."""
    return TEST_RELEASE_VERSION


@pytest.fixture(autouse=True, scope="session")  # type: ignore
def download_release_index(test_release_version: str) -> None:
    """Download release index for testing purposes."""
    download_existing_release_index(release=test_release_version)
