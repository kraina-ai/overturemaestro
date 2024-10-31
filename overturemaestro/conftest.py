"""Fixtures for doctests."""

import doctest
from doctest import OutputChecker

import pandas
import pytest
from pytest_mock import MockerFixture

IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")


class CustomOutputChecker(OutputChecker):
    """Custom doctest OutputChecker for ignoring logs from functions."""

    def check_output(self: doctest.OutputChecker, want: str, got: str, optionflags: int) -> bool:
        """Skips output checking if IGNORE_RESULT flag is present."""
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


doctest.OutputChecker = CustomOutputChecker  # type: ignore


@pytest.fixture(autouse=True, scope="session")  # type: ignore
def pandas_terminal_width() -> None:
    """Change pandas dataframe display options."""
    pandas.set_option("display.width", 90)
    pandas.set_option("display.max_colwidth", 35)


@pytest.fixture(autouse=True, scope="function")  # type: ignore
def patch_get_available_versions(mocker: MockerFixture) -> None:
    """Mock getting available release versions without GitHub."""
    mocker.patch(
        "overturemaestro.release_index._load_all_available_release_versions_from_github",
        return_value=[
            "2024-01-01-test",
            "2024-04-16-beta.0",
            "2024-05-16-beta.0",
            "2024-06-13-beta.0",
            "2024-06-13-beta.1",
            "2024-07-22.0",
            "2024-08-20.0",
            "2024-09-18.0",
            "2024-10-23.0",
        ],
    )
