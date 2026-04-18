"""Fixtures for doctests."""

import doctest
from doctest import OutputChecker

import pandas
import pytest

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
