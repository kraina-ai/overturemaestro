"""Regression tests for overlapping progress displays."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pytest

from overturemaestro._rich_progress import VERBOSITY_MODE, progress_display


def _run_operation(
    spinner: bool, mode: VERBOSITY_MODE, inner: Optional[Callable[[], None]] = None
) -> None:
    with (
        progress_display(mode) as display,
        display.spinner("Working") if spinner else display.bar("Working", 1),
    ):
        if inner:
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                executor.submit(inner).result(timeout=5)
            finally:
                # waiting for a blocked operation would hang the test instead of failing it
                executor.shutdown(wait=False)


@pytest.mark.parametrize("outer_spinner", [False, True])  # type: ignore
@pytest.mark.parametrize("inner_spinner", [False, True])  # type: ignore
@pytest.mark.parametrize("mode", ["verbose", "transient", "silent"])  # type: ignore
def test_overlapping_operations(
    outer_spinner: bool, inner_spinner: bool, mode: VERBOSITY_MODE
) -> None:
    """Test if an operation can run while a display of another one is still active."""
    _run_operation(outer_spinner, mode, lambda: _run_operation(inner_spinner, mode))


def test_display_recovers_after_exception() -> None:
    """Test if a failed operation releases the display for the next one."""
    with pytest.raises(ValueError, match="operation failed"), progress_display("transient"):
        raise ValueError("operation failed")

    _run_operation(False, "transient")
