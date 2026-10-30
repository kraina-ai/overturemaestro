"""Offline regressions for overlapping progress displays."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from overturemaestro import _rich_progress as progress_module


@pytest.mark.parametrize("first_spinner", [False, True])
@pytest.mark.parametrize("second_spinner", [False, True])
@pytest.mark.parametrize("mode", ["verbose", "transient", "silent"])
@pytest.mark.parametrize("force_terminal", [False, True])
def test_overlapping_progress(monkeypatch, first_spinner, second_spinner, mode, force_terminal):
    """Another worker can finish while the first display remains active."""
    monkeypatch.setattr(progress_module, "FORCE_TERMINAL", force_terminal)

    def make_progress(spinner):
        if spinner:
            return progress_module.TrackProgressSpinner("Working", verbosity_mode=mode)
        return progress_module.TrackProgressBar(verbosity_mode=mode)

    first = make_progress(first_spinner)
    second = make_progress(second_spinner)

    def run_second():
        with second:
            task = second.add_task("Concurrent work", total=1)
            second.update(task, advance=1, refresh=True)
        return second.tasks[-1].completed

    with ThreadPoolExecutor(max_workers=1) as executor:
        with first:
            assert executor.submit(run_second).result(timeout=5) == 1
            if mode != "silent":
                assert first.live.is_started
    # Reusing a display after contention must not leave it permanently silent.
    with second:
        if mode != "silent":
            assert second.live.is_started


def test_progress_recovers_after_exception(monkeypatch):
    """A failed operation releases its display for the next operation."""
    monkeypatch.setattr(progress_module, "FORCE_TERMINAL", False)
    with pytest.raises(ValueError, match="operation failed"):
        with progress_module.TrackProgressBar():
            raise ValueError("operation failed")
    with progress_module.TrackProgressBar() as progress:
        assert progress.live.is_started


def test_owner_finishes_before_overlapping_operation():
    """Stopping a suppressed operation must not stop a newer display."""
    first = progress_module.TrackProgressBar()
    second = progress_module.TrackProgressBar()
    first.start()
    second.start()
    first.stop()
    try:
        with progress_module.TrackProgressBar() as third:
            second.stop()
            assert third.live.is_started
    finally:
        first.stop()
        second.stop()


def test_start_failure_releases_display(monkeypatch):
    """A rendering failure is propagated without blocking future displays."""

    def fail_start(self):
        raise RuntimeError("render failed")

    with monkeypatch.context() as patch:
        patch.setattr(progress_module.Progress, "start", fail_start)
        with pytest.raises(RuntimeError, match="render failed"):
            progress_module.TrackProgressBar().start()
    with progress_module.TrackProgressBar() as progress:
        assert progress.live.is_started


def test_silent_operation_does_not_claim_display():
    """Explicitly silent work leaves the display available."""
    with progress_module.TrackProgressBar(verbosity_mode="silent"):
        with progress_module.TrackProgressBar() as progress:
            progress.start()
            assert progress.live.is_started
