import multiprocessing
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty, Queue
from time import sleep, time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from overturemaestro._rich_progress import VERBOSITY_MODE, TrackProgressSpinner

if TYPE_CHECKING:  # pragma: no cover
    from multiprocessing.managers import ValueProxy
    from threading import Lock

    import pyarrow as pa
    import pyarrow.fs as fs
    from rich.progress import Progress

# Using `spawn` method to enable integration with Polars and probably other Rust-based libraries
# https://docs.pola.rs/user-guide/misc/multiprocessing/
ctx: multiprocessing.context.SpawnContext = multiprocessing.get_context("spawn")


class MultiprocessingRuntimeError(RuntimeError): ...


def _job(
    queue: Queue[tuple[str, int]],
    tracker: "ValueProxy[int]",
    tracker_lock: "Lock",
    save_path: Path,
    function: Callable[[str, int, "pa.Table"], "pa.Table"],
    columns: Optional[list[str]],
    filesystem: "fs.FileSystem",
) -> None:  # pragma: no cover
    import hashlib

    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    current_pid = multiprocessing.current_process().pid

    writers = {}
    while not queue.empty():
        try:
            file_name, row_group_index = None, None
            file_name, row_group_index = queue.get_nowait()

            fragment_manual = ds.ParquetFileFormat().make_fragment(
                file=file_name,
                filesystem=filesystem,
                row_groups=[row_group_index],
            )
            row_group_table = fragment_manual.to_table(columns=columns)
            if row_group_table.num_rows == 0:
                with tracker_lock:
                    tracker.value += 1
                continue

            result_table = function(file_name, row_group_index, row_group_table)

            if result_table.num_rows == 0:
                with tracker_lock:
                    tracker.value += 1
                continue

            h = hashlib.new("sha256")
            h.update(result_table.schema.to_string().encode())
            schema_hash = h.hexdigest()

            if schema_hash not in writers:
                filepath = save_path / str(current_pid) / f"{schema_hash}.parquet"
                filepath.parent.mkdir(exist_ok=True, parents=True)
                writers[schema_hash] = pq.ParquetWriter(filepath, result_table.schema)

            writers[schema_hash].write_table(result_table)

            with tracker_lock:
                tracker.value += 1
        except Empty:
            pass
        except Exception as ex:
            if file_name is not None and row_group_index is not None:
                queue.put((file_name, row_group_index))

            msg = (
                f"Error in worker (PID: {current_pid},"
                f" Parquet: {file_name}, Row group: {row_group_index})"
            )
            raise MultiprocessingRuntimeError(msg) from ex

    for writer in writers.values():
        writer.close()


class WorkerProcess(ctx.Process):  # type: ignore[name-defined,misc]
    def __init__(self, *args: Any, **kwargs: Any):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception: Optional[tuple[Exception, str]] = None

    def run(self) -> None:  # pragma: no cover
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            import traceback

            tb: str = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self) -> Optional[tuple[Exception, str]]:
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class SingletonContextManager(SyncManager):
    def __new__(cls, ctx: multiprocessing.context.SpawnContext) -> "SingletonContextManager":
        if not hasattr(cls, "instance"):
            cls.instance = ctx.Manager()
        return cast(SingletonContextManager, cls.instance)


def _read_row_group_number(path: str, filesystem: "fs.FileSystem") -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path, filesystem=filesystem).num_row_groups)


def map_parquet_dataset(
    dataset_path: Union[str, Path, list[str], list[Path]],
    destination_path: Path,
    function: Callable[[str, int, "pa.Table"], "pa.Table"],
    progress_description: str,
    columns: Optional[list[str]] = None,
    filesystem: Optional["fs.FileSystem"] = None,
    report_progress_as_text: bool = True,
    verbosity_mode: VERBOSITY_MODE = "transient",
    max_workers: Optional[int] = None,
) -> None:
    """
    Apply a function over parquet dataset in a multiprocessing environment.

    Will save results in multiple files in a destination path.

    Args:
        dataset_path (Union[str, list[str]]): Path(s) of the parquet dataset.
        destination_path (Path): Path of the destination.
        function (Callable[[str, pa.Table], pa.Table]): Function to apply over a parquet file name
            and a row group table. Will save resulting table in a new parquet file.
        progress_description (str): Progress bar description.
        columns (Optional[list[str]], optional): List of columns to read. Defaults to `None`.
        filesystem (Optional[fs.FileSystem], optional): Filesystem for the dataset.
            Defaults to `None`.
        report_progress_as_text (bool, optional): Whether to report task progress every minute.
            Defaults to `True`.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.
    """
    with TrackProgressSpinner(
        "Preparing multiprocessing environment", verbosity_mode=verbosity_mode
    ):
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        import pyarrow.parquet as pq

        from overturemaestro._rich_progress import TrackProgressBar

        manager = SingletonContextManager(ctx=ctx)

        queue: Queue[tuple[str, int]] = manager.Queue()
        tracker: ValueProxy[int] = manager.Value("i", 0)
        tracker_lock: Lock = manager.Lock()

        dataset = pq.ParquetDataset(dataset_path, filesystem=filesystem)

        no_cpus = multiprocessing.cpu_count()

        min_no_workers = 32 if no_cpus >= 8 else 16
        no_scan_workers = min(
            max(min_no_workers, no_cpus + 4), 64
        )  # minimum 16 / 32 workers, but not more than 64

        no_processing_workers = no_cpus

        if max_workers:
            no_scan_workers = min(max_workers, no_scan_workers)
            no_processing_workers = min(max_workers, no_processing_workers)

    total_files = len(dataset.files)

    with (
        TrackProgressBar(verbosity_mode=verbosity_mode) as progress,
        ProcessPoolExecutor(max_workers=min(no_scan_workers, total_files)) as ex,
    ):
        fn = partial(_read_row_group_number, filesystem=dataset.filesystem)
        row_group_numbers = list(
            progress.track(
                ex.map(fn, dataset.files, chunksize=1),
                description="Reading all parquet files row groups",
                total=total_files,
            )
        )

        for pq_file, row_group_number in zip(dataset.files, row_group_numbers):
            for row_group in range(row_group_number):
                queue.put((pq_file, row_group))

    total = queue.qsize()

    destination_path.mkdir(parents=True, exist_ok=True)

    try:
        processes = [
            WorkerProcess(
                target=_job,
                args=(
                    queue,
                    tracker,
                    tracker_lock,
                    destination_path,
                    function,
                    columns,
                    dataset.filesystem,
                ),
            )
            for _ in range(min(no_processing_workers, total))
        ]
        with TrackProgressBar(verbosity_mode=verbosity_mode) as progress_bar:
            progress_bar.add_task(description=progress_description, total=total)
            _run_processes(
                processes=processes,
                queue=queue,
                tracker=tracker,
                tracker_lock=tracker_lock,
                total=total,
                progress_bar=progress_bar,
                report_progress_as_text=report_progress_as_text,
            )
    finally:  # pragma: no cover
        _report_exceptions(processes=processes)


def _run_processes(
    processes: list[WorkerProcess],
    queue: Queue[tuple[str, int]],
    tracker: "ValueProxy[int]",
    tracker_lock: "Lock",
    total: int,
    progress_bar: "Progress",
    report_progress_as_text: bool,
) -> None:
    # Run processes
    for p in processes:
        if queue.empty():
            break
        p.start()

    sleep_time = 0.1
    next_update_time = time()
    while any(process.is_alive() for process in processes):
        if any(p.exception for p in processes):  # pragma: no cover
            break

        with tracker_lock:
            completed = tracker.value

        progress_bar.update(task_id=progress_bar.task_ids[0], completed=completed, refresh=True)

        if report_progress_as_text:
            current_time = time()
            if time() >= next_update_time:
                progress_bar.print_progress()
                next_update_time = current_time + 60

        sleep(sleep_time)
        sleep_time = min(1.0, sleep_time + 0.1)

    progress_bar.update(task_id=progress_bar.task_ids[0], completed=total, refresh=True)


def _report_exceptions(processes: list[WorkerProcess]) -> None:
    # In case of exception
    exceptions = []
    for p in processes:
        if p.is_alive():
            p.terminate()

        if p.exception:
            exceptions.append(p.exception)

    if exceptions:
        # use ExceptionGroup in Python3.11
        _raise_multiple(exceptions)


def _raise_multiple(exceptions: list[tuple[Exception, str]]) -> None:
    if not exceptions:
        return
    try:
        error, traceback = exceptions.pop()
        msg = f"{error}\n\nOriginal {traceback}"
        raise type(error)(msg)
    finally:
        _raise_multiple(exceptions)
