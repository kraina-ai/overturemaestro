"""
Data downloader utilities.

Functions used to download Overture Maps data before local filtering.
"""

import multiprocessing
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast, overload

from overturemaestro._exceptions import MissingColumnError
from overturemaestro.elapsed_time_decorator import show_total_elapsed_time_decorator

if TYPE_CHECKING:  # pragma: no cover
    from pyarrow import Schema
    from pyarrow.compute import Expression
    from shapely.geometry.base import BaseGeometry

    from overturemaestro._rich_progress import VERBOSITY_MODE

PYARROW_EXPRESSION = tuple[Any, Any, Any]
PYARROW_FILTER = Union["Expression", list[PYARROW_EXPRESSION], list[list[PYARROW_EXPRESSION]]]

__all__ = [
    "download_data",
    "download_data_for_multiple_types",
]


@overload
def download_data_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    *,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]],
    columns_to_download: Optional[list[Optional[list[str]]]],
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> list[Path]: ...


@overload
def download_data_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    release: str,
    *,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]],
    columns_to_download: Optional[list[Optional[list[str]]]],
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> list[Path]: ...


@overload
def download_data_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    release: Optional[str] = None,
    *,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]],
    columns_to_download: Optional[list[Optional[list[str]]]],
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> list[Path]: ...


@show_total_elapsed_time_decorator
def download_data_for_multiple_types(
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    release: Optional[str] = None,
    *,
    pyarrow_filters: Optional[list[Optional[PYARROW_FILTER]]] = None,
    columns_to_download: Optional[list[Optional[list[str]]]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> list[Path]:
    """
    Downloads the data for the given release for multiple types.

    Args:
        theme_type_pairs (list[tuple[str, str]]): Pairs of themes and types of the dataset.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        pyarrow_filters (Optional[list[Optional[PYARROW_FILTER]]], optional): A list of pyarrow
            expressions used to filter specific theme type pair. Must be the same length as the list
            of theme type pairs. Defaults to None.
        columns_to_download (Optional[list[Optional[list[str]]]], optional): A list of pyarrow
            expressions used to filter specific theme type pair. Must be the same length as the list
            of theme type pairs. Defaults to None.
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.

    Returns:
        list[Path]: List of saved Geoparquet files paths.
    """
    import tempfile

    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from overturemaestro._rich_progress import TrackProgressBar
    from overturemaestro.release_index import get_newest_release_version

    if pyarrow_filters is not None and len(theme_type_pairs) != len(pyarrow_filters):
        raise ValueError("Pyarrow filters length doesn't match length of theme type pairs.")

    if columns_to_download is not None and len(theme_type_pairs) != len(columns_to_download):
        raise ValueError("Columns to download length doesn't match length of theme type pairs.")

    if not release:
        release = get_newest_release_version()

    working_directory = Path(working_directory)
    working_directory.mkdir(parents=True, exist_ok=True)

    all_result_file_paths = []
    theme_type_pairs_to_download = []
    pyarrow_filters_list = []
    columns_to_download_list = []
    result_file_paths_to_download = []
    for idx, (theme_value, type_value) in enumerate(theme_type_pairs):
        _pyarrow_filter = pyarrow_filters[idx] if pyarrow_filters else None
        _columns_to_download = columns_to_download[idx] if columns_to_download else None

        if _pyarrow_filter is not None:
            from pyarrow.parquet import filters_to_expression

            _pyarrow_filter = filters_to_expression(_pyarrow_filter)

        result_file_path = working_directory / _generate_result_file_path(
            release=release,
            theme=theme_value,
            type=type_value,
            geometry_filter=geometry_filter,
            pyarrow_filter=_pyarrow_filter,
            columns_to_download=_columns_to_download,
        )
        all_result_file_paths.append(result_file_path)

        if not result_file_path.exists() or ignore_cache:
            theme_type_pairs_to_download.append((theme_value, type_value))
            pyarrow_filters_list.append(_pyarrow_filter)
            columns_to_download_list.append(_columns_to_download)
            result_file_paths_to_download.append(result_file_path)

    if theme_type_pairs_to_download:
        with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)
            raw_parquet_files_per_pair = _download_data(
                release=release,
                theme_type_pairs=theme_type_pairs_to_download,
                geometry_filter=geometry_filter,
                pyarrow_filters=pyarrow_filters_list,
                columns_to_download=columns_to_download_list,
                working_directory=tmp_dir_path,
                verbosity_mode=verbosity_mode,
                max_workers=max_workers,
            )

            with TrackProgressBar(verbosity_mode=verbosity_mode) as progress:
                for result_file_path, raw_parquet_files in progress.track(
                    zip(result_file_paths_to_download, raw_parquet_files_per_pair),
                    total=len(result_file_paths_to_download),
                    description="Saving final geoparquet files",
                ):
                    final_dataset = ds.dataset(raw_parquet_files)
                    result_file_path.parent.mkdir(exist_ok=True, parents=True)
                    with pq.ParquetWriter(result_file_path, final_dataset.schema) as writer:
                        for batch in final_dataset.to_batches():
                            writer.write_batch(batch)

    return all_result_file_paths


@overload
def download_data(
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    *,
    pyarrow_filter: Optional[PYARROW_FILTER] = None,
    columns_to_download: Optional[list[str]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def download_data(
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    release: str,
    *,
    pyarrow_filter: Optional[PYARROW_FILTER] = None,
    columns_to_download: Optional[list[str]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@overload
def download_data(
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    release: Optional[str] = None,
    *,
    pyarrow_filter: Optional[PYARROW_FILTER] = None,
    columns_to_download: Optional[list[str]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> Path: ...


@show_total_elapsed_time_decorator
def download_data(
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    release: Optional[str] = None,
    *,
    pyarrow_filter: Optional[PYARROW_FILTER] = None,
    columns_to_download: Optional[list[str]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
    verbosity_mode: "VERBOSITY_MODE" = "transient",
    max_workers: Optional[int] = None,
) -> Path:
    """
    Downloads the data for the given release.

    Args:
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        release (Optional[str], optional): Release version. If not provided, will automatically load
            newest available release version. Defaults to None.
        pyarrow_filter (Optional[PYARROW_FILTER], optional): Filters to apply on a pyarrow dataset.
            Can be pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]]. Defaults to None.
        columns_to_download (Optional[list[str]], optional): List of columns to download.
            Automatically adds geometry column to the list. If None, will download all columns.
            Defaults to None.
        result_file_path (Union[str, Path], optional): Where to save
            the geoparquet file. If not provided, will be generated based on hashes
            from filters. Defaults to `None`.
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".
        max_workers (Optional[int], optional): Max number of multiprocessing workers used to
            process the dataset. Defaults to None.

    Returns:
        Path: Saved Geoparquet file path.
    """
    import tempfile

    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from overturemaestro._rich_progress import TrackProgressSpinner
    from overturemaestro.release_index import get_newest_release_version

    if not release:
        release = get_newest_release_version()

    working_directory = Path(working_directory)
    working_directory.mkdir(parents=True, exist_ok=True)

    if pyarrow_filter is not None:
        from pyarrow.parquet import filters_to_expression

        pyarrow_filter = filters_to_expression(pyarrow_filter)

    if result_file_path is None:
        result_file_path = working_directory / _generate_result_file_path(
            release=release,
            theme=theme,
            type=type,
            geometry_filter=geometry_filter,
            pyarrow_filter=pyarrow_filter,
            columns_to_download=columns_to_download,
        )

    result_file_path = Path(result_file_path)

    if not result_file_path.exists() or ignore_cache:
        with tempfile.TemporaryDirectory(dir=Path(working_directory).resolve()) as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)
            raw_parquet_files = _download_data(
                release=release,
                theme_type_pairs=[(theme, type)],
                geometry_filter=geometry_filter,
                pyarrow_filters=[pyarrow_filter],
                columns_to_download=[columns_to_download],
                working_directory=tmp_dir_path,
                verbosity_mode=verbosity_mode,
                max_workers=max_workers,
            )[0]

            final_dataset = ds.dataset(raw_parquet_files)

            result_file_path.parent.mkdir(exist_ok=True, parents=True)

            with (
                TrackProgressSpinner("Saving final geoparquet file", verbosity_mode=verbosity_mode),
                pq.ParquetWriter(result_file_path, final_dataset.schema) as writer,
            ):
                for batch in final_dataset.to_batches():
                    writer.write_batch(batch)

    return result_file_path


@show_total_elapsed_time_decorator
def _download_data(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    pyarrow_filters: Optional[list[Union["Expression", None]]],
    columns_to_download: Optional[list[Optional[list[str]]]],
    working_directory: Path,
    verbosity_mode: "VERBOSITY_MODE",
    max_workers: Optional[int],
) -> list[list[Path]]:
    if pyarrow_filters is not None and len(theme_type_pairs) != len(pyarrow_filters):
        raise ValueError("Pyarrow filters length doesn't match length of theme type pairs.")

    if columns_to_download is not None and len(theme_type_pairs) != len(columns_to_download):
        raise ValueError("Columns to download length doesn't match length of theme type pairs.")

    # force tuple structure
    theme_type_pairs = [(theme_value, type_value) for theme_value, type_value in theme_type_pairs]

    all_row_groups_to_download = _prepare_row_groups_for_download(
        release=release,
        theme_type_pairs=theme_type_pairs,
        geometry_filter=geometry_filter,
        pyarrow_filters=pyarrow_filters,
        columns_to_download=columns_to_download,
        verbosity_mode=verbosity_mode,
    )

    downloaded_parquet_files = _download_prepared_row_groups(
        all_row_groups_to_download=all_row_groups_to_download,
        theme_type_pairs=theme_type_pairs,
        geometry_filter=geometry_filter,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
    )

    filtered_parquet_files = _filter_downloaded_files(
        downloaded_parquet_files=downloaded_parquet_files,
        theme_type_pairs=theme_type_pairs,
        geometry_filter=geometry_filter,
        working_directory=working_directory,
        verbosity_mode=verbosity_mode,
        max_workers=max_workers,
    )

    grouped_parquet_paths = _group_parquet_files(
        filtered_parquet_files=filtered_parquet_files, theme_type_pairs=theme_type_pairs
    )

    return grouped_parquet_paths


def _prepare_row_groups_for_download(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    pyarrow_filters: Optional[list[Union["Expression", None]]],
    columns_to_download: Optional[list[Optional[list[str]]]],
    verbosity_mode: "VERBOSITY_MODE",
) -> list[dict[str, Any]]:
    from overturemaestro.release_index import load_release_index

    all_row_groups_to_download = []

    for idx, (theme_value, type_value) in enumerate(theme_type_pairs):
        dataset_index = load_release_index(
            release=release,
            theme=theme_value,
            type=type_value,
            geometry_filter=geometry_filter,
            verbosity_mode=verbosity_mode,
        )

        dataset_index = (
            dataset_index.explode("row_indexes_ranges")
            .groupby(["filename", "row_group"])["row_indexes_ranges"]
            .apply(lambda x: [pair.tolist() for pair in x.values])
            .reset_index()
        )

        row_groups_to_download = dataset_index[
            ["filename", "row_group", "row_indexes_ranges"]
        ].to_dict(orient="records")

        _pyarrow_filter = pyarrow_filters[idx] if pyarrow_filters else None
        _columns_to_download = columns_to_download[idx] if columns_to_download else None

        for row_group_to_download in row_groups_to_download:
            row_group_to_download["theme"] = theme_value
            row_group_to_download["type"] = type_value
            row_group_to_download["user_defined_pyarrow_filter"] = _pyarrow_filter
            row_group_to_download["columns_to_download"] = _columns_to_download

        all_row_groups_to_download.extend(row_groups_to_download)

    return all_row_groups_to_download


def _download_prepared_row_groups(
    all_row_groups_to_download: list[dict[str, Any]],
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    working_directory: Path,
    verbosity_mode: "VERBOSITY_MODE",
    max_workers: Optional[int],
) -> list[Path]:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from overturemaestro._rich_progress import TrackProgressBar

    total_row_groups = len(all_row_groups_to_download)

    min_no_workers = 8
    no_workers = min(
        max(min_no_workers, multiprocessing.cpu_count() + 4), 32
    )  # minimum 8 workers, but not more than 32

    if max_workers:
        no_workers = min(max_workers, no_workers)

    with (
        TrackProgressBar(verbosity_mode=verbosity_mode) as progress,
        ProcessPoolExecutor(
            max_workers=min(no_workers, total_row_groups),
            mp_context=multiprocessing.get_context("spawn"),
        ) as ex,
    ):
        fn = partial(
            _download_single_parquet_row_group_multiprocessing,
            bbox=geometry_filter.bounds,
            working_directory=working_directory,
        )
        theme_type_task_description = (
            f"{theme_type_pairs[0][0]}/{theme_type_pairs[0][1]}"
            if len(theme_type_pairs) == 1
            else f"{len(theme_type_pairs)} datasets"
        )
        downloaded_parquet_files = list(
            progress.track(
                ex.map(
                    fn,
                    all_row_groups_to_download,
                    chunksize=1,
                ),
                description=f"Downloading parquet files ({theme_type_task_description})",
                total=total_row_groups,
            )
        )

    return downloaded_parquet_files


def _filter_downloaded_files(
    downloaded_parquet_files: list[Path],
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    working_directory: Path,
    verbosity_mode: "VERBOSITY_MODE",
    max_workers: Optional[int],
) -> list[Path]:
    from functools import partial

    from overturemaestro._parquet_multiprocessing import map_parquet_dataset

    if not geometry_filter.equals(geometry_filter.envelope):
        destination_path = working_directory / Path("intersected_data")
        fn = partial(_filter_data_properly, geometry_filter=geometry_filter)
        theme_type_task_description = (
            f"{theme_type_pairs[0][0]}/{theme_type_pairs[0][1]}"
            if len(theme_type_pairs) == 1
            else f"{len(theme_type_pairs)} datasets"
        )
        map_parquet_dataset(
            dataset_path=downloaded_parquet_files,
            destination_path=destination_path,
            function=fn,
            progress_description=f"Filtering data by geometry ({theme_type_task_description})",
            report_progress_as_text=False,
            verbosity_mode=verbosity_mode,
            max_workers=max_workers,
        )

        filtered_parquet_files = list(destination_path.glob("**/*.parquet"))
    else:
        filtered_parquet_files = downloaded_parquet_files

    return filtered_parquet_files


def _group_parquet_files(
    filtered_parquet_files: list[Path],
    theme_type_pairs: list[tuple[str, str]],
) -> list[list[Path]]:
    if len(theme_type_pairs) == 1:
        grouped_parquet_paths = [filtered_parquet_files]
    else:
        grouped_parquet_paths = [[] for _ in theme_type_pairs]
        for parquet_file in filtered_parquet_files:
            import pyarrow.parquet as pq

            metadata = pq.read_schema(parquet_file).metadata
            theme_value = metadata[b"_theme"].decode()
            type_value = metadata[b"_type"].decode()
            idx = theme_type_pairs.index((theme_value, type_value))
            grouped_parquet_paths[idx].append(parquet_file)

    return grouped_parquet_paths


def _download_single_parquet_row_group_multiprocessing(
    params: dict[str, Any],
    bbox: tuple[float, float, float, float],
    working_directory: Path,
) -> Path:
    from pyarrow.lib import ArrowInvalid

    retries = 10
    while retries > 0:
        try:
            downloaded_path = _download_single_parquet_row_group(
                **params,
                bbox=bbox,
                working_directory=working_directory,
            )
            return downloaded_path
        except (MissingColumnError, ArrowInvalid):
            raise
        except Exception:  # pragma: no cover
            retries -= 1
            if retries == 0:
                raise

    raise Exception()


def _download_single_parquet_row_group(
    theme: str,
    type: str,
    filename: str,
    row_group: int,
    row_indexes_ranges: list[list[int]],
    bbox: tuple[float, float, float, float],
    user_defined_pyarrow_filter: Optional["Expression"],
    columns_to_download: Optional[list[str]],
    working_directory: Path,
) -> Path:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as fs
    from overturemaps.cli import get_writer
    from overturemaps.core import geoarrow_schema_adapter

    from overturemaestro._geometry_clustering import decompress_ranges

    row_indexes = decompress_ranges(row_indexes_ranges)
    xmin, ymin, xmax, ymax = bbox
    pyarrow_filter = (
        (pc.field("bbox", "xmin") < xmax)
        & (pc.field("bbox", "xmax") > xmin)
        & (pc.field("bbox", "ymin") < ymax)
        & (pc.field("bbox", "ymax") > ymin)
    )

    if user_defined_pyarrow_filter is not None:
        pyarrow_filter = pyarrow_filter & user_defined_pyarrow_filter

    fragment_manual = ds.ParquetFileFormat().make_fragment(
        file=filename,
        filesystem=fs.S3FileSystem(anonymous=True, region="us-west-2"),
        row_groups=[row_group],
    )
    geoarrow_full_schema = geoarrow_schema_adapter(fragment_manual.physical_schema)
    metadata = geoarrow_full_schema.metadata or {}
    metadata["_theme"] = theme
    metadata["_type"] = type
    geoarrow_full_schema = geoarrow_full_schema.with_metadata(metadata)
    geoarrow_schema_filtered = geoarrow_full_schema

    filtering_columns = None

    columns_to_remove = set()

    if columns_to_download:
        nonexistent_columns = set(columns_to_download).difference(geoarrow_full_schema.names)

        if nonexistent_columns:
            raise MissingColumnError(
                f"Cannot download given columns: {', '.join(sorted(nonexistent_columns))}"
            )

        if "geometry" not in columns_to_download:
            columns_to_download.append("geometry")

        # Create list of columns used to filter data by bbox
        columns_required_for_filtering = {"bbox"}
        if user_defined_pyarrow_filter is not None:
            columns_required_for_filtering = columns_required_for_filtering.union(
                _get_filtering_columns_from_pyarrow_filter(
                    schema=geoarrow_full_schema, pyarrow_filter=user_defined_pyarrow_filter
                )
            )

        filtering_columns = [
            column_name
            for column_name in geoarrow_full_schema.names
            if column_name in columns_to_download or column_name in columns_required_for_filtering
        ]

        # Reorder list of columns to download to match pyarrow schema
        columns_to_download = [
            column_name
            for column_name in geoarrow_full_schema.names
            if column_name in columns_to_download
        ]

        # Remove unused columns from schema
        for column_name in geoarrow_schema_filtered.names:
            if column_name in columns_to_download:
                continue
            geoarrow_schema_filtered = geoarrow_schema_filtered.remove(
                geoarrow_schema_filtered.get_field_index(column_name)
            )

        columns_to_remove = set(filtering_columns).difference(columns_to_download)

    file_path = working_directory / _generate_row_group_file_name(
        filename, row_group, row_indexes_ranges, bbox
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with get_writer(
        output_format="geoparquet", path=file_path, schema=geoarrow_schema_filtered
    ) as writer:
        writer.write_table(
            fragment_manual.scanner(schema=geoarrow_full_schema, columns=filtering_columns)
            .take(row_indexes)
            .filter(pyarrow_filter)
            .drop_columns(list(columns_to_remove))
        )

    return file_path


def _get_filtering_columns_from_pyarrow_filter(
    schema: "Schema", pyarrow_filter: "Expression"
) -> set[str]:
    import re

    import pyarrow.substrait as pas

    parsed_expression = pas.deserialize_expressions(
        pyarrow_filter.to_substrait(schema=schema)
    ).expressions["expression"]

    pattern = r"FieldPath\((\d+(?: \d+)*)\)"
    field_paths = re.findall(pattern, str(parsed_expression))

    column_names = set(schema.field(int(field_path.split()[0])).name for field_path in field_paths)
    return column_names


def _generate_row_group_file_name(
    filename: str,
    row_group: int,
    row_indexes_ranges: list[list[int]],
    bbox: tuple[float, float, float, float],
) -> Path:
    import hashlib

    row_indexes_ranges_str = str(
        [
            (int(range_pair[0]), int(range_pair[1]))  # noqa: FURB123
            for range_pair in sorted(row_indexes_ranges, key=operator.itemgetter(0))
        ]
    )
    h = hashlib.new("sha256")
    h.update(row_indexes_ranges_str.encode())
    row_indexes_ranges_hash_part = h.hexdigest()

    h = hashlib.new("sha256")
    h.update(",".join(str(round(value, 7)) for value in bbox).encode())
    bbox_hash_part = h.hexdigest()

    stripped_filename_part = Path(filename.split("/", 2)[-1])
    stem = stripped_filename_part.stem.split(".")[0]
    stem = f"{stem}_{row_group}_{row_indexes_ranges_hash_part[:8]}_{bbox_hash_part[:8]}"
    return stripped_filename_part.with_stem(stem)


def _generate_result_file_path(
    release: str,
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    pyarrow_filter: Optional["Expression"],
    columns_to_download: Optional[list[str]],
    # keep_all_tags: bool,
    # explode_tags: bool,
    # filter_osm_ids: list[str],
    # save_as_wkt: bool,
) -> Path:
    import hashlib

    clipping_geometry_hash_part = _generate_geometry_hash(geometry_filter)

    pyarrow_filter_hash_part = "nofilter"
    if pyarrow_filter is not None:
        h = hashlib.new("sha256")
        h.update(str(pyarrow_filter).encode())
        pyarrow_filter_hash_part = h.hexdigest()

    columns_hash_part = ""
    if columns_to_download is not None:
        h = hashlib.new("sha256")
        h.update(str(sorted(columns_to_download)).encode())
        columns_hash_part = f"_{h.hexdigest()}"

    return (
        Path(release)
        / f"theme={theme}"
        / f"type={type}"
        / f"{clipping_geometry_hash_part}_{pyarrow_filter_hash_part}{columns_hash_part}.parquet"
    )


def _generate_geometry_hash(geometry_filter: "BaseGeometry") -> str:
    import hashlib

    import shapely.wkt as wktlib

    oriented_geometry = _get_oriented_geometry_filter(geometry_filter)
    h = hashlib.new("sha256")
    h.update(wktlib.dumps(oriented_geometry).encode())
    clipping_geometry_hash_part = h.hexdigest()

    return clipping_geometry_hash_part


def _get_oriented_geometry_filter(geometry_filter: "BaseGeometry") -> "BaseGeometry":
    import itertools

    from shapely import LinearRing, Polygon
    from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

    geometry = geometry_filter

    if isinstance(geometry, LinearRing):
        # https://stackoverflow.com/a/73073112/7766101
        new_coords = []
        if geometry.is_ccw:
            perimeter = list(geometry.coords)
        else:
            perimeter = list(geometry.coords)[::-1]
        smallest_point = sorted(perimeter)[0]
        double_iteration = itertools.chain(perimeter[:-1], perimeter)
        for point in double_iteration:
            if point == smallest_point:
                new_coords.append((round(point[0], 7), round(point[1], 7)))
                while len(new_coords) < len(perimeter):
                    next_point = next(double_iteration)
                    new_coords.append((round(next_point[0], 7), round(next_point[1], 7)))
                break
        return LinearRing(new_coords)
    if isinstance(geometry, Polygon):
        oriented_exterior = _get_oriented_geometry_filter(geometry.exterior)
        oriented_interiors = [
            cast(BaseGeometry, _get_oriented_geometry_filter(interior))
            for interior in geometry.interiors
        ]
        return Polygon(
            oriented_exterior,
            sorted(oriented_interiors, key=lambda geom: (geom.centroid.x, geom.centroid.y)),
        )
    elif isinstance(geometry, BaseMultipartGeometry):
        oriented_geoms = [
            cast(BaseGeometry, _get_oriented_geometry_filter(geom)) for geom in geometry.geoms
        ]
        return geometry.__class__(
            sorted(oriented_geoms, key=lambda geom: (geom.centroid.x, geom.centroid.y))
        )

    return geometry


def _filter_data_properly(
    parquet_filename: str,
    parquet_row_group: int,
    pyarrow_table: Any,
    geometry_filter: "BaseGeometry",
) -> Any:
    import geopandas as gpd
    from geoarrow.rust.core import GeometryArray
    from shapely import STRtree

    matching_indexes = STRtree(
        gpd.GeoSeries.from_arrow(
            GeometryArray.from_arrow(pyarrow_table["geometry"].combine_chunks())
        )
    ).query(geometry_filter, predicate="intersects")
    return pyarrow_table.take(matching_indexes)
