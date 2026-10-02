"""
Build a GeoParquet index file from a GeoParquet dataset.

Based on code by Jacob Wasserman. Calculates a lookup file with bounding boxes per parquet file row
group.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import pyarrow.fs as fs
import pyarrow.parquet as pq

from overturemaestro._rich_progress import VERBOSITY_MODE, TrackProgressBar, TrackProgressSpinner

INDEX_COLUMNS = ("filename", "row_group", "xmin", "ymin", "xmax", "ymax")


def get_rects_parallel(
    dataset_path: Union[str, Path, list[str], list[Path]],
    filesystem: Optional[fs.FileSystem] = None,
    verbosity_mode: VERBOSITY_MODE = "transient",
) -> pd.DataFrame:
    """
    Calculate bounding boxes for each row group of a parquet file.

    Args:
        dataset_path (Union[str, list[str]]): Path(s) of the parquet dataset.
        filesystem (Optional[fs.FileSystem], optional): Filesystem for the dataset.
            Defaults to None.
        verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
            verbosity mode. Can be one of: silent, transient and verbose. Silent disables
            output completely. Transient tracks progress, but removes output after finished.
            Verbose leaves all progress outputs in the stdout. Defaults to "transient".

    Returns:
        pd.DataFrame: DataFrame with calculated bounding boxes.
    """
    with TrackProgressSpinner("Reading all release parquet files", verbosity_mode=verbosity_mode):
        dataset = pq.ParquetDataset(dataset_path, filesystem=filesystem)
        total_files = len(dataset.files)

        no_cpus = multiprocessing.cpu_count()
        min_no_workers = 32 if no_cpus >= 8 else 16
        no_scan_workers = min(max(min_no_workers, no_cpus + 4), 64)

    with (
        TrackProgressBar(verbosity_mode=verbosity_mode) as progress,
        ProcessPoolExecutor(max_workers=min(no_scan_workers, total_files)) as ex,
    ):
        fn = partial(_parquet_path_to_dataframe, filesystem=dataset.filesystem)
        gdfs = list(
            progress.track(
                ex.map(fn, dataset.files),
                description="Generating Overture Maps release cache index",
                total=total_files,
            )
        )

    nonempty_gdfs = [gdf for gdf in gdfs if not gdf.empty]
    if not nonempty_gdfs:
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    df = pd.DataFrame(pd.concat(nonempty_gdfs, ignore_index=True))
    return df


def _parquet_path_to_dataframe(
    path: str, filesystem: Optional[fs.FileSystem] = None
) -> pd.DataFrame:
    metadata = pq.read_metadata(path, filesystem=filesystem)
    df = _metadata_to_dataframe(metadata, path, filesystem=filesystem)

    return df


def _metadata_to_dataframe(
    metadata: pq.FileMetaData, path: str, filesystem: Optional[fs.FileSystem] = None
) -> pd.DataFrame:
    rows = []

    def row(
        path: str, row_group_idx: int, bbox: tuple[float, float, float, float]
    ) -> dict[str, Any]:
        return {
            "filename": path,
            "row_group": row_group_idx,
            "xmin": bbox[0],
            "ymin": bbox[1],
            "xmax": bbox[2],
            "ymax": bbox[3],
        }

    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)

        if row_group.num_rows == 0:
            continue

        bbox_indices = _get_bbox_column_indices(row_group)
        statistics = [row_group.column(bbox_index).statistics for bbox_index in bbox_indices]

        if any(
            column_statistics is None
            or column_statistics.min is None
            or column_statistics.max is None
            for column_statistics in statistics
        ):
            xmin, ymin, xmax, ymax = _calculate_bbox_from_row_group(path, i, filesystem)
        else:
            xmin = statistics[0].min
            ymin = statistics[1].min
            xmax = statistics[2].max
            ymax = statistics[3].max

        rows.append(row(path, i, (xmin, ymin, xmax, ymax)))

    df = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))

    return df


def _calculate_bbox_from_row_group(
    path: str, row_group_idx: int, filesystem: Optional[fs.FileSystem] = None
) -> tuple[float, float, float, float]:
    """Calculate a bounding box of a row group directly from its data."""
    import pyarrow.compute as pc

    table = pq.ParquetFile(path, filesystem=filesystem).read_row_group(
        row_group_idx, columns=["bbox"]
    )
    bbox = table.column("bbox").combine_chunks()

    return (
        pc.min(bbox.field("xmin")).as_py(),
        pc.min(bbox.field("ymin")).as_py(),
        pc.max(bbox.field("xmax")).as_py(),
        pc.max(bbox.field("ymax")).as_py(),
    )


def _get_bbox_column_indices(row_group: pq.RowGroupMetaData) -> tuple[int, int, int, int]:
    columns = {row_group.column(i).path_in_schema: i for i in range(row_group.num_columns)}
    return (columns["bbox.xmin"], columns["bbox.ymin"], columns["bbox.xmax"], columns["bbox.ymax"])
