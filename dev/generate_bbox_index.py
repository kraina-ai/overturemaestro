# ruff: noqa
# type: ignore
"""
Build a GeoParquet file from a GeoParquet dataset.

Code by Jacob Wasserman.
The schema is:
path: string
row_group_idx: int
bbox: struct(minx: double, miny: double, maxx: double, maxy: double)
geometry: WKB geometry
"""


# from dataclasses import dataclass
# import json
import logging
import queue
import threading
from pathlib import Path

import geopandas as gpd

# import adlfs
import pandas as pd
import pyarrow.fs as fs
import pyarrow.parquet as pq
import shapely.geometry

logging.basicConfig(level=logging.INFO)


def parquet_path_to_geodataframe(path, filesystem=None):
    metadata = pq.read_metadata(path, filesystem=filesystem)
    gdf = metadata_to_geodataframe(metadata, path, include_row_groups=False)

    return gdf


def metadata_to_geodataframe(metadata, path, include_row_groups=False):
    rows = []

    def row(path, row_group_idx, bbox, num_rows):
        return {
            "row_group_index": row_group_idx,
            "filename": path,
            "bbox": {
                "xmin": bbox[0],
                "ymin": bbox[1],
                "xmax": bbox[2],
                "ymax": bbox[3],
            },
            "geometry": shapely.geometry.box(*bbox),
            "num_rows": num_rows,
        }

    bbox_indices = get_bbox_column_indices(metadata.row_group(0))
    file_bbox = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        bbox_indices = get_bbox_column_indices(row_group)

        xmin = row_group.column(bbox_indices[0]).statistics.min
        ymin = row_group.column(bbox_indices[1]).statistics.min
        xmax = row_group.column(bbox_indices[2]).statistics.max
        ymax = row_group.column(bbox_indices[3]).statistics.max

        file_bbox[0] = min(file_bbox[0], xmin)
        file_bbox[1] = min(file_bbox[1], ymin)
        file_bbox[2] = max(file_bbox[2], xmax)
        file_bbox[3] = max(file_bbox[3], ymax)

        if include_row_groups:
            rows.append(row(path, i, [xmin, ymin, xmax, ymax], row_group.num_rows))

    # Let's use -1 as the row group index for the whole file
    rows.append(row(path, -1, file_bbox, metadata.num_rows))

    gdf = gpd.GeoDataFrame(rows)

    return gdf


def get_bbox_column_indices(row_group) -> tuple[int, int, int, int]:
    columns = {row_group.column(i).path_in_schema: i for i in range(row_group.num_columns)}
    bbox_indices = []
    for name in ("bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax"):
        bbox_indices.append(columns[name])
    return bbox_indices


def get_rects(ds: pq.ParquetDataset):
    gdfs = []
    for f in ds.files:
        file_gdf = parquet_path_to_geodataframe(f, filesystem=ds.filesystem)
        gdfs.append(file_gdf)

    gdf = gpd.GeoDataFrame(pd.concat(gdfs))
    return gdf


def worker(name, filesystem, work_queue, result_queue):
    logging.info(f"starting worker {name}")
    while True:
        try:
            path = work_queue.get_nowait()
        except queue.Empty:
            break

        logging.info(f"worker {name} getting {path}")
        gdf = parquet_path_to_geodataframe(path, filesystem=filesystem)
        result_queue.put(gdf)
        work_queue.task_done()
        logging.info(f"worker {name} processed {path}")


def get_rects_parallel(ds: pq.ParquetDataset, n_threads: int = 10):
    work_queue = queue.Queue()
    result_queue = queue.Queue()

    for f in ds.files:
        work_queue.put_nowait(f)

    for i in range(n_threads):
        threading.Thread(
            target=worker,
            args=(str(i), ds.filesystem, work_queue, result_queue),
        ).start()

    gdfs = []
    n = 0
    while n < len(ds.files):
        gdfs.append(result_queue.get())
        result_queue.task_done()
        n += 1
    assert result_queue.empty()

    logging.info("workers complete. Merging dataframes.")
    gdf = gpd.GeoDataFrame(pd.concat(gdfs))
    return gdf


def main(theme, type):
    f_name = Path(f"_index_{theme}_{type}.parquet")
    if not f_name.exists():
        ds = pq.ParquetDataset(
            f"overturemaps-us-west-2/release/2024-09-18.0/theme={theme}/type={type}",
            filesystem=fs.S3FileSystem(anonymous=True, region="us-west-2"),
        )
        gdf = get_rects_parallel(ds)
        gdf = gdf.drop(["row_group_index", "num_rows"], axis=1)
        gdf.to_parquet(f_name, index=False)


if __name__ == "__main__":
    main("buildings", "building")
    main("transportation", "segment")
