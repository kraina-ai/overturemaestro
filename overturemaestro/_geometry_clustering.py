import itertools
from collections.abc import Generator, Iterable
from typing import Any

EARTH_RADIUS_KM = 6371
CLUSTERING_THRESHOLD = 20 / EARTH_RADIUS_KM  # calculate 20 kilometers threshold


def calculate_row_group_bounding_box(
    parquet_filename: str, parquet_row_group: int, pyarrow_table: Any
) -> Any:
    import polars as pl
    from sklearn.cluster import Birch

    df = (
        pl.from_arrow(pyarrow_table)
        .with_row_index(name="row_index")
        .unnest("bbox")
        .select(
            [
                pl.col("row_index"),
                pl.col("xmin"),
                pl.col("xmax"),
                pl.col("ymin"),
                pl.col("ymax"),
                pl.sum_horizontal("xmin", "xmax").truediv(2).radians().alias("x_rad"),
                pl.sum_horizontal("ymin", "ymax").truediv(2).radians().alias("y_rad"),
            ]
        )
    )

    cluster_labels = (
        Birch(n_clusters=None, threshold=CLUSTERING_THRESHOLD).fit(df[["x_rad", "y_rad"]]).labels_
    )

    return (
        df.with_columns(pl.Series(name="label", values=cluster_labels))
        .group_by("label")
        .agg(
            pl.col("xmin").min(),
            pl.col("xmax").max(),
            pl.col("ymin").min(),
            pl.col("ymax").max(),
            pl.col("row_index").alias("row_indexes"),
        )
        .select(
            pl.lit(parquet_filename).alias("filename"),
            pl.lit(parquet_row_group).alias("row_group"),
            pl.col("xmin"),
            pl.col("xmax"),
            pl.col("ymin"),
            pl.col("ymax"),
            pl.col("row_indexes")
            .map_elements(
                lambda lst: list(calculate_ranges(lst)), return_dtype=pl.List(pl.List(pl.Int64))
            )
            .cast(pl.List(pl.List(pl.UInt32)))
            .alias("row_indexes_ranges"),
        )
    ).to_arrow()


# https://stackoverflow.com/a/4629241
def calculate_ranges(i: Iterable[int]) -> Generator[tuple[int, int]]:
    for _a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b_lst = list(b)
        yield b_lst[0][1], b_lst[-1][1]


def decompress_ranges(ranges: list[list[int]]) -> list[int]:
    return list(
        itertools.chain.from_iterable(
            range(range_bounds[0], range_bounds[1] + 1) for range_bounds in ranges
        )
    )
