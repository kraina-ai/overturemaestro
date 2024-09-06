"""
Data downloader utilities.

Functions used to download Overture Maps data before local filtering.
"""

import multiprocessing
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

if TYPE_CHECKING:
    from pyarrow.compute import Expression
    from shapely.geometry.base import BaseGeometry

pyarrow_expression = tuple[Any, Any, Any]
pyarrow_filters = Union["Expression", list[pyarrow_expression], list[list[pyarrow_expression]]]

__all__ = [
    "download_data",
    "download_data_for_multiple_types",
]


def download_data_for_multiple_types(
    release: str,
    theme_type_pairs: list[tuple[str, str]],
    geometry_filter: "BaseGeometry",
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
) -> list[Path]:
    """
    Downloads the data for the given release for multiple types.

    Args:
        release (str): Release version.
        theme_type_pairs (list[tuple[str, str]]): Pairs of themes and types of the dataset.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".

    Returns:
        list[Path]: List of saved Geoparquet files paths.
    """
    return [
        download_data(
            release=release,
            theme=theme_value,
            type=type_value,
            geometry_filter=geometry_filter,
            ignore_cache=ignore_cache,
            working_directory=working_directory,
        )
        for theme_value, type_value in theme_type_pairs
    ]


def download_data(
    release: str,
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    pyarrow_filter: Optional[pyarrow_filters] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    ignore_cache: bool = False,
    working_directory: Union[str, Path] = "files",
) -> Path:
    """
    Downloads the data for the given release.

    Args:
        release (str): Release version.
        theme (str): Theme of the dataset.
        type (str): Type of the dataset.
        geometry_filter (BaseGeometry): Geometry used to filter data.
        pyarrow_filter (Optional[pyarrow_filters], optional): Filters to apply on a pyarrow dataset.
            Can be pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]]. Defaults to None.
        result_file_path (Union[str, Path], optional): Where to save
            the geoparquet file. If not provided, will be generated based on hashes
            from filters. Defaults to `None`.
        ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
            Defaults to False.
        working_directory (Union[str, Path], optional): Directory where to save
            the downloaded `*.parquet` files. Defaults to "files".

    Returns:
        Path: Saved Geoparquet file path.
    """
    import tempfile

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
        )

    result_file_path = Path(result_file_path)

    if not result_file_path.exists() or ignore_cache:
        with tempfile.TemporaryDirectory(
            dir=Path(Path(working_directory)).resolve()
        ) as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)
            _download_data(
                release=release,
                theme=theme,
                type=type,
                geometry_filter=geometry_filter,
                pyarrow_filter=pyarrow_filter,
                result_file_path=result_file_path,
                work_directory=tmp_dir_path,
            )
    return result_file_path


def _download_data(
    release: str,
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    pyarrow_filter: Optional["Expression"],
    result_file_path: Path,
    work_directory: Path,
) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from overturemaestro._parquet_multiprocessing import map_parquet_dataset
    from overturemaestro._rich_progress import TrackProgressBar
    from overturemaestro.release_index import load_release_index

    dataset_index = load_release_index(release, theme, type, geometry_filter)

    dataset_index = (
        dataset_index.explode("row_indexes_ranges")
        .groupby(["filename", "row_group"])["row_indexes_ranges"]
        .apply(list)
        .reset_index()
    )

    row_groups_to_download = dataset_index[["filename", "row_group", "row_indexes_ranges"]].to_dict(
        orient="records"
    )

    no_cpus = multiprocessing.cpu_count()
    min_no_workers = 32 if no_cpus >= 8 else 16
    no_workers = min(
        max(min_no_workers, multiprocessing.cpu_count() + 4), 64
    )  # minimum 16 / 32 workers, but not more than 64

    with TrackProgressBar() as progress:
        total_row_groups = len(row_groups_to_download)
        with ProcessPoolExecutor(max_workers=min(no_workers, total_row_groups)) as ex:
            fn = partial(
                _download_single_parquet_row_group_multiprocessing,
                bbox=geometry_filter.bounds,
                pyarrow_filter=pyarrow_filter,
                work_directory=work_directory,
            )
            downloaded_parquet_files = list(
                progress.track(
                    ex.map(
                        fn,
                        row_groups_to_download,
                        chunksize=1,
                    ),
                    description=f"Downloading parquet files ({theme}/{type})",
                    total=total_row_groups,
                )
            )

    if not geometry_filter.equals(geometry_filter.envelope):
        destination_path = work_directory / Path("intersected_data")

        fn = partial(_filter_data_properly, geometry_filter=geometry_filter)
        map_parquet_dataset(
            dataset_path=downloaded_parquet_files,
            destination_path=destination_path,
            function=fn,
            progress_description=f"Filtering data by geometry ({theme}/{type})",
            report_progress_as_text=False,
        )

        filtered_parquet_files = list(destination_path.glob("*.parquet"))
    else:
        filtered_parquet_files = downloaded_parquet_files

    final_dataset = ds.dataset(filtered_parquet_files)

    result_file_path.parent.mkdir(exist_ok=True, parents=True)

    with pq.ParquetWriter(result_file_path, final_dataset.schema) as writer:
        for batch in final_dataset.to_batches():
            writer.write_batch(batch)


def _download_single_parquet_row_group_multiprocessing(
    params: dict[str, Any],
    bbox: tuple[float, float, float, float],
    pyarrow_filter: Optional["Expression"],
    work_directory: Path,
) -> Path:
    return _download_single_parquet_row_group(
        **params, bbox=bbox, pyarrow_filter=pyarrow_filter, work_directory=work_directory
    )


def _download_single_parquet_row_group(
    filename: str,
    row_group: int,
    row_indexes_ranges: list[list[int]],
    bbox: tuple[float, float, float, float],
    pyarrow_filter: Optional["Expression"],
    work_directory: Path,
) -> Path:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as fs
    from overturemaps.cli import get_writer
    from overturemaps.core import geoarrow_schema_adapter

    from overturemaestro._geometry_clustering import decompress_ranges

    row_indexes = decompress_ranges(row_indexes_ranges)
    xmin, ymin, xmax, ymax = bbox
    bbox_filter = (
        (pc.field("bbox", "xmin") < xmax)
        & (pc.field("bbox", "xmax") > xmin)
        & (pc.field("bbox", "ymin") < ymax)
        & (pc.field("bbox", "ymax") > ymin)
    )

    if pyarrow_filter is not None:
        bbox_filter = bbox_filter & pyarrow_filter

    fragment_manual = ds.ParquetFileFormat().make_fragment(
        file=filename,
        filesystem=fs.S3FileSystem(anonymous=True, region="us-west-2"),
        row_groups=[row_group],
    )
    geoarrow_schema = geoarrow_schema_adapter(fragment_manual.physical_schema)

    file_path = work_directory / _generate_row_group_file_name(
        filename, row_group, row_indexes_ranges, bbox
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with get_writer(output_format="geoparquet", path=file_path, schema=geoarrow_schema) as writer:
        writer.write_table(
            fragment_manual.scanner(schema=geoarrow_schema).take(row_indexes).filter(bbox_filter)
        )

    return file_path


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


def _filter_data_properly(
    parquet_filename: str,
    parquet_row_group: int,
    pyarrow_table: Any,
    geometry_filter: "BaseGeometry",
) -> Any:
    import geopandas as gpd
    import numpy as np
    from geoarrow.rust.core import WKBArray
    from shapely import Point, STRtree, get_coordinates

    geoseries = gpd.GeoSeries.from_arrow(
        WKBArray.from_arrow(pyarrow_table["geometry"].combine_chunks())
    )

    # First pass - find all simple examples - any point inside geometry filter
    exploded_coords = (
        geoseries.apply(get_coordinates).rename("coords").reset_index().explode("coords")
    )
    tree = STRtree(exploded_coords["coords"].apply(Point))
    matching_indexes = (
        exploded_coords["index"].iloc[tree.query(geometry_filter, predicate="intersects")].unique()
    )

    # Second pass - find harder examples - geometries bigger than filter or cross intersecting
    non_matching_indexes = np.setdiff1d(exploded_coords["index"].unique(), matching_indexes)
    non_matching_geometries = geoseries.iloc[non_matching_indexes].reset_index()
    matching_geometries = non_matching_geometries[
        non_matching_geometries.intersects(geometry_filter)
    ]

    joined_matching_indexes = np.concat([matching_indexes, matching_geometries["index"]])
    return pyarrow_table.take(joined_matching_indexes)


def _generate_result_file_path(
    release: str,
    theme: str,
    type: str,
    geometry_filter: "BaseGeometry",
    pyarrow_filter: Optional["Expression"],
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

    return (
        Path(release)
        / f"theme={theme}"
        / f"type={type}"
        / f"{clipping_geometry_hash_part}_{pyarrow_filter_hash_part}.parquet"
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
