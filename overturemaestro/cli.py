"""CLI module for the OvertureMaestro functions."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, cast

import click
import typer

from overturemaestro._geopandas_api_version import GEOPANDAS_NEW_API

if TYPE_CHECKING:  # pragma: no cover
    from overturemaestro._rich_progress import VERBOSITY_MODE

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, rich_markup_mode="rich")


def _version_callback(value: bool) -> None:
    if value:
        from overturemaestro import __app_name__, __version__

        typer.echo(f"{__app_name__} {__version__}")
        raise typer.Exit()


def _display_release_versions_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        from rich import print as rprint
        from rich.table import Table

        from overturemaestro import get_available_release_versions

        release_versions = get_available_release_versions()

        table = Table()
        table.add_column("Release Version", no_wrap=True)
        for release_version in release_versions:
            table.add_row(release_version)

        rprint(table)

        raise typer.Exit()


def _display_theme_type_pairs_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        from rich import print as rprint
        from rich.table import Table

        from overturemaestro import get_available_theme_type_pairs, get_newest_release_version

        param_values = {p.name: p.default for p in ctx.command.params}
        param_values.update(ctx.params)
        release_version = cast(
            str, param_values.get("release_version") or get_newest_release_version()
        )
        theme_type_pairs = get_available_theme_type_pairs(release=release_version)

        table = Table(title=f"{release_version} release")
        table.add_column("Theme", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        for theme_value, type_value in theme_type_pairs:
            table.add_row(theme_value, type_value)

        rprint(table)
        raise typer.Exit()


def _path_callback(ctx: typer.Context, value: Path) -> Path:
    if not Path(value).exists():
        raise typer.BadParameter(f"File not found error: {value}")
    return value


def _empty_path_callback(ctx: typer.Context, value: Path) -> Optional[Path]:
    if not value:
        return None
    return _path_callback(ctx, value)


class BboxGeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in bounding box form."""

    name = "BBOX"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        try:
            from shapely import box

            bbox_values = [float(x.strip()) for x in value.split(",")]
            return box(*bbox_values)
        except ValueError:  # ValueError raised when passing non-numbers to float()
            raise typer.BadParameter(
                "Cannot parse provided bounding box."
                " Valid value must contain 4 floating point numbers"
                " separated by commas."
            ) from None


class WktGeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in WKT form."""

    name = "TEXT (WKT)"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None
        try:
            from shapely import from_wkt

            return from_wkt(value)
        except Exception:
            raise typer.BadParameter("Cannot parse provided WKT") from None


class GeoJsonGeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in GeoJSON form."""

    name = "TEXT (GeoJSON)"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None
        try:
            from shapely import from_geojson

            return from_geojson(value)
        except Exception:
            raise typer.BadParameter("Cannot parse provided GeoJSON") from None


class GeoFileGeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in geo file form."""

    name = "PATH"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None

        value = _path_callback(ctx=ctx, value=value)

        try:
            import geopandas as gpd

            gdf = gpd.read_file(value)
            if GEOPANDAS_NEW_API:
                return gdf.union_all()
            else:
                return gdf.unary_union
        except Exception:
            raise typer.BadParameter("Cannot parse provided geo file") from None


class GeocodeGeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in string Nominatim query form."""

    name = "TEXT"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None

        try:
            from overturemaestro.geocode import geocode_to_geometry

            return geocode_to_geometry(value)
        except Exception:
            raise typer.BadParameter("Cannot geocode provided Nominatim query") from None


class GeohashGeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in string Nominatim query form."""

    name = "TEXT (Geohash)"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None

        try:
            import geopandas as gpd
            from geohash import bbox as geohash_bbox
            from shapely.geometry import box

            geometries = []
            for geohash in value.split(","):
                bounds = geohash_bbox(geohash.strip())
                geometries.append(
                    box(minx=bounds["w"], miny=bounds["s"], maxx=bounds["e"], maxy=bounds["n"])
                )
            if GEOPANDAS_NEW_API:
                return gpd.GeoSeries(geometries).union_all()
            else:
                return gpd.GeoSeries(geometries).unary_union
        except Exception:
            raise
            # raise typer.BadParameter(f"Cannot parse provided Geohash value: {geohash}") from None


class H3GeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in string Nominatim query form."""

    name = "TEXT (H3)"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None

        try:
            import geopandas as gpd
            import h3
            from shapely.geometry import Polygon

            geometries = []  # noqa: FURB138
            for h3_cell in value.split(","):
                geometries.append(
                    Polygon([coords[::-1] for coords in h3.cell_to_boundary(h3_cell.strip())])
                )
            if GEOPANDAS_NEW_API:
                return gpd.GeoSeries(geometries).union_all()
            else:
                return gpd.GeoSeries(geometries).unary_union
        except Exception as ex:
            raise typer.BadParameter(f"Cannot parse provided H3 values: {value}") from ex


class S2GeometryParser(click.ParamType):  # type: ignore
    """Parser for geometry in string Nominatim query form."""

    name = "TEXT (S2)"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None

        try:
            import geopandas as gpd
            from s2 import s2
            from shapely.geometry import Polygon

            geometries = []  # noqa: FURB138
            for s2_index in value.split(","):
                geometries.append(
                    Polygon(s2.s2_to_geo_boundary(s2_index.strip(), geo_json_conformant=True))
                )
            if GEOPANDAS_NEW_API:
                return gpd.GeoSeries(geometries).union_all()
            else:
                return gpd.GeoSeries(geometries).unary_union
        except Exception:
            raise typer.BadParameter(f"Cannot parse provided S2 value: {s2_index}") from None


class PyArrowExpressionParser(click.ParamType):  # type: ignore
    """Parser for geometry in string Nominatim query form."""

    name = "FILTER"

    def convert(self, value, param=None, ctx=None):  # type: ignore
        """Convert parameter value."""
        if not value:
            return None

        try:
            parts = value.split()

            if len(parts) != 3:
                raise typer.BadParameter(
                    "Provided expression is not in a required format (3 elements in a string)."
                ) from None

            from pyarrow.parquet import filters_to_expression

            # Parse numbers
            if parts[2].replace(".", "", 1).isdigit():
                parts[2] = float(parts[2])

            # TODO: optional - add support for "in" operator with list of values

            # Check if columns are nested
            column_part = parts[0]
            if "." in column_part:
                columns = tuple(column_part.split("."))
                parts = [[columns, parts[1], parts[2]]]
            elif "," in column_part:
                import warnings

                warnings.warn(
                    (
                        "Found columns split by a comma. New suggested format are"
                        " column names separated by a dot. For compatilibity reasons"
                        " OvertureMaestro will split columns separated by a comma,"
                        " but it will result in this warning."
                    ),
                    DeprecationWarning,
                    stacklevel=0,
                )
                columns = tuple(column_part.split(","))
                parts = [[columns, parts[1], parts[2]]]

            pyarrow_filter = filters_to_expression([parts])
            return pyarrow_filter

        except Exception as ex:
            raise typer.BadParameter(
                f"Cannot parse provided PyArrow Expression: {value}."
                " Required format: <column_name> <operator> <value>."
            ) from ex


@app.command()  # type: ignore
def main(
    theme_value: Annotated[
        Optional[str],
        typer.Argument(
            help="Data [bold yellow]theme[/bold yellow] value",
            metavar="theme",
            show_default=False,
        ),
    ] = None,
    type_value: Annotated[
        Optional[str],
        typer.Argument(
            help="Feature [bold yellow]type[/bold yellow] within [yellow]theme[/yellow]",
            metavar="type",
            show_default=False,
        ),
    ] = None,
    release_version: Annotated[
        Optional[str],
        typer.Option(
            "--release-version",
            "--release",
            help=(
                "OvertureMaps dataset release version."
                " If not provided, will automatically select the newest"
                " available version."
            ),
            is_eager=True,
            show_default=False,
        ),
    ] = None,
    geom_filter_bbox: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the"
                " [bold dark_orange]bounding box[/bold dark_orange] format - 4 floating point"
                " numbers separated by commas."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=BboxGeometryParser(),
            show_default=False,
        ),
    ] = None,
    geom_filter_file: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the"
                " [bold dark_orange]file[/bold dark_orange] format - any that can be opened by"
                " GeoPandas. Will return the unary union of the geometries in the file."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=GeoFileGeometryParser(),
            show_default=False,
        ),
    ] = None,
    geom_filter_geocode: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the"
                " [bold dark_orange]string to geocode[/bold dark_orange] format - it will be"
                " geocoded to the geometry using Nominatim API (GeoPy library)."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=GeocodeGeometryParser(),
            show_default=False,
        ),
    ] = None,
    geom_filter_geojson: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the [bold dark_orange]GeoJSON[/bold dark_orange]"
                " format."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=GeoJsonGeometryParser(),
            show_default=False,
        ),
    ] = None,
    geom_filter_index_geohash: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the"
                " [bold dark_orange]Geohash index[/bold dark_orange]"
                " format. Separate multiple values with a comma."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=GeohashGeometryParser(),
            show_default=False,
        ),
    ] = None,
    geom_filter_index_h3: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the [bold dark_orange]H3 index[/bold dark_orange]"
                " format. Separate multiple values with a comma."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=H3GeometryParser(),
            show_default=False,
        ),
    ] = None,
    geom_filter_index_s2: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the [bold dark_orange]S2 index[/bold dark_orange]"
                " format. Separate multiple values with a comma."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=S2GeometryParser(),
            show_default=False,
        ),
    ] = None,
    geom_filter_wkt: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Geometry to use as a filter in the [bold dark_orange]WKT[/bold dark_orange]"
                " format."
                " Cannot be used together with other"
                " [bold bright_cyan]geom-filter-...[/bold bright_cyan] parameters."
            ),
            click_type=WktGeometryParser(),
            show_default=False,
        ),
    ] = None,
    pyarrow_filters: Annotated[
        Optional[list[str]],
        typer.Option(
            "--filter",
            "--pyarrow-filter",
            help=(
                "Filters to apply on a pyarrow dataset."
                " Required format: <column(s)> <operator> <value>."
                " Nested column names should be passed separated by a dot."
                " Can pass multiple filters."
            ),
            click_type=PyArrowExpressionParser(),
            show_default=False,
        ),
    ] = None,
    result_file_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help=(
                "Path where to save final geoparquet file. If not provided, it will be generated"
                " automatically based on the input pbf file name."
            ),
            show_default=False,
        ),
    ] = None,
    ignore_cache: Annotated[
        bool,
        typer.Option(
            "--ignore-cache/",
            "--no-cache/",
            help="Whether to ignore previously precalculated geoparquet files or not.",
            show_default=False,
        ),
    ] = False,
    working_directory: Annotated[
        Path,
        typer.Option(
            "--working-directory",
            "--work-dir",
            help=(
                "Directory where to save the parsed parquet and geoparquet files."
                " Will be created if doesn't exist."
            ),
        ),
    ] = "files",  # type: ignore
    silent_mode: Annotated[
        bool,
        typer.Option(
            "--silent/",
            help="Whether to disable progress reporting.",
            show_default=False,
        ),
    ] = False,
    transient_mode: Annotated[
        bool,
        typer.Option(
            "--transient/",
            help="Whether to make more transient (concise) progress reporting.",
            show_default=False,
        ),
    ] = False,
    # allow_uncovered_geometry: Annotated[
    #     bool,
    #     typer.Option(
    #         "--allow-uncovered-geometry/",
    #         help=(
    #             "Suppresses an error if some geometry parts aren't covered by any OSM extract."
    #             " Works only when PbfFileReader is asked to download OSM extracts automatically."
    #         ),
    #         show_default=False,
    #     ),
    # ] = False,
    show_release_versions: Annotated[
        Optional[bool],
        typer.Option(
            "--show-release-versions",
            help="Show available OvertureMaps release versions and exit.",
            callback=_display_release_versions_callback,
            is_eager=False,
        ),
    ] = None,
    show_theme_type_pairs: Annotated[
        Optional[bool],
        typer.Option(
            "--show-theme-type-pairs",
            help="Show available OvertureMaps theme type pairs for the release and exit.",
            callback=_display_theme_type_pairs_callback,
            is_eager=False,
        ),
    ] = None,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show the application's version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    OvertureMaestro CLI.

    Wraps public functions and prints final path to the saved geoparquet file at the end.
    """
    number_of_geometries_provided = sum(
        geom is not None
        for geom in (
            geom_filter_bbox,
            geom_filter_file,
            geom_filter_geocode,
            geom_filter_geojson,
            geom_filter_index_geohash,
            geom_filter_index_h3,
            geom_filter_index_s2,
            geom_filter_wkt,
        )
    )
    if theme_value is None or type_value is None or number_of_geometries_provided == 0:
        from click.exceptions import UsageError

        raise UsageError(
            message=(
                "OvertureMaestro requires theme, type and a geometry filter"
                " (one of --geom-filter-bbox --geom-filter-file, --geom-filter-geocode,"
                " --geom-filter-geojson, --geom-filter-index-geohash,"
                " --geom-filter-index-h3, --geom-filter-index-s2, --geom-filter-wkt)"
                " to download the data."
            ),
        )

    if number_of_geometries_provided > 1:
        raise typer.BadParameter("Provided more than one geometry for filtering")

    geometry_filter_value = (
        geom_filter_bbox
        or geom_filter_file
        or geom_filter_geocode
        or geom_filter_geojson
        or geom_filter_index_geohash
        or geom_filter_index_h3
        or geom_filter_index_s2
        or geom_filter_wkt
    )

    logging.disable(logging.CRITICAL)

    if transient_mode and silent_mode:
        raise typer.BadParameter("Cannot pass both silent and transient mode at once.")

    verbosity_mode: VERBOSITY_MODE = "verbose"

    if transient_mode:
        verbosity_mode = "transient"
    elif silent_mode:
        verbosity_mode = "silent"

    from overturemaestro import convert_geometry_to_parquet, get_available_theme_type_pairs

    if (theme_value, type_value) not in get_available_theme_type_pairs(
        release=release_version  # type: ignore[arg-type]
    ):
        raise typer.BadParameter(
            f"Dataset of theme = {theme_value} and type = {type_value} doesn't exist."
        )

    pyarrow_filter = None

    if pyarrow_filters:
        import functools
        import operator

        pyarrow_filter = functools.reduce(operator.and_, pyarrow_filters)

    geoparquet_path = convert_geometry_to_parquet(
        theme=theme_value,
        type=type_value,
        geometry_filter=geometry_filter_value,
        release=release_version,
        ignore_cache=ignore_cache,
        working_directory=working_directory,
        result_file_path=result_file_path,
        verbosity_mode=verbosity_mode,
        pyarrow_filter=pyarrow_filter,
    )

    typer.secho(geoparquet_path, fg="green")
