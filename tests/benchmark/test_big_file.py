"""Test big file pipeline."""

from parametrization import Parametrization as P

from overturemaestro import convert_geometry_to_parquet, geocode_to_geometry


@P.parameters("geocode_filter", "theme_type_pair")  # type: ignore
@P.case("Spain", "spain", ("buildings", "building"))  # type: ignore
def test_big_file(
    geocode_filter: str, theme_type_pair: tuple[str, str], test_release_version: str
) -> None:
    """Test if big file is working in a low memory environment."""
    geometry_filter = geocode_to_geometry(geocode_filter)
    convert_geometry_to_parquet(
        geometry_filter=geometry_filter,
        theme=theme_type_pair[0],
        type=theme_type_pair[1],
        release=test_release_version,
        verbosity_mode="verbose",
        ignore_cache=True,
    )
