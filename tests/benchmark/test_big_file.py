"""Test big file pipeline."""

from parametrization import Parametrization as P

from overturemaestro import convert_geometry_to_parquet, geocode_to_geometry
from overturemaestro.advanced_functions import convert_geometry_to_wide_form_parquet_for_all_types


@P.parameters("geocode_filter", "theme_type_pair")  # type: ignore
@P.case("Spain", "spain", ("buildings", "building"))  # type: ignore
def test_big_file(
    geocode_filter: str, theme_type_pair: tuple[str, str], test_release_version: str
) -> None:
    """Test if big file is working in a low memory environment."""
    geometry_filter = geocode_to_geometry(geocode_filter)
    output_file = convert_geometry_to_parquet(
        geometry_filter=geometry_filter,
        theme=theme_type_pair[0],
        type=theme_type_pair[1],
        release=test_release_version,
        verbosity_mode="verbose",
        ignore_cache=True,
    )
    print(output_file)


@P.parameters("geocode_filter")  # type: ignore
@P.case("Portugal", "portugal")  # type: ignore
def test_big_file_wide_mode(geocode_filter: str, test_release_version: str) -> None:
    """Test if big file is working in a low memory environment."""
    geometry_filter = geocode_to_geometry(geocode_filter)
    output_file = convert_geometry_to_wide_form_parquet_for_all_types(
        geometry_filter=geometry_filter,
        release=test_release_version,
        verbosity_mode="verbose",
        ignore_cache=True,
    )
    print(output_file)
