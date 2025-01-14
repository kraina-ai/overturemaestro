"""Tests for CLI."""

import uuid
from pathlib import Path
from typing import Optional

import pytest
from parametrization import Parametrization as P
from typer.testing import CliRunner

from overturemaestro import __app_name__, __version__, cli
from overturemaestro.conftest import TEST_RELEASE_VERSIONS_LIST
from overturemaestro.data_downloader import _generate_geometry_hash
from overturemaestro.release_index import get_available_theme_type_pairs, get_newest_release_version
from tests.conftest import (
    TEST_RELEASE_VERSION,
    geometry_bbox_str,
    geometry_boundary_file_path,
    geometry_box,
    geometry_geojson,
    geometry_wkt,
)

runner = CliRunner()

# TODO: add pyarrow nested test


def random_str() -> str:
    """Return random string."""
    return str(uuid.uuid4())


def test_version() -> None:
    """Test if version is properly returned."""
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert f"{__app_name__} {__version__}\n" in result.stdout


@P.parameters("args")  # type: ignore
@P.case(
    "No args",
    [],
)  # type: ignore
@P.case(
    "Theme only",
    ["base"],
)  # type: ignore
@P.case(
    "Theme type only",
    ["base", "water"],
)  # type: ignore
def test_theme_type_and_geometry_filter_is_required(args) -> None:
    """Test if cannot run without pbf file and without geometry filter."""
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 2
    assert "OvertureMaestro requires theme, type and a geometry filter" in result.stdout


def test_basic_run(test_release_version: str) -> None:
    """Test if runs properly without options."""
    theme_value = "buildings"
    type_value = "building"

    result = runner.invoke(
        cli.app,
        [
            "--release",
            test_release_version,
            "--geom-filter-bbox",
            geometry_bbox_str(),
            theme_value,
            type_value,
        ],
    )

    geometry_hash = _generate_geometry_hash(geometry_box())

    assert result.exit_code == 0
    assert (
        str(
            Path(f"files/{test_release_version}/theme={theme_value}/type={type_value}")
            / f"{geometry_hash}_nofilter.parquet"
        )
        in result.stdout
    )


def test_silent_mode(test_release_version: str) -> None:
    """Test if runs properly without reporting status."""
    theme_value = "buildings"
    type_value = "building"
    result = runner.invoke(
        cli.app,
        [
            "--release",
            test_release_version,
            "--geom-filter-bbox",
            geometry_bbox_str(),
            theme_value,
            type_value,
            "--silent",
            "--ignore-cache",
        ],
    )

    geometry_hash = _generate_geometry_hash(geometry_box())

    print(result.stdout)

    assert result.exit_code == 0
    assert (
        str(
            Path(f"files/{test_release_version}/theme={theme_value}/type={type_value}")
            / f"{geometry_hash}_nofilter.parquet"
        )
        == result.stdout.strip()
    )


def test_transient_mode(test_release_version: str) -> None:
    """Test if runs properly without reporting status."""
    theme_value = "buildings"
    type_value = "building"
    result = runner.invoke(
        cli.app,
        [
            "--release",
            test_release_version,
            "--geom-filter-bbox",
            geometry_bbox_str(),
            theme_value,
            type_value,
            "--transient",
            "--ignore-cache",
        ],
    )
    output_lines = result.stdout.strip().split("\n")

    geometry_hash = _generate_geometry_hash(geometry_box())

    assert result.exit_code == 0
    assert len(result.stdout.strip().split("\n")) == 2
    assert "Finished operation in" in output_lines[0]
    assert (
        str(
            Path(f"files/{test_release_version}/theme={theme_value}/type={type_value}")
            / f"{geometry_hash}_nofilter.parquet"
        )
        == output_lines[1]
    )


@P.parameters("args", "expected_result")  # type: ignore
@P.case(
    "Output",
    [
        "--output",
        "files/monaco_output_long.parquet",
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
    ],
    "files/monaco_output_long.parquet",
)  # type: ignore
@P.case(
    "Output short",
    [
        "-o",
        "files/monaco_output_short.parquet",
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
    ],
    "files/monaco_output_short.parquet",
)  # type: ignore
@P.case(
    "Base without release",
    [
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSIONS_LIST[-1]}/"
    f"theme=buildings/type=building/{_generate_geometry_hash(geometry_box())}_nofilter.parquet",
)  # type: ignore
@P.case(
    "Working directory",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--working-directory",
        "files/workdir",
    ],
    f"files/workdir/{TEST_RELEASE_VERSION}/"
    f"theme=buildings/type=building/{_generate_geometry_hash(geometry_box())}_nofilter.parquet",
)  # type: ignore
@P.case(
    "Ignore cache",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--output",
        "files/monaco_output.parquet",
        "--ignore-cache",
    ],
    "files/monaco_output.parquet",
)  # type: ignore
@P.case(
    "Ignore cache short",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--output",
        "files/monaco_output.parquet",
        "--no-cache",
    ],
    "files/monaco_output.parquet",
)  # type: ignore
@P.case(
    "Silent",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--silent",
        "--output",
        "files/monaco_output.parquet",
    ],
    "files/monaco_output.parquet",
)  # type: ignore
@P.case(
    "Transient",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--transient",
        "--output",
        "files/monaco_output.parquet",
    ],
    "files/monaco_output.parquet",
)  # type: ignore
@P.case(
    "Output with working directory",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--working-directory",
        "files/workdir",
        "-o",
        "files/monaco_output.parquet",
    ],
    "files/monaco_output.parquet",
)  # type: ignore
@P.case(
    "PyArrow filtering",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--filter",
        "subtype = residential",
    ],
    f"files/{TEST_RELEASE_VERSION}/theme=buildings/type=building/"
    f"{_generate_geometry_hash(geometry_box())}_b22759b51edd150209b17a03319daa6796f574478c5f4f79ae2efa62e65e3ba1.parquet",
)  # type: ignore
@P.case(
    "Geometry WKT filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-wkt",
        geometry_wkt(),
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/09c3fc0471538594b784be7c52782837c7a26753c2b26097b780581fa0a6bfc6_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry GeoJSON filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-geojson",
        geometry_geojson(),
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/82c0fdfa2d5654818a03540644834d70c353e3f82f9d8f201c37420aeb35118e_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry file filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-file",
        geometry_boundary_file_path(),
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/6a869bcfa1a49ade8b76569e48e4142bce29098815bf37e57155a18204f2bbbc_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry geocode filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-geocode",
        "Monaco-Ville, Monaco",
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/e7f0b78a0fdc16c4db31c9767fa4e639eadaa8e83a9b90e07b521f4925cdf4b3_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry Geohash filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-index-geohash",
        "spv2bc",
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/c08889e81575260e7ea2bc9764ddaa7c5e1141270a890b022799689d39dfe4d5_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry Geohash filter multiple",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-index-geohash",
        "spv2bc,spv2bfr",
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/1bd33e0afc3cd0efcb4740185b8a05ecaf1bac916d571403768939b82844b43e_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry H3 filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-index-h3",
        "8a3969a40ac7fff",
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/a2f8d5114760394646aa999a1204adaa48ad686b3fcadb0b25fd02322c16dff4_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry H3 filter multiple",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-index-h3",
        "8a3969a40ac7fff,893969a4037ffff",
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/e50e6489d4faba664a3c7f9729b7a3bedafb7a396ae826521f32b556d0b554f1_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry S2 filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-index-s2",
        "12cdc28bc",
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/5c3d61eb108819e543a1a59fe6c67658f817c0453d728b5aa007f227453d5bf6_nofilter.parquet",
)  # type: ignore
@P.case(
    "Geometry S2 filter multiple",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-index-s2",
        "12cdc28bc,12cdc28f",
        "buildings",
        "building",
    ],
    f"files/{TEST_RELEASE_VERSION}/"
    "theme=buildings/type=building/cda5d65e169e3ff04970e66e80b021937a5ae141ed72428cc0d9a3764bf076db_nofilter.parquet",
)  # type: ignore
def test_proper_args(args: list[str], expected_result: str) -> None:
    """Test if runs properly with options."""
    result = runner.invoke(cli.app, args)
    print(result.stdout)
    if result.exception:
        print(result.exception)

    assert result.exit_code == 0
    assert str(Path(expected_result)) in result.stdout


@P.parameters("args")  # type: ignore
@P.case(
    "Geometry WKT filter with GeoJSON",
    ["buildings", "building", "--geom-filter-wkt", geometry_geojson()],
)  # type: ignore
@P.case(
    "Geometry GeoJSON filter with WKT",
    ["buildings", "building", "--geom-filter-geojson", geometry_wkt()],
)  # type: ignore
@P.case(
    "Geometry Geohash filter with random string",
    ["buildings", "building", "--geom-filter-index-geohash", random_str()],
)  # type: ignore
@P.case(
    "Geometry H3 filter with Geohash", ["buildings", "building", "--geom-filter-index-h3", "spv2bc"]
)  # type: ignore
@P.case(
    "Geometry H3 filter with S2", ["buildings", "building", "--geom-filter-index-h3", "12cdc28bc"]
)  # type: ignore
@P.case(
    "Geometry H3 filter with random string",
    ["buildings", "building", "--geom-filter-index-h3", random_str()],
)  # type: ignore
@P.case(
    "Geometry S2 filter with random string",
    ["buildings", "building", "--geom-filter-index-s2", random_str()],
)  # type: ignore
@P.case(
    "Geometry BBOX filter with wrong values",
    ["buildings", "building", "--geom-filter-bbox", random_str()],
)  # type: ignore
@P.case(
    "Geometry BBOX filter with spaces",
    ["buildings", "building", "--geom-filter-bbox", "10,", "-5,", "6,", "17"],
)  # type: ignore
@P.case(
    "Geometry two filters",
    [
        "buildings",
        "building",
        "--geom-filter-geojson",
        geometry_geojson(),
        "--geom-filter-wkt",
        geometry_wkt(),
    ],
)  # type: ignore
@P.case(
    "Geometry nonexistent file filter",
    ["buildings", "building", "--geom-filter-file", "nonexistent_geojson_file.geojson"],
)  # type: ignore
@P.case(
    "Wrong theme & type",
    ["wrong_theme", "wrong_type", "--geom-filter-bbox", geometry_bbox_str()],
)  # type: ignore
@P.case(
    "Wrong theme",
    ["wrong_theme", "building", "--geom-filter-bbox", geometry_bbox_str()],
)  # type: ignore
@P.case(
    "Wrong type",
    ["buildings", "wrong_type", "--geom-filter-bbox", geometry_bbox_str()],
)  # type: ignore
@P.case(
    "Wrong release",
    [
        "buildings",
        "building",
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "--release",
        random_str(),
    ],
)  # type: ignore
@P.case(
    "PyArrow filter with random strings",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--filter",
        f"{random_str()} = {random_str()}",
    ],
)  # type: ignore
@P.case(
    "PyArrow filter with wrong columns filter",
    [
        "--release",
        TEST_RELEASE_VERSION,
        "--geom-filter-bbox",
        geometry_bbox_str(),
        "buildings",
        "building",
        "--filter",
        "confidence > 0.9",
    ],
)  # type: ignore
def test_wrong_args(args: list[str], capsys: pytest.CaptureFixture) -> None:
    """Test if doesn't run properly with options."""
    # Fix for the I/O error from the Click repository
    # https://github.com/pallets/click/issues/824#issuecomment-1583293065
    with capsys.disabled():
        result = runner.invoke(cli.app, args)
        assert result.exit_code != 0


def test_displaying_release_versions(
    capsys: pytest.CaptureFixture,
) -> None:
    """Test if displaying OSM extracts works."""
    with capsys.disabled():
        result = runner.invoke(cli.app, ["--show-release-versions"])
        output = result.stdout

        assert result.exit_code == 0
        assert len(output) > 0
        assert "Release Version" in output
        assert TEST_RELEASE_VERSION in output


@pytest.mark.parametrize(
    "release_version",
    [TEST_RELEASE_VERSION, None],
)  # type: ignore
def test_displaying_theme_type_pairs(
    release_version: Optional[str],
    capsys: pytest.CaptureFixture,
) -> None:
    """Test if displaying OSM extracts works."""
    with capsys.disabled():
        release_args = ["--release", release_version] if release_version else []
        result = runner.invoke(cli.app, ["--show-theme-type-pairs", *release_args])
        output = result.stdout

        assert result.exit_code == 0
        assert len(output) > 0
        assert (release_version or get_newest_release_version()) in output
        assert "Theme" in output
        assert "Type" in output

        if release_version:
            for theme_value, type_value in get_available_theme_type_pairs(release_version):
                assert theme_value in output
                assert type_value in output
