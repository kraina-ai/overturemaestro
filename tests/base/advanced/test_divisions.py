"""Tests for advanced divisions functions."""

from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely import Point, box

import overturemaestro.advanced_functions.divisions as divisions_module
from overturemaestro._constants import GEOMETRY_COLUMN, INDEX_COLUMN
from overturemaestro.advanced_functions import (
    convert_bounding_box_to_divisions_geodataframe,
    convert_geometry_to_divisions_geodataframe,
    convert_geometry_to_divisions_parquet,
)
from tests.conftest import bbox

HIERARCHY_TYPE = pa.list_(
    pa.list_(
        pa.struct(
            [("division_id", pa.string()), ("subtype", pa.string()), ("name", pa.string())]
        )
    )
)
NAMES_TYPE = pa.struct([("primary", pa.string()), ("common", pa.map_(pa.string(), pa.string()))])
BBOX_TYPE = pa.struct([(name, pa.float64()) for name in ("xmin", "xmax", "ymin", "ymax")])


def _save(path: Path, columns: dict[str, pa.Array], geometries: list[Any]) -> Path:
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=4326)
    table = pa.table(gdf.to_arrow(index=False))
    for name, values in columns.items():
        table = table.append_column(name, values)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _mock_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, areas: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace download with local files, returns list of download calls."""
    calls: list[dict[str, Any]] = []
    without_division = {area["id"] for area in areas if area["hierarchy"] is None}

    def download(**kwargs: Any) -> Path:
        calls.append(kwargs)
        if kwargs["type"] == "division_area":
            geometries = [area["geometry"] for area in areas]
            return _save(
                tmp_path / "areas.parquet",
                {
                    "id": pa.array([f"area_{area['id']}" for area in areas], pa.string()),
                    "division_id": pa.array([area["id"] for area in areas], pa.string()),
                    "subtype": pa.array(["locality"] * len(areas), pa.string()),
                    "names": pa.array(
                        [
                            {
                                "primary": area["name"],
                                "common": [("en", area["name_en"])] if area["name_en"] else [],
                            }
                            for area in areas
                        ],
                        NAMES_TYPE,
                    ),
                    "country": pa.array(["PL"] * len(areas), pa.string()),
                    "region": pa.array(["PL-02"] * len(areas), pa.string()),
                    "bbox": pa.array(
                        [
                            dict(zip(("xmin", "ymin", "xmax", "ymax"), geometry.bounds))
                            for geometry in geometries
                        ],
                        BBOX_TYPE,
                    ),
                },
                geometries,
            )
        found = [area for area in areas if area["id"] not in without_division]
        return _save(
            tmp_path / "divisions.parquet",
            {
                "id": pa.array([area["id"] for area in found], pa.string()),
                "hierarchies": pa.array([[area["hierarchy"]] for area in found], HIERARCHY_TYPE),
                "parent_division_id": pa.array(
                    [area["hierarchy"][-2]["division_id"] for area in found], pa.string()
                ),
                "wikidata": pa.array([None] * len(found), pa.string()),
                "population": pa.array([1] * len(found), pa.int32()),
            },
            [Point(0, 0)] * len(found),
        )

    monkeypatch.setattr(divisions_module, "convert_geometry_to_parquet", download)
    return calls


def _area(
    id: str,
    hierarchy: Optional[list[tuple[str, str, str]]],
    geometry: Any,
    name_en: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "name": f"Name {id}",
        "name_en": name_en,
        "geometry": geometry,
        "hierarchy": hierarchy
        and [{"division_id": i, "subtype": s, "name": n} for i, s, n in hierarchy],
    }


COUNTRY, REGION = ("pl", "country", "Polska"), ("ds", "region", "Dolnośląskie")
CITY, DISTRICT = ("wro", "locality", "Wrocław"), ("sm", "locality", "Stare Miasto")


def test_hierarchy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test if hierarchy is flattened and order of areas is kept."""
    calls = _mock_download(
        monkeypatch,
        tmp_path,
        [
            _area("sm", [COUNTRY, REGION, CITY, DISTRICT], box(17.0, 51.1, 17.1, 51.2), "Old Town"),
            _area("wro", [COUNTRY, REGION, CITY], box(16.8, 51.0, 17.2, 51.2)),
            _area("orphan", None, box(17.1, 51.1, 17.2, 51.2)),
        ],
    )

    gdf = convert_geometry_to_divisions_geodataframe(
        "locality", Point(17.05, 51.15), release="r", working_directory=tmp_path
    )

    assert gdf.index.name == INDEX_COLUMN
    assert gdf.index.tolist() == ["sm", "wro", "orphan"]
    assert gdf.crs == "EPSG:4326"
    assert gdf["division_area_id"].tolist() == ["area_sm", "area_wro", "area_orphan"]
    assert gdf["name"].tolist() == ["Name sm", "Name wro", "Name orphan"]
    assert gdf["geometry"].iloc[0].equals(box(17.0, 51.1, 17.1, 51.2))

    nested, city, orphan = (gdf.loc[id].to_dict() for id in gdf.index)
    assert nested["name_en"] == "Old Town"
    assert nested["parent_division_id"] == "wro"
    assert (nested["country_name"], nested["region_id"]) == ("Polska", "ds")
    # outermost locality
    assert (nested["locality_id"], nested["locality_name"]) == ("wro", "Wrocław")
    assert [h["division_id"] for h in nested["hierarchies"][0]] == ["pl", "ds", "wro", "sm"]
    assert city["locality_id"] == "wro"
    assert gdf["county_id"].isna().all()
    assert gdf["name_en"].isna().tolist() == [False, True, True]
    assert gdf["country_id"].isna().tolist() == [False, False, True]
    assert gdf["country"].tolist() == ["PL"] * 3

    # points are loaded using bounds of areas
    assert [call["type"] for call in calls] == ["division_area", "division"]
    assert calls[1]["geometry_filter"].bounds == pytest.approx((16.7, 50.9, 17.3, 51.3))
    assert ("subtype", "=", "locality") in calls[1]["pyarrow_filter"]


def test_empty_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test if lack of divisions results in an empty file with all columns."""
    calls = _mock_download(monkeypatch, tmp_path, [])

    path = convert_geometry_to_divisions_parquet(
        "locality", box(0, 0, 1, 1), release="r", result_file_path=tmp_path / "result.parquet"
    )

    assert path == tmp_path / "result.parquet"
    assert pq.read_metadata(path).num_rows == 0
    assert {INDEX_COLUMN, "region_name", "hierarchies", GEOMETRY_COLUMN} <= set(
        pq.read_schema(path).names
    )
    assert calls[1]["geometry_filter"].bounds == pytest.approx((-0.1, -0.1, 1.1, 1.1))


def test_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test if existing result isn't generated again."""
    calls = _mock_download(monkeypatch, tmp_path, [_area("wro", [COUNTRY, CITY], box(0, 0, 1, 1))])

    for ignore_cache, expected_types in (
        (False, ["division_area", "division"]),
        (False, ["division_area"]),
        (True, ["division_area", "division"]),
    ):
        calls.clear()
        convert_geometry_to_divisions_parquet(
            "locality",
            Point(0.5, 0.5),
            release="r",
            working_directory=tmp_path,
            ignore_cache=ignore_cache,
        )
        assert [call["type"] for call in calls] == expected_types


def test_unknown_subtype() -> None:
    """Test if unknown subtype is reported."""
    with pytest.raises(ValueError, match="Unknown division subtype: city"):
        convert_geometry_to_divisions_parquet("city", Point(0, 0))  # type: ignore[arg-type]


def test_real_data(test_release_version: str, tmp_path: Path) -> None:
    """Test if real data is properly joined."""
    gdf = convert_bounding_box_to_divisions_geodataframe(
        "country", bbox(), release=test_release_version, working_directory=tmp_path
    )

    assert "MC" in gdf["country"].tolist()
    assert (gdf["country_id"] == gdf.index).all()
    assert gdf["country_name"].notna().all()
