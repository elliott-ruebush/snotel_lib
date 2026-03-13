import datetime as dt

import geopandas as gpd
import pandas as pd
import polars as pl
import pytest
from shapely.geometry import Point

from snotel_lib.calculation import (
    accumulate_precip_by_water_year,
    format_rows,
    get_min_and_max_rows,
    get_top_bot,
)


def test_format_rows():
    df = pl.DataFrame(
        {
            "station_id": ["A", "B"],
            "station_name": ["Station A", "Station B"],
            "state": ["CO", "WA"],
            "elevation_m": [1000, 2000],
            "snow_depth_m": [1.2345, 2.3456],
            "year": [2021, 2022],
        }
    )

    rows = format_rows(df, "snow_depth_m", round_digits=2, extra_cols=["year"])

    assert len(rows) == 2
    assert rows[0]["value"] == 1.23
    assert rows[0]["year"] == 2021
    assert rows[0]["name"] == "Station A"


def test_get_top_bot():
    df = pl.DataFrame(
        {
            "station_id": ["A", "B", "C", "D"],
            "station_name": ["A", "B", "C", "D"],
            "state": ["CO", "CO", "CO", "CO"],
            "elevation_m": [1, 2, 3, 4],
            "val": [10.0, 5.0, 1.0, 20.0],
            "abs_val": [10.0, 5.0, 1.0, 20.0],
        }
    )

    res = get_top_bot(df, "val", top_n=2, bot_n=2, sort_by="abs_val")
    assert res["total_count"] == 4
    # Top should be D (20), A (10)
    assert res["top"][0]["station_id"] == "D"
    assert res["top"][1]["station_id"] == "A"
    # Bot should be C (1), B (5)
    assert res["bottom"][-1]["station_id"] == "C"


def test_format_rows_nulls():
    df = pl.DataFrame(
        {
            "station_id": ["A", "B"],
            "station_name": ["Station A", "Station B"],
            "state": ["CO", "WA"],
            "elevation_m": [1000, 2000],
            "snow_depth_m": [1.2345, None],
            "year": [None, 2022],
        }
    )
    rows = format_rows(df, "snow_depth_m", round_digits=2, extra_cols=["year"])

    assert rows[0]["value"] == 1.23
    assert rows[0]["year"] is None
    assert rows[1]["value"] is None
    assert rows[1]["year"] == 2022


def test_get_top_bot_edge_cases():
    # Tie values, less rows than top_n/bot_n, and nulls
    df = pl.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "val": [10.0, 10.0, None],  # 2 valid values
        }
    )

    res = get_top_bot(df, "val", top_n=5, bot_n=5)
    assert res["total_count"] == 2
    assert len(res["top"]) == 2
    assert len(res["bottom"]) == 2


def test_get_min_and_max_rows():
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    data = {
        "code": ["S1", "S2", "S3"],
        "name": ["Station 1", "Station 2", "Station 3"],
        "end_date": [today, yesterday, today - dt.timedelta(days=5)],
        "val": [10.0, 20.0, 5.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
    }
    gdf = gpd.GeoDataFrame(data).set_index("code")
    gdf["end_date"] = pd.to_datetime(gdf["end_date"]).dt.date

    result = get_min_and_max_rows(gdf, "val")  # ty: ignore[invalid-argument-type]

    # Only S1 and S2 are within last 2 days
    assert len(result) == 2
    assert "S2" in result.index  # Max
    assert "S1" in result.index  # Min
    assert result.loc["S2", "val"] == 20.0
    assert result.loc["S1", "val"] == 10.0


def test_accumulate_precip_by_water_year():
    # Test dates crossing water year boundary (Oct 1)
    # A single station's data
    df = pl.DataFrame(
        {
            "datetime": [dt.date(2023, 9, 30), dt.date(2023, 10, 1), dt.date(2023, 10, 2)],
            "precip_m": [0.5, 0.2, 0.1],
        }
    )

    res = accumulate_precip_by_water_year(df)

    # Oct 1 should reset the accumulation
    # Sept 30 (WY 2023): 0.5
    # Oct 1 (WY 2024): 0.2
    # Oct 2 (WY 2024): 0.2 + 0.1 = 0.3
    assert res["precip_m"].to_list() == pytest.approx([0.5, 0.2, 0.3])


def test_accumulate_precip_by_water_year_multi_station():
    # Multiple stations, crossing water year boundary
    df = pl.DataFrame(
        {
            "station_id": ["A", "A", "A", "B", "B"],
            "datetime": [
                dt.date(2023, 9, 30),
                dt.date(2023, 10, 1),
                dt.date(2023, 10, 2),
                dt.date(2023, 9, 30),
                dt.date(2023, 10, 1),
            ],
            "precip_m": [0.5, 0.2, 0.1, 0.8, 0.1],
        }
    )

    res = accumulate_precip_by_water_year(df, is_all_stations=True)

    # Accumulation should be per station AND per water year
    # A: [0.5, 0.2, 0.3]
    # B: [0.8, 0.1]
    expected = [0.5, 0.2, 0.3, 0.8, 0.1]
    assert res["precip_m"].to_list() == pytest.approx(expected)
