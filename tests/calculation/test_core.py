import polars as pl

from snotel_lib.calculation import format_rows, get_top_bot


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
