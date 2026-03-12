from datetime import date

import polars as pl

from snotel_lib.calculation import compute_diff_metrics


def test_compute_diff_metrics():
    df = pl.DataFrame(
        {
            "station_id": ["A", "A", "A", "A", "A", "A", "A", "A"],
            "datetime": [date(2023, 1, i) for i in range(1, 9)],
            "snow_depth_m": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5],
            "swe_m": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3],
        }
    )

    cutoff = date(2023, 1, 7)
    res = compute_diff_metrics(df.with_columns(pl.col("datetime").cast(pl.Date)), cutoff)

    assert res.height == 1

    # 24h prior is day 7
    diff24_snow = res.select("snow_depth_24h_diff").item()
    assert abs(diff24_snow - 0.5) < 0.001

    diff24_swe = res.select("swe_24h_diff").item()
    assert abs(diff24_swe - 0.2) < 0.001


def test_compute_diff_metrics_missing_and_null():
    df = pl.DataFrame(
        {
            "station_id": ["B", "B", "B"],
            "datetime": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "snow_depth_m": [1.0, None, 2.0],
            "swe_m": [0.1, 0.2, None],
        }
    )

    # 24h diff for a None row or referencing a None row yields None in Polars.
    cutoff = date(2023, 1, 1)
    res = compute_diff_metrics(df.with_columns(pl.col("datetime").cast(pl.Date)), cutoff)

    assert res.height == 1

    # snow_depth_m is 2.0. Prev row (01-02) is None, so 24h_diff should be None.
    assert res.select("snow_depth_24h_diff").item() is None
    # swe_m is None. Prev row is 0.2, 24h_diff is None.
    assert res.select("swe_24h_diff").item() is None

    # Check when the last row doesn't make the cutoff
    cutoff_future = date(2023, 2, 1)
    res_future = compute_diff_metrics(df.with_columns(pl.col("datetime").cast(pl.Date)), cutoff_future)
    assert res_future.height == 0
