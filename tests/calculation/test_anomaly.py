import datetime

import polars as pl

from snotel_lib.calculation import compute_live_z_score


def test_compute_live_z_score():
    # 6 years of data on Jan 1st.
    df = pl.DataFrame(
        {
            "station_id": ["A"] * 6,
            "datetime": [
                datetime.date(2020, 1, 1),
                datetime.date(2021, 1, 1),
                datetime.date(2022, 1, 1),
                datetime.date(2023, 1, 1),
                datetime.date(2024, 1, 1),
                datetime.date(2025, 1, 1),  # Latest observation
            ],
            "swe_m": [1.0, 1.2, 0.8, 1.1, 0.9, 1.5],  # mean=1.0, std=0.158
            "snow_depth_m": [1.0] * 6,
        }
    )

    res = compute_live_z_score(df)
    assert len(res) == 1

    z_score = res.select("live_z_score").item()
    # Math:
    # 2020: 1.0, 2021: 1.2, 2022: 0.8, 2023: 1.1, 2024: 0.9 -> Mean 1.0, std_dev = sqrt(0.1/4) = 0.158113883
    # 2025: 1.5 -> (1.5 - 1.0) / 0.158113883 = 3.162

    assert abs(z_score - 3.162) < 0.01


def test_compute_live_z_score_zero_variance():
    # 6 years of data where SWE is exactly the same, resulting in 0 variance
    df = pl.DataFrame(
        {
            "station_id": ["B"] * 6,
            "datetime": [
                datetime.date(2020, 1, 1),
                datetime.date(2021, 1, 1),
                datetime.date(2022, 1, 1),
                datetime.date(2023, 1, 1),
                datetime.date(2024, 1, 1),
                datetime.date(2025, 1, 1),
            ],
            "swe_m": [1.0, 1.0, 1.0, 1.0, 1.0, 1.5],  # Hist mean=1.0, std=0.0
            "snow_depth_m": [1.0] * 6,
        }
    )

    res = compute_live_z_score(df)
    # Because standard deviation is zero, the live_z_score would be inf/NaN
    # Our function is designed to filter out non-finite z-scores.
    assert len(res) == 0


def test_compute_live_z_score_staleness_filter():
    """
    Regression test for the Skookum Creek bug: a station with stale cache data
    (last reading months before the global max date) was generating spurious high
    Z-scores because the "current" reading was from a snow-free period, compared
    against historically snow-free historical days.

    We use two stations here:
    - Station A: recent, up to the global max date 2026-02-26
    - Station B: stale, last reading is 2025-10-01 (~150 days behind)

    Station B should be excluded from Z-score results.
    """
    # Station A has 6 years of Feb-26 data and a current reading on 2026-02-26
    station_a = pl.DataFrame(
        {
            "station_id": ["A"] * 6,
            "datetime": [
                datetime.date(2020, 2, 26),
                datetime.date(2021, 2, 26),
                datetime.date(2022, 2, 26),
                datetime.date(2023, 2, 26),
                datetime.date(2024, 2, 26),
                datetime.date(2026, 2, 26),  # Current
            ],
            "swe_m": [0.4, 0.5, 0.45, 0.6, 0.35, 0.47],
            "snow_depth_m": [1.0] * 6,
        }
    )

    # Station B's last reading is from 2025-10-01 (stale — ~150 days before global max)
    station_b = pl.DataFrame(
        {
            "station_id": ["B"] * 6,
            "datetime": [
                datetime.date(2020, 10, 1),
                datetime.date(2021, 10, 1),
                datetime.date(2022, 10, 1),
                datetime.date(2023, 10, 1),
                datetime.date(2024, 10, 1),
                datetime.date(2025, 10, 1),  # "Current" but stale vs global 2026-02-26
            ],
            "swe_m": [0.0, 0.01, 0.0, 0.02, 0.0, 0.01],  # Near-zero early-season
            "snow_depth_m": [0.0] * 6,
        }
    )

    df = pl.concat([station_a, station_b])
    res = compute_live_z_score(df, max_staleness_days=14)

    # Station B should be excluded; only Station A should appear
    station_ids = res.select("station_id").to_series().to_list()
    assert "B" not in station_ids, "Stale station B should have been excluded"
    assert "A" in station_ids, "Recent station A should be included"
