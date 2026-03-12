import polars as pl

from snotel_lib.calculation import compute_consistency_metrics


def test_compute_consistency_metrics():
    df = pl.DataFrame(
        {
            "station_id": ["A"] * 6,
            "water_year": [2000, 2001, 2002, 2003, 2004, 2005],
            "snow_depth_m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    res = compute_consistency_metrics(df, min_observations_per_year=1)

    assert res.height == 1
    assert res.select("wy_count").item() == 6
    assert res.select("all_time_max").item() == 6.0
    assert res.select("all_time_max_year").item() == 2005


def test_compute_consistency_metrics_edge_cases():
    # Test skipping years with < 330 observations if min_observations_per_year=330
    df = pl.DataFrame(
        {
            "station_id": ["B"] * 400 + ["B"] * 300 + ["B"] * 400 * 3,
            "water_year": [2000] * 400 + [2001] * 300 + [2002] * 400 + [2003] * 400 + [2004] * 400,
            "snow_depth_m": [1.0] * 400 + [2.0] * 300 + [3.0] * 400 + [4.0] * 400 + [5.0] * 400,
        }
    )

    # 2001 has 300 obs, so it should be filtered out
    # Only 4 valid years (2000, 2002, 2003, 2004)
    # The requirement is >= 5 valid years, so res should be empty
    res = compute_consistency_metrics(df, min_observations_per_year=330)
    assert res.height == 0

    # Add another valid year to make it 5 valid years
    df2 = pl.concat(
        [
            df,
            pl.DataFrame(
                {
                    "station_id": ["B"] * 400,
                    "water_year": [2005] * 400,
                    "snow_depth_m": [6.0] * 400,
                }
            ),
        ]
    )
    res2 = compute_consistency_metrics(df2, min_observations_per_year=330)
    assert res2.height == 1
    assert res2.select("wy_count").item() == 5  # 2001 was skipped
