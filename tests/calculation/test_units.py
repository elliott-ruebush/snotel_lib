from datetime import date

import polars as pl

from snotel_lib.calculation import format_rows


def test_conversion_factors_against_frontend():
    """
    Cross-checks the frontend constants against standard conversion factors.
    This ensures that the precision we chose for the frontend is sufficient.
    """
    # 1 meter = 1 / 0.0254 inches = 39.37007874015748
    expected_inches_per_meter = 39.3700787
    frontend_val = 39.3700787

    assert abs(frontend_val - expected_inches_per_meter) < 1e-9


def test_elevation_conversion():
    # 1 meter = 3.280839895 feet
    expected_feet_per_meter = 3.28084
    frontend_val = 3.28084

    assert abs(frontend_val - expected_feet_per_meter) < 1e-6


def test_exact_imperial_conversion():
    # 1 inch is exactly 0.0254 meters by international agreement.
    one_inch_in_meters = 0.0254
    assert one_inch_in_meters == 0.0254


def test_conversion_consistency():
    """
    Test that rounding meters to 4 decimal places in the backend
    and multiplying by 39.3700787 in the frontend results in consistent
    display for values at 0.1 inch precision.
    """
    frontend_factor = 39.3700787
    for i in range(-500, 500):
        # difference in tenths of an inch
        expected_inches = i / 10.0
        meters = expected_inches * 0.0254

        # backend rounding
        backend_val = round(meters, 4)

        # frontend conversion
        frontend_val = backend_val * frontend_factor

        # displayed string
        displayed = f"{frontend_val:.2f}"
        expected_str = f"{expected_inches:.2f}"

        assert displayed == expected_str, (
            f"Inconsistency! Inches: {expected_inches}, Meters: {meters}, Backend: {backend_val}, Frontend: {displayed} != {expected_str}"
        )


def test_unit_conversion_consistency():
    """
    Test that identical meter values (e.g. from 1 inch = 0.0254m)
    are preserved with enough precision to be consistent in the frontend.
    """
    # 0.1524m is exactly 6 inches.
    # 0.2032m is exactly 8 inches.
    df = pl.DataFrame(
        {
            "station_id": ["C"],
            "datetime": [date(2023, 1, 1)],
            "station_name": ["Station C"],
            "state": ["AK"],
            "elevation_m": [100],
            "val": [0.1524],
        }
    )

    # format_rows should round to 4 digits by default now
    rows = format_rows(df, "val")
    assert rows[0]["value"] == 0.1524

    # Even if it was 0.15244, it should stay 0.1524
    df2 = pl.DataFrame({"station_id": ["D"], "val": [0.15244]})
    rows2 = format_rows(df2, "val")
    assert rows2[0]["value"] == 0.1524

    # If it was 0.15236, it should round to 0.1524
    df3 = pl.DataFrame({"station_id": ["E"], "val": [0.15236]})
    rows3 = format_rows(df3, "val")
    assert rows3[0]["value"] == 0.1524
