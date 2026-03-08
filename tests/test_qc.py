from datetime import date

import polars as pl
from polars.testing import assert_frame_equal

from snotel_lib.clean import DEFAULT_CHECKS, QCCheck, QCLogSchema, run_qc
from snotel_lib.schemas import SnotelDataSchema


def test_negative_filter_removes_row():
    df = pl.DataFrame(
        {
            SnotelDataSchema.datetime: [date(2023, 1, 1), date(2023, 1, 2)],
            SnotelDataSchema.swe_m: [0.5, -0.1],
            SnotelDataSchema.snow_depth_m: [1.0, 1.0],
        }
    )

    res = run_qc(df, "123_SNTL", DEFAULT_CHECKS)

    expected_data = pl.DataFrame(
        {
            "datetime": [date(2023, 1, 1)],
            "swe_m": [0.5],
            "snow_depth_m": [1.0],
        }
    )
    assert_frame_equal(res.data, expected_data)

    expected_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 2)],
            QCLogSchema.variable: [SnotelDataSchema.swe_m],
            QCLogSchema.qc_type: ["FILTER"],
            QCLogSchema.reason: ["NEGATIVE_VALUE"],
            QCLogSchema.explanation: ["Negative SWE detected"],
        }
    )
    assert_frame_equal(res.qc, expected_qc)


def test_spike_flag_detected():
    dates = [date(2023, 1, i) for i in range(1, 12)]
    # 10 rows stable, 1 row spiked
    df = pl.DataFrame(
        {
            SnotelDataSchema.datetime: dates,
            SnotelDataSchema.swe_m: [0.5] * 10 + [2.0],  # 2.0 > 3 * 0.5
            SnotelDataSchema.snow_depth_m: [1.0] * 11,
        }
    )

    res = run_qc(df, "123_SNTL", DEFAULT_CHECKS)

    # Assert data unchanged (spike is a FLAG, not a FILTER)
    assert_frame_equal(res.data, df)

    expected_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 11)],
            QCLogSchema.variable: [SnotelDataSchema.swe_m],
            QCLogSchema.qc_type: ["FLAG"],
            QCLogSchema.reason: ["SPIKE_3X_MEDIAN"],
            QCLogSchema.explanation: ["Value exceeds 3x 7-day rolling median"],
        }
    )
    assert_frame_equal(res.qc, expected_qc)


def test_combined_filter_and_flag():
    dates = [date(2023, 1, i) for i in range(1, 13)]
    swe_vals = [0.5] * 10 + [-0.5, 2.0]
    df = pl.DataFrame(
        {
            SnotelDataSchema.datetime: dates,
            SnotelDataSchema.swe_m: swe_vals,
            SnotelDataSchema.snow_depth_m: [1.0] * 12,
        }
    )

    res = run_qc(df, "123_SNTL", DEFAULT_CHECKS)

    expected_data = pl.DataFrame(
        {
            SnotelDataSchema.datetime: [*dates[:10], dates[11]],
            SnotelDataSchema.swe_m: [0.5] * 10 + [2.0],
            SnotelDataSchema.snow_depth_m: [1.0] * 11,
        }
    )
    assert_frame_equal(res.data, expected_data)

    expected_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL", "123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 11), date(2023, 1, 12)],
            QCLogSchema.variable: [SnotelDataSchema.swe_m, SnotelDataSchema.swe_m],
            QCLogSchema.qc_type: ["FILTER", "FLAG"],
            QCLogSchema.reason: ["NEGATIVE_VALUE", "SPIKE_3X_MEDIAN"],
            QCLogSchema.explanation: ["Negative SWE detected", "Value exceeds 3x 7-day rolling median"],
        }
    )
    assert_frame_equal(res.qc, expected_qc)


def test_no_issues():
    df = pl.DataFrame(
        {
            SnotelDataSchema.datetime: [date(2023, 1, 1)],
            SnotelDataSchema.swe_m: [0.5],
            SnotelDataSchema.snow_depth_m: [1.0],
        }
    )

    res = run_qc(df, "123_SNTL", DEFAULT_CHECKS)
    assert_frame_equal(res.data, df)

    assert len(res.qc) == 0
    assert QCLogSchema.station_id in res.qc.columns
    assert QCLogSchema.datetime in res.qc.columns


def test_custom_check_registration():
    df = pl.DataFrame(
        {
            SnotelDataSchema.datetime: [date(2023, 1, 1), date(2023, 1, 2)],
            SnotelDataSchema.swe_m: [0.5, 0.5],
            SnotelDataSchema.snow_depth_m: [1.0, 50.0],
        }
    )

    absurd_depth_check = QCCheck(
        variable=SnotelDataSchema.snow_depth_m,
        qc_type="FILTER",
        reason="ABSURD_DEPTH",
        expr=pl.col(SnotelDataSchema.snow_depth_m) > 20.0,
        explanation="Depth > 20m",
    )

    res = run_qc(df, "123_SNTL", [absurd_depth_check])

    expected_data = pl.DataFrame(
        {
            SnotelDataSchema.datetime: [date(2023, 1, 1)],
            SnotelDataSchema.swe_m: [0.5],
            SnotelDataSchema.snow_depth_m: [1.0],
        }
    )
    assert_frame_equal(res.data, expected_data)

    expected_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 2)],
            QCLogSchema.variable: [SnotelDataSchema.snow_depth_m],
            QCLogSchema.qc_type: ["FILTER"],
            QCLogSchema.reason: ["ABSURD_DEPTH"],
            QCLogSchema.explanation: ["Depth > 20m"],
        }
    )
    assert_frame_equal(res.qc, expected_qc)
