from datetime import date

import polars as pl
from polars.testing import assert_frame_equal

from snotel_lib.schemas import AllSnotelDataSchema, SnotelDataSchema
from snotel_lib.validation import (
    FilterCheck,
    FilterList,
    FlagList,
    QCLogSchema,
    day_over_day_delta_flag,
    precip_vs_swe_change_flag,
    range_filter,
    run_qc,
    swe_exceeds_snow_depth_filter,
    unlikely_snow_ratio_flag,
)


def test_out_of_range_filter_removes_row():
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL", "123_SNTL"],
            SnotelDataSchema.datetime: [date(2023, 1, 1), date(2023, 1, 2)],
            SnotelDataSchema.swe_m: [0.5, -0.1],
            SnotelDataSchema.snow_depth_m: [1.0, 1.0],
            SnotelDataSchema.precip_m: [0.5, 0.5],
        }
    )

    res = run_qc(
        df, FilterList([range_filter(SnotelDataSchema.swe_m, "SWE_RANGE_FILTER", low=0, high=5)]), FlagList([])
    )

    expected_data = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL"],
            AllSnotelDataSchema.datetime: [date(2023, 1, 1)],
            AllSnotelDataSchema.swe_m: [0.5],
            AllSnotelDataSchema.snow_depth_m: [1.0],
            AllSnotelDataSchema.precip_m: [0.5],
        }
    )
    assert_frame_equal(res.data, expected_data)

    expected_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 2)],
            QCLogSchema.name: ["SWE_RANGE_FILTER"],
            QCLogSchema.explanation: ["swe_m outside plausible range [0, 5]"],
        }
    )
    assert_frame_equal(res.filter_log, expected_qc)
    assert len(res.flag_log) == 0


def test_spike_flag_detected_multi_station():
    """Verify spikes are detected and correctly partitioned across multiple stations."""
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["STA1", "STA1", "STA1", "STA2", "STA2", "STA2"],
            AllSnotelDataSchema.datetime: [
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3),
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3),
            ],
            AllSnotelDataSchema.swe_m: [
                0.5,
                0.5,
                1.1,  # STA1 Spike on Jan 3: 0.6 > 0.5
                0.5,
                0.6,
                0.7,  # STA2 No spike: 0.1 < 0.5
            ],
            AllSnotelDataSchema.snow_depth_m: [2.0] * 6,  # Ensure depth > SWE to avoid filtering
            AllSnotelDataSchema.precip_m: [1.0] * 6,
        }
    )

    res = run_qc(df, FilterList([]), FlagList([day_over_day_delta_flag(SnotelDataSchema.swe_m, "SWE", limit=0.5)]))

    # Spike should only be flagged for STA1 on Jan 3
    expected_spike = pl.DataFrame(
        {
            QCLogSchema.station_id: ["STA1"],
            QCLogSchema.datetime: [date(2023, 1, 3)],
            QCLogSchema.name: ["LARGE_DAILY_INCREASE_SWE"],
            QCLogSchema.explanation: ["swe_m increased by more than 0.5 in a single day"],
        }
    )
    spike_flags = res.flag_log.filter(pl.col(QCLogSchema.name) == "LARGE_DAILY_INCREASE_SWE")
    assert_frame_equal(spike_flags, expected_spike)


def test_combined_filter_and_flag():
    dates = [date(2023, 1, i) for i in range(1, 13)]
    swe_vals = [0.5] * 10 + [-0.5, 1.1]
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL"] * 12,
            SnotelDataSchema.datetime: dates,
            SnotelDataSchema.swe_m: swe_vals,
            SnotelDataSchema.snow_depth_m: [1.0] * 11 + [2.0],  # depth=2.0 for swe=1.1
            SnotelDataSchema.precip_m: [0.1] * 12,  # Start low
        }
    ).with_columns(
        pl.when(pl.col(SnotelDataSchema.datetime) == date(2023, 1, 12))
        .then(0.7)
        .otherwise(pl.col(SnotelDataSchema.precip_m))
        .alias(SnotelDataSchema.precip_m)
    )

    res = run_qc(
        df,
        FilterList([range_filter(SnotelDataSchema.swe_m, "SWE_RANGE_FILTER", low=0, high=5)]),
        FlagList([day_over_day_delta_flag(SnotelDataSchema.swe_m, "SWE", limit=0.5)]),
    )

    expected_data = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL"] * 11,
            SnotelDataSchema.datetime: [*dates[:10], dates[11]],
            SnotelDataSchema.swe_m: [0.5] * 10 + [1.1],
            SnotelDataSchema.snow_depth_m: [1.0] * 10 + [2.0],
            SnotelDataSchema.precip_m: [0.1] * 10 + [0.7],
        }
    )
    assert_frame_equal(res.data.select(expected_data.columns), expected_data)

    expected_filter_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 11)],
            QCLogSchema.name: ["SWE_RANGE_FILTER"],
            QCLogSchema.explanation: ["swe_m outside plausible range [0, 5]"],
        }
    )
    assert_frame_equal(res.filter_log, expected_filter_qc)

    expected_flag_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 12)],
            QCLogSchema.name: ["LARGE_DAILY_INCREASE_SWE"],
            QCLogSchema.explanation: ["swe_m increased by more than 0.5 in a single day"],
        }
    )
    assert_frame_equal(res.flag_log, expected_flag_qc)


def test_no_issues():
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL"],
            SnotelDataSchema.datetime: [date(2023, 1, 1)],
            SnotelDataSchema.swe_m: [0.5],
            SnotelDataSchema.snow_depth_m: [1.0],
            SnotelDataSchema.precip_m: [0.5],
        }
    )

    res = run_qc(
        df, FilterList([range_filter(SnotelDataSchema.swe_m, "SWE_RANGE_FILTER", low=0, high=5)]), FlagList([])
    )
    assert_frame_equal(res.data, df)

    assert len(res.filter_log) == 0
    assert len(res.flag_log) == 0


def test_custom_check_registration():
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL", "123_SNTL"],
            SnotelDataSchema.datetime: [date(2023, 1, 1), date(2023, 1, 2)],
            SnotelDataSchema.swe_m: [0.5, 0.5],
            SnotelDataSchema.snow_depth_m: [1.0, 50.0],
            SnotelDataSchema.precip_m: [0.5, 0.5],
        }
    )

    absurd_depth_check = FilterCheck(
        name="ABSURD_DEPTH",
        expr=pl.col(SnotelDataSchema.snow_depth_m) > 20.0,
        explanation="Depth > 20m",
    )

    res = run_qc(df, FilterList([absurd_depth_check]), FlagList([]))

    expected_data = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL"],
            SnotelDataSchema.datetime: [date(2023, 1, 1)],
            SnotelDataSchema.swe_m: [0.5],
            SnotelDataSchema.snow_depth_m: [1.0],
            SnotelDataSchema.precip_m: [0.5],
        }
    )
    assert_frame_equal(res.data, expected_data)

    expected_qc = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 2)],
            QCLogSchema.name: ["ABSURD_DEPTH"],
            QCLogSchema.explanation: ["Depth > 20m"],
        }
    )
    assert_frame_equal(res.filter_log, expected_qc)


def test_swe_exceeds_snow_depth():
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL", "123_SNTL"],
            SnotelDataSchema.datetime: [date(2023, 1, 1), date(2023, 1, 2)],
            SnotelDataSchema.swe_m: [1.5, 0.02],
            SnotelDataSchema.snow_depth_m: [1.0, 0.01],
            SnotelDataSchema.precip_m: [0.9, 0.9],  # Stay within [0, 1] range to avoid precip filter
        }
    )
    res = run_qc(df, FilterList([swe_exceeds_snow_depth_filter()]), FlagList([]))
    expected_filter = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 1)],
            QCLogSchema.name: ["SWE_EXCEEDS_SNOW_DEPTH"],
            QCLogSchema.explanation: ["SWE exceeds snow depth, which is physically impossible"],
        }
    )
    assert_frame_equal(res.filter_log, expected_filter)

    expected_data = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL"],
            SnotelDataSchema.datetime: [date(2023, 1, 2)],
            SnotelDataSchema.swe_m: [0.02],
            SnotelDataSchema.snow_depth_m: [0.01],
            SnotelDataSchema.precip_m: [0.9],
        }
    )
    assert_frame_equal(res.data.select(expected_data.columns), expected_data)


def test_snow_density_flag():
    """Verify that the snow density flag correctly identifies physically implausible SWE/depth ratios."""
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["S1"] * 4,
            AllSnotelDataSchema.datetime: [date(2023, 1, i) for i in range(1, 5)],
            AllSnotelDataSchema.swe_m: [
                0.01,  # 1% density (too thin)
                0.10,  # 10% density (normal)
                0.70,  # 70% density (too thick/dense)
                0.04,  # >70% density, but shallow (< 0.05 min_depth threshold)
            ],
            AllSnotelDataSchema.snow_depth_m: [1.0, 1.0, 1.0, 0.04],
        }
    )

    # Isolate the snow density flag for testing
    res = run_qc(
        df,
        FilterList([]),
        FlagList([unlikely_snow_ratio_flag(minimum_density=0.02, maximum_density=0.6, min_depth_m=0.05)]),
    )

    # Should have 2 flags: 1% and 70% (4th row is bypassed because depth is < 0.05m min threshold)
    expected_flags = pl.DataFrame(
        {
            QCLogSchema.station_id: ["S1", "S1"],
            QCLogSchema.datetime: [date(2023, 1, 1), date(2023, 1, 3)],
            QCLogSchema.name: ["IMPLAUSIBLE_SNOW_DENSITY"] * 2,
            QCLogSchema.explanation: [
                "SWE/snow_depth ratio outside expected range of [0.02, 0.6]",
                "SWE/snow_depth ratio outside expected range of [0.02, 0.6]",
            ],
        }
    )
    flags = res.flag_log.filter(pl.col(QCLogSchema.name) == "IMPLAUSIBLE_SNOW_DENSITY")
    assert_frame_equal(flags, expected_flags)


def test_precip_vs_swe_mismatch():
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["123_SNTL"] * 2,
            SnotelDataSchema.datetime: [date(2023, 1, 1), date(2023, 1, 2)],
            SnotelDataSchema.swe_m: [0.5, 0.7],  # +0.2m increase
            SnotelDataSchema.precip_m: [0.5, 0.55],  # 0.05m daily precip
            SnotelDataSchema.snow_depth_m: [1.0, 1.5],
        }
    )
    res = run_qc(df, FilterList([]), FlagList([precip_vs_swe_change_flag(threshold_m=0.1)]))
    expected_flag = pl.DataFrame(
        {
            QCLogSchema.station_id: ["123_SNTL"],
            QCLogSchema.datetime: [date(2023, 1, 2)],
            QCLogSchema.name: ["PRECIP_VS_SWE_MISMATCH"],
            QCLogSchema.explanation: ["Daily changes in SWE and precipitation differ by more than 0.1m"],
        }
    )
    assert_frame_equal(res.flag_log, expected_flag)


def test_list_concatenation_and_exclude():
    c1 = FilterCheck(name="C1", expr=pl.col(AllSnotelDataSchema.station_id) == "A")
    c2 = FilterCheck(name="C2", expr=pl.col(AllSnotelDataSchema.station_id) == "B")

    l1 = FilterList([c1])
    l2 = FilterList([c2])

    combined = l1 + l2
    assert combined.checks == [c1, c2]

    excluded = combined.exclude(c1)
    assert excluded.checks == [c2]


def test_precip_vs_swe_cross_check_behavior():
    # Test that it flags when diff exceeds threshold across multiple stations
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["A", "A", "B", "B"],
            AllSnotelDataSchema.datetime: [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 2)],
            AllSnotelDataSchema.swe_m: [
                0.0,
                0.2,  # A: +0.2 change (Mismatch)
                0.5,
                0.5,  # B: 0.0 change (No mismatch)
            ],
            AllSnotelDataSchema.precip_m: [
                0.0,
                0.0,  # A: 0 change
                0.0,
                0.0,  # B: 0 change
            ],
            AllSnotelDataSchema.snow_depth_m: [1.0] * 4,
        }
    )

    res = run_qc(df, FilterList([]), FlagList([precip_vs_swe_change_flag(threshold_m=0.1)]))

    expected_mismatch = pl.DataFrame(
        {
            QCLogSchema.station_id: ["A"],
            QCLogSchema.datetime: [date(2023, 1, 2)],
            QCLogSchema.name: ["PRECIP_VS_SWE_MISMATCH"],
            QCLogSchema.explanation: ["Daily changes in SWE and precipitation differ by more than 0.1m"],
        }
    )
    mismatches = res.flag_log.filter(pl.col(QCLogSchema.name) == "PRECIP_VS_SWE_MISMATCH")
    assert_frame_equal(mismatches, expected_mismatch)


def test_qc_temporal_ordering_robustness():
    """Verify that run_qc correctly handles shuffled input by sorting before checks."""
    df = pl.DataFrame(
        {
            AllSnotelDataSchema.station_id: ["STA1", "STA1"],
            AllSnotelDataSchema.datetime: [date(2023, 1, 2), date(2023, 1, 1)],
            AllSnotelDataSchema.swe_m: [0.7, 0.5],  # Chronologically: 0.5 -> 0.7 (increase of 0.2)
            AllSnotelDataSchema.precip_m: [0.0, 0.0],
            AllSnotelDataSchema.snow_depth_m: [1.0, 1.0],
        }
    )

    check = precip_vs_swe_change_flag(threshold_m=0.1)
    res = run_qc(df, FilterList([]), FlagList([check]))

    expected_flag = pl.DataFrame(
        {
            QCLogSchema.station_id: ["STA1"],
            QCLogSchema.datetime: [date(2023, 1, 2)],
            QCLogSchema.name: ["PRECIP_VS_SWE_MISMATCH"],
            QCLogSchema.explanation: ["Daily changes in SWE and precipitation differ by more than 0.1m"],
        }
    )
    assert_frame_equal(res.flag_log, expected_flag)
