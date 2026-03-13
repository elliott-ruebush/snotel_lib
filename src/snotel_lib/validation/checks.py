import polars as pl

from snotel_lib.schemas import AllSnotelDataSchema

from ..constants import (
    MAX_PRECIP_M,
    MAX_SNOW_DENSITY,
    MAX_SNOW_DEPTH_M,
    MAX_SWE_M,
    MIN_DEPTH_M_FOR_DENSITY_CHECK,
    MIN_PRECIP_M,
    MIN_SNOW_DENSITY,
    MIN_SNOW_DEPTH_M,
    MIN_SWE_M,
    MIN_SWE_M_FOR_DEPTH_CHECK,
    PRECIP_SWE_MAX_DELTA_M,
    SPIKE_LIMIT_PRECIP_M,
    SPIKE_LIMIT_SNOW_DEPTH_M,
    SPIKE_LIMIT_SWE_M,
)
from ..schemas import SnotelDataSchema
from .models import FilterCheck, FilterList, FlagCheck, FlagList


def range_filter(
    col: str,
    name: str,
    *,
    low: float | None = None,
    high: float | None = None,
) -> FilterCheck:
    """Filter rows where col falls outside [low, high] (either bound optional)."""
    if low is not None and high is not None:
        expr = ~pl.col(col).is_between(low, high)
    elif low is not None:
        expr = pl.col(col) < low
    elif high is not None:
        expr = pl.col(col) > high
    else:
        raise ValueError("At least one of low or high must be provided")

    bounds = f"[{low if low is not None else '-∞'}, {high if high is not None else '+∞'}]"
    return FilterCheck(
        name=name,
        expr=expr,
        explanation=f"{col} outside plausible range {bounds}",
    )


# Negative values and physically implausible upper bounds
SNOW_DEPTH_RANGE_FILTER = range_filter(
    SnotelDataSchema.snow_depth_m, "SNOW_DEPTH_RANGE_FILTER", low=MIN_SNOW_DEPTH_M, high=MAX_SNOW_DEPTH_M
)
SWE_RANGE_FILTER = range_filter(SnotelDataSchema.swe_m, "SWE_RANGE_FILTER", low=MIN_SWE_M, high=MAX_SWE_M)
PRECIP_RANGE_FILTER = range_filter(
    SnotelDataSchema.precip_m, "PRECIP_RANGE_FILTER", low=MIN_PRECIP_M, high=MAX_PRECIP_M
)


# SWE > snow depth is physically impossible - snow can't be denser in water weight than water
# We only apply this check if SWE is at least 1 inch (0.0254m) to avoid noise at near-zero values
def swe_exceeds_snow_depth_filter(min_swe_m: float = MIN_SWE_M_FOR_DEPTH_CHECK) -> FilterCheck:
    return FilterCheck(
        name="SWE_EXCEEDS_SNOW_DEPTH",
        expr=(pl.col(SnotelDataSchema.swe_m) > min_swe_m)
        & (pl.col(SnotelDataSchema.swe_m) > pl.col(SnotelDataSchema.snow_depth_m)),
        explanation="SWE exceeds snow depth, which is physically impossible",
    )


SWE_EXCEEDS_SNOW_DEPTH_FILTER = swe_exceeds_snow_depth_filter()


def unlikely_snow_ratio_flag(minimum_density: float, maximum_density: float, min_depth_m: float) -> FlagCheck:
    """
    Flag for snow ratios that fall outside of an expected range. Upper bounding somewhere around firn's density (year-round snow)
    https://earthscience.stackexchange.com/questions/4391/converting-glacier-volume-to-mass-what-ice-density-to-use
    Only applied when snow depth is above a minimum threshold to avoid noise when snowpack is negligible
    """
    has_min_snow_depth = pl.col(SnotelDataSchema.snow_depth_m) > min_depth_m
    has_positive_swe = pl.col(SnotelDataSchema.swe_m) > 0

    snow_density_ratio = pl.col(SnotelDataSchema.swe_m) / pl.col(SnotelDataSchema.snow_depth_m)
    is_plausible_snow_density = snow_density_ratio.is_between(minimum_density, maximum_density)
    return FlagCheck(
        name="IMPLAUSIBLE_SNOW_DENSITY",
        expr=(has_min_snow_depth & has_positive_swe & ~is_plausible_snow_density),
        explanation=f"SWE/snow_depth ratio outside expected range of [{minimum_density}, {maximum_density}]",
    )


SNOW_DENSITY_FLAG = unlikely_snow_ratio_flag(
    minimum_density=MIN_SNOW_DENSITY, maximum_density=MAX_SNOW_DENSITY, min_depth_m=MIN_DEPTH_M_FOR_DENSITY_CHECK
)


def precip_vs_swe_change_flag(threshold_m: float) -> FlagCheck:
    """
    Cross-variable check: SWE increase should not significantly differ from precipitation increase
    as they both measure water.

    Note: this flag does not necessarily mean the data is incorrect. Wind transport and
    snowpack dynamics can lead to discrepancies in snow pillow and precipitation gauge measurements.
    See the below link for an fun write-up by Dr. Mark W. Williams from CU Boulder's INSTAAR.
    http://snobear.colorado.edu/SnowHydro/Measuring_Snow/Snotel/snotel.html
    """
    swe_delta = pl.col(SnotelDataSchema.swe_m).diff().over(AllSnotelDataSchema.station_id)
    precip_delta = pl.col(SnotelDataSchema.precip_m).diff().over(AllSnotelDataSchema.station_id)
    return FlagCheck(
        name="PRECIP_VS_SWE_MISMATCH",
        expr=(swe_delta - precip_delta).abs() > threshold_m,
        explanation=f"Daily changes in SWE and precipitation differ by more than {threshold_m}m",
    )


PRECIP_VS_SWE_CHANGE_FLAG = precip_vs_swe_change_flag(threshold_m=PRECIP_SWE_MAX_DELTA_M)


def day_over_day_delta_flag(col: str, name: str, limit: float) -> FlagCheck:
    """Flag rows where the day-over-day increase exceeds `limit`, partitioned per station."""
    return FlagCheck(
        name=f"LARGE_DAILY_INCREASE_{name}",
        expr=pl.col(col).diff().over(AllSnotelDataSchema.station_id) > limit,
        explanation=f"{col} increased by more than {limit} in a single day",
    )


# Large single-day increases are suspicious but not impossible (e.g. extreme storm)
SPIKE_SNOW_DEPTH_FLAG = day_over_day_delta_flag(
    SnotelDataSchema.snow_depth_m, "SNOW_DEPTH", limit=SPIKE_LIMIT_SNOW_DEPTH_M
)
SPIKE_SWE_FLAG = day_over_day_delta_flag(SnotelDataSchema.swe_m, "SWE", limit=SPIKE_LIMIT_SWE_M)
SPIKE_PRECIP_FLAG = day_over_day_delta_flag(SnotelDataSchema.precip_m, "PRECIP", limit=SPIKE_LIMIT_PRECIP_M)

DEFAULT_FILTERS = FilterList(
    [
        SNOW_DEPTH_RANGE_FILTER,
        SWE_RANGE_FILTER,
        PRECIP_RANGE_FILTER,
        SWE_EXCEEDS_SNOW_DEPTH_FILTER,
    ]
)

DEFAULT_FLAGS = FlagList(
    [
        SPIKE_SNOW_DEPTH_FLAG,
        SPIKE_SWE_FLAG,
        SNOW_DENSITY_FLAG,
        PRECIP_VS_SWE_CHANGE_FLAG,
    ]
)
