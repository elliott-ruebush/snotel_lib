import polars as pl

from ..schemas import SnotelDataSchema
from .models import QCCheck

NEGATIVE_SNOW_DEPTH_FILTER = QCCheck(
    variable=SnotelDataSchema.snow_depth_m,
    qc_type="FILTER",
    reason="NEGATIVE_VALUE",
    expr=pl.col(SnotelDataSchema.snow_depth_m) < 0,
    explanation="Negative snow depth detected",
)

NEGATIVE_SWE_FILTER = QCCheck(
    variable=SnotelDataSchema.swe_m,
    qc_type="FILTER",
    reason="NEGATIVE_VALUE",
    expr=pl.col(SnotelDataSchema.swe_m) < 0,
    explanation="Negative SWE detected",
)

SENSOR_ERROR_SNOW_DEPTH_FILTER = QCCheck(
    variable=SnotelDataSchema.snow_depth_m,
    qc_type="FILTER",
    reason="SENSOR_ERROR_HIGH",
    expr=pl.col(SnotelDataSchema.snow_depth_m) > 15,
    explanation="Snow depth exceeds physically plausible maximum (15m)",
)

SENSOR_ERROR_SWE_FILTER = QCCheck(
    variable=SnotelDataSchema.swe_m,
    qc_type="FILTER",
    reason="SENSOR_ERROR_HIGH",
    expr=pl.col(SnotelDataSchema.swe_m) > 5,
    explanation="SWE exceeds physically plausible maximum (5m)",
)


def _spike_expr(col: str) -> pl.Expr:
    # Uses backward-looking rolling median to ensure stability
    rolling_median = pl.col(col).rolling_median(window_size=7, center=False, min_samples=3)
    return pl.col(col) > (3 * rolling_median)


SPIKE_SNOW_DEPTH_FLAG = QCCheck(
    variable=SnotelDataSchema.snow_depth_m,
    qc_type="FLAG",
    reason="SPIKE_3X_MEDIAN",
    expr=_spike_expr(SnotelDataSchema.snow_depth_m),
    explanation="Value exceeds 3x 7-day rolling median",
)

SPIKE_SWE_FLAG = QCCheck(
    variable=SnotelDataSchema.swe_m,
    qc_type="FLAG",
    reason="SPIKE_3X_MEDIAN",
    expr=_spike_expr(SnotelDataSchema.swe_m),
    explanation="Value exceeds 3x 7-day rolling median",
)

DEFAULT_CHECKS = [
    NEGATIVE_SNOW_DEPTH_FILTER,
    NEGATIVE_SWE_FILTER,
    SENSOR_ERROR_SNOW_DEPTH_FILTER,
    SENSOR_ERROR_SWE_FILTER,
    SPIKE_SNOW_DEPTH_FLAG,
    SPIKE_SWE_FLAG,
]
