import pandera.polars as pl_pa
import polars as pl
from pandera.typing import Series

from snotel_lib.schemas import AllSnotelDataSchema, SnotelDataSchema


class DiffMetricsSchema(pl_pa.DataFrameModel):
    """Schema for the output of compute_diff_metrics."""

    station_id: Series[pl.String] = pl_pa.Field()
    datetime: Series[pl.Date] = pl_pa.Field()

    # We include these but they might be optional depending on the station data
    swe_m: Series[pl.Float32] = pl_pa.Field(nullable=True)
    snow_depth_m: Series[pl.Float32] = pl_pa.Field(nullable=True)

    snow_depth_24h_diff: Series[pl.Float32] = pl_pa.Field(nullable=True)
    snow_depth_48h_diff: Series[pl.Float32] = pl_pa.Field(nullable=True)
    snow_depth_7d_diff: Series[pl.Float32] = pl_pa.Field(nullable=True)

    swe_24h_diff: Series[pl.Float32] = pl_pa.Field(nullable=True)
    swe_48h_diff: Series[pl.Float32] = pl_pa.Field(nullable=True)
    swe_7d_diff: Series[pl.Float32] = pl_pa.Field(nullable=True)

    class Config:
        strict = False  # Diff metrics might contain other columns from original DF
        coerce = True


def compute_diff_metrics(df: pl.DataFrame, recent_cutoff) -> pl.DataFrame:
    diff_df = df.with_columns(
        [
            (
                pl.col(SnotelDataSchema.snow_depth_m)
                - pl.col(SnotelDataSchema.snow_depth_m).shift(1).over(AllSnotelDataSchema.station_id)
            ).alias("snow_depth_24h_diff"),
            (
                pl.col(SnotelDataSchema.snow_depth_m)
                - pl.col(SnotelDataSchema.snow_depth_m).shift(2).over(AllSnotelDataSchema.station_id)
            ).alias("snow_depth_48h_diff"),
            (
                pl.col(SnotelDataSchema.snow_depth_m)
                - pl.col(SnotelDataSchema.snow_depth_m).shift(7).over(AllSnotelDataSchema.station_id)
            ).alias("snow_depth_7d_diff"),
            (
                pl.col(SnotelDataSchema.swe_m)
                - pl.col(SnotelDataSchema.swe_m).shift(1).over(AllSnotelDataSchema.station_id)
            ).alias("swe_24h_diff"),
            (
                pl.col(SnotelDataSchema.swe_m)
                - pl.col(SnotelDataSchema.swe_m).shift(2).over(AllSnotelDataSchema.station_id)
            ).alias("swe_48h_diff"),
            (
                pl.col(SnotelDataSchema.swe_m)
                - pl.col(SnotelDataSchema.swe_m).shift(7).over(AllSnotelDataSchema.station_id)
            ).alias("swe_7d_diff"),
        ]
    )
    res = (
        diff_df.filter(pl.col(SnotelDataSchema.datetime) >= recent_cutoff)
        .group_by(AllSnotelDataSchema.station_id)
        .last()
    )

    return DiffMetricsSchema.validate(res)
