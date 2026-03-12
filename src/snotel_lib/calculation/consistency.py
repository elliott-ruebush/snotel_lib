import pandera.polars as pl_pa
import polars as pl
from pandera.typing import Series

from snotel_lib.schemas import AllSnotelDataSchema, SnotelDataSchema


class ConsistencySchema(pl_pa.DataFrameModel):
    """Schema for the output of compute_consistency_metrics."""

    station_id: Series[pl.String] = pl_pa.Field()
    std_dev: Series[pl.Float64] = pl_pa.Field(nullable=True)
    all_time_max: Series[pl.Float32] = pl_pa.Field(nullable=True)
    all_time_max_year: Series[pl.Int64] = pl_pa.Field(nullable=True)
    all_time_min: Series[pl.Float32] = pl_pa.Field(nullable=True)
    all_time_min_year: Series[pl.Int64] = pl_pa.Field(nullable=True)
    wy_count: Series[pl.UInt32] = pl_pa.Field()

    class Config:
        strict = True
        coerce = True


def compute_consistency_metrics(df: pl.DataFrame, min_observations_per_year: int = 330) -> pl.DataFrame:
    yearly_df = df.group_by([AllSnotelDataSchema.station_id, "water_year"]).agg(
        [
            pl.col(SnotelDataSchema.snow_depth_m).max().alias("yearly_max_depth"),
            pl.col(SnotelDataSchema.snow_depth_m).is_not_null().sum().alias("obs_count"),
        ]
    )

    valid_yearly_df = yearly_df.filter(
        (pl.col("yearly_max_depth").is_not_null()) & (pl.col("obs_count") >= min_observations_per_year)
    ).sort([AllSnotelDataSchema.station_id, "yearly_max_depth"])

    res = (
        valid_yearly_df.group_by(AllSnotelDataSchema.station_id)
        .agg(
            pl.col("yearly_max_depth").std().alias("std_dev"),
            pl.col("yearly_max_depth").last().alias("all_time_max"),
            pl.col("water_year").last().alias("all_time_max_year"),
            pl.col("yearly_max_depth").first().alias("all_time_min"),
            pl.col("water_year").first().alias("all_time_min_year"),
            pl.col("yearly_max_depth").count().alias("wy_count"),
        )
        .filter(pl.col("wy_count") >= 5)
    )

    return ConsistencySchema.validate(res)
