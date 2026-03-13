import pandera.pandas as pd_pa
import pandera.polars as pl_pa
import polars as pl
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries


class SnotelDataSchema(pl_pa.DataFrameModel):
    """Schema types for daily SNOTEL station data in Polars."""

    datetime: Series[pl.Date] = pl_pa.Field()
    swe_m: Series[pl.Float32] = pl_pa.Field(nullable=True)
    snow_depth_m: Series[pl.Float32] = pl_pa.Field(nullable=True)
    precip_m: Series[pl.Float32] = pl_pa.Field(nullable=True)
    tavg_c: Series[pl.Float32] = pl_pa.Field(nullable=True)
    tmin_c: Series[pl.Float32] = pl_pa.Field(nullable=True)
    tmax_c: Series[pl.Float32] = pl_pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class AllSnotelDataSchema(SnotelDataSchema):
    """Schema for combined SNOTEL station data in Polars, including station_id."""

    station_id: Series[pl.String] = pl_pa.Field()


class StationMetadataSchema(pd_pa.DataFrameModel):
    """Schema for SNOTEL station metadata."""

    station_id: Series[str] = pd_pa.Field()
    station_name: Series[str] = pd_pa.Field(nullable=True)
    network: Series[str] = pd_pa.Field(nullable=True)
    elevation_m: Series[float] = pd_pa.Field(nullable=True)
    latitude: Series[float] = pd_pa.Field(nullable=True)
    longitude: Series[float] = pd_pa.Field(nullable=True)
    state: Series[str] = pd_pa.Field(nullable=True)
    huc: Series[str] = pd_pa.Field(nullable=True)
    mgrs: Series[str] = pd_pa.Field(nullable=True)
    mountain_range: Series[str] = pd_pa.Field(nullable=True)
    begin_date: Series[pd_pa.Date] = pd_pa.Field(nullable=True)
    end_date: Series[pd_pa.Date] = pd_pa.Field(nullable=True)
    csv_data: Series[bool] = pd_pa.Field(nullable=True)
    geometry: GeoSeries = pd_pa.Field(nullable=True)

    class Config:
        strict = False  # geojson has many columns, we only strictly care about these
        coerce = True
