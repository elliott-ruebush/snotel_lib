from typing import ClassVar

import pandera.pandas as pa
import polars as pl
from pandera.typing import Index, Series
from pandera.typing.geopandas import GeoSeries


class SnotelDataSchema:
    """Schema types for daily SNOTEL station data in Polars."""

    dtypes: ClassVar[dict[str, pl.DataType]] = {  # ty: ignore[invalid-assignment]
        "datetime": pl.Date,
        "swe_m": pl.Float32,
        "snow_depth_m": pl.Float32,
        "precip_m": pl.Float32,
        "tavg_c": pl.Float32,
        "tmin_c": pl.Float32,
        "tmax_c": pl.Float32,
    }


SNOTEL_DATA_DTYPES = SnotelDataSchema.dtypes


class StationMetadataSchema(pa.DataFrameModel):
    """Schema for SNOTEL station metadata."""

    code: Index[str] = pa.Field(check_name=True)
    name: Series[str] = pa.Field(nullable=True)
    network: Series[str] = pa.Field(nullable=True)
    elevation_m: Series[float] = pa.Field(nullable=True)
    latitude: Series[float] = pa.Field(nullable=True)
    longitude: Series[float] = pa.Field(nullable=True)
    state: Series[str] = pa.Field(nullable=True)
    huc: Series[str] = pa.Field(nullable=True)
    mgrs: Series[str] = pa.Field(nullable=True)
    mountain_range: Series[str] = pa.Field(nullable=True)
    begin_date: Series[pa.Date] = pa.Field(nullable=True)
    end_date: Series[pa.Date] = pa.Field(nullable=True)
    csv_data: Series[bool] = pa.Field(nullable=True)
    geometry: GeoSeries = pa.Field(nullable=True)

    class Config:
        strict = False  # geojson has many columns, we only strictly care about these
        coerce = True
