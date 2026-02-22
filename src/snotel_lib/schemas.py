import pandas as pd
import pandera.pandas as pa
from pandera.typing import Index, Series
from pandera.typing.geopandas import GeoSeries


class SnotelDataSchema(pa.DataFrameModel):
    """Schema for daily SNOTEL station data."""

    datetime: Index[pd.Timestamp] = pa.Field(check_name=True)
    swe_m: Series[float] = pa.Field(nullable=True)
    snow_depth_m: Series[float] = pa.Field(nullable=True)
    precip_m: Series[float] = pa.Field(nullable=True)
    tavg_c: Series[float] = pa.Field(nullable=True)
    tmin_c: Series[float] = pa.Field(nullable=True)
    tmax_c: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = False  # Allow other columns if egagli adds them
        coerce = True  # Coerce types if possible


class StationMetadataSchema(pa.DataFrameModel):
    """Schema for SNOTEL station metadata."""

    code: Index[str] = pa.Field(check_name=True)
    name: Series[str] = pa.Field(nullable=True)
    network: Series[str] = pa.Field(nullable=True)
    elevation_m: Series[float] = pa.Field(nullable=True)
    latitude: Series[float] = pa.Field(nullable=True)
    longitude: Series[float] = pa.Field(nullable=True)
    state: Series[str] = pa.Field(nullable=True)
    geometry: GeoSeries = pa.Field(nullable=True)

    class Config:
        strict = False  # geojson has many columns, we only strictly care about these
        coerce = True
