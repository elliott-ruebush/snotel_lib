import logging
import time
from datetime import datetime
from pathlib import Path

import polars as pl
from dateutil.relativedelta import relativedelta
from metloom.pointdata import SnotelPointData
from metloom.variables import SnotelVariables
from pandera.typing.geopandas import GeoDataFrame

from ..config import STATION_CACHE_DAYS
from ..io import cast_to_schema, dtypes_from_schema
from ..schemas import SnotelDataSchema, StationMetadataSchema
from .base import BaseSnotelClient

logger = logging.getLogger(__name__)

STATION_DATA_COLUMN_MAP = {
    "SWE": SnotelDataSchema.swe_m,
    "SNOWDEPTH": SnotelDataSchema.snow_depth_m,
    "PRECIPITATION": SnotelDataSchema.precip_m,
    "AVG AIR TEMP": SnotelDataSchema.tavg_c,
    "MIN AIR TEMP": SnotelDataSchema.tmin_c,
    "MAX AIR TEMP": SnotelDataSchema.tmax_c,
}


class MetloomClient(BaseSnotelClient):
    """Client for fetching SNOTEL data directly via the Metloom library (NRCS AWDB API)."""

    def get_stations_metadata(self, force_update: bool = False) -> GeoDataFrame[StationMetadataSchema]:
        """Fetch metadata for all SNOTEL stations. Uses the fast egagli metadata index since AWDB SOAP is slow for bulk metadata."""
        from .egagli_client import EgagliClient

        logger.info("Delegating metadata fetch to EgagliClient (fast GeoJSON index).")
        return EgagliClient(self.cache_dir).get_stations_metadata(force_update=force_update)

    def get_station_data(
        self,
        station_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        force_update: bool = False,
    ) -> pl.DataFrame:
        """Fetch daily SNOTEL data for a specific station via Metloom."""
        start_time = time.perf_counter()
        # Clean colons from station_id for filename
        safe_station_id = station_id.replace(":", "_")
        cache_path = self.cache_dir / f"metloom_{safe_station_id}.parquet"

        if not force_update and self._is_cache_valid(cache_path, STATION_CACHE_DAYS):
            logger.info(f"Retrieving metloom data for {station_id} from local cache: {cache_path}")
            df = pl.read_parquet(cache_path)
            res = super()._filter_and_process(df, start_date, end_date)
            logger.info(
                f"Data retrieval for {station_id} took {time.perf_counter() - start_time:.2f}s "
                f"(cache hit, {len(res)} rows)"
            )
            return res

        res = self._fetch_and_cache_station_data(station_id, cache_path, start_date, end_date)
        logger.info(
            f"Data retrieval for {station_id} took {time.perf_counter() - start_time:.2f}s "
            f"(cache miss, {len(res)} rows)"
        )
        return res

    def _fetch_and_cache_station_data(
        self, station_id: str, cache_path: Path, start_date: str | None, end_date: str | None
    ) -> pl.DataFrame:
        logger.info(f"Fetching metloom data for {station_id}...")

        # Metloom uses 'name' optionally, we just need the code for instantiation.
        # e.g. station_id = '713:CO:SNTL'
        snotel_point = SnotelPointData(station_id, "Unknown Snotel")

        if not start_date:
            s_date = datetime.now() - relativedelta(years=10)  # 10 year backfill if none
        else:
            s_date = datetime.strptime(start_date, "%Y-%m-%d")

        if not end_date:
            e_date = datetime.now()
        else:
            e_date = datetime.strptime(end_date, "%Y-%m-%d")

        vrs = [
            SnotelVariables.SWE,  # type: ignore[unresolved-attribute]
            SnotelVariables.SNOWDEPTH,  # type: ignore[unresolved-attribute]
            SnotelVariables.PRECIPITATION,  # type: ignore[unresolved-attribute]
            SnotelVariables.TEMPAVG,  # type: ignore[unresolved-attribute]
            SnotelVariables.TEMPMIN,  # type: ignore[unresolved-attribute]
            SnotelVariables.TEMPMAX,  # type: ignore[unresolved-attribute]
        ]

        # fetching could throw an error if no data available
        try:
            pandas_df = snotel_point.get_daily_data(s_date, e_date, vrs)
        except Exception as e:
            logger.error("Failed to fetch data for %s from Metloom: %s", station_id, e)
            # Return empty matching dataframe
            schema = dtypes_from_schema(SnotelDataSchema)
            return pl.DataFrame(schema=schema)

        if pandas_df is None or pandas_df.empty:
            schema = dtypes_from_schema(SnotelDataSchema)
            return pl.DataFrame(schema=schema)

        # Metloom returns a GeoDataFrame with a multi-index of (datetime, site)
        pandas_df = pandas_df.reset_index()

        # PyArrow struggles with shapely geometry objects, drop it before converting
        if "geometry" in pandas_df.columns:
            pandas_df = pandas_df.drop(columns=["geometry"])

        df = pl.from_pandas(pandas_df)

        # Process and save
        df = self._parse_metloom_geodataframe(df)
        df = self._convert_units(df)

        df = cast_to_schema(df, SnotelDataSchema, column_map=STATION_DATA_COLUMN_MAP)
        df = df.sort([SnotelDataSchema.datetime])

        df.write_parquet(cache_path)

        return super()._filter_and_process(df, start_date, end_date)

    def get_all_station_data(self, force_update: bool = False) -> pl.DataFrame:
        raise NotImplementedError(
            "Metloom does not easily support a single bulk download for all station history. Use get_station_data in a parallel map."
        )

    def _parse_metloom_geodataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse the metloom GeoDataFrame to extract site, drop geometry/units, and handle timezone."""
        if "datetime" in df.columns:
            df = df.rename({"datetime": SnotelDataSchema.datetime})

        if SnotelDataSchema.datetime in df.columns:
            dt_col_schema = df.schema[SnotelDataSchema.datetime]
            if dt_col_schema.__class__.__name__ == "Datetime":
                if getattr(dt_col_schema, "time_zone", None) is not None:
                    df = df.with_columns(pl.col(SnotelDataSchema.datetime).dt.replace_time_zone(None))

        # Drop columns ending with _units, geometry, datasource, and site
        drop_cols = [col for col in df.columns if col.endswith("_units") or col in ["geometry", "datasource", "site"]]
        if drop_cols:
            df = df.drop(drop_cols)

        return df

    def _convert_units(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert metloom imperial units to metric (inches to meters, F to C)."""
        # Convert metloom imperial units to metric (inches to meters)
        # Metloom returns SWE, SNWD, PRECIP in inches. TEMP in Fahrenheit.
        if "SWE" in df.columns:
            df = df.with_columns(pl.col("SWE") * 0.0254)
        if "SNOWDEPTH" in df.columns:
            df = df.with_columns(pl.col("SNOWDEPTH") * 0.0254)
        if "PRECIPITATION" in df.columns:
            df = df.with_columns(pl.col("PRECIPITATION") * 0.0254)

        # Fahrenheit to Celsius
        if "AVG AIR TEMP" in df.columns:
            df = df.with_columns((pl.col("AVG AIR TEMP") - 32.0) * (5.0 / 9.0))
        if "MIN AIR TEMP" in df.columns:
            df = df.with_columns((pl.col("MIN AIR TEMP") - 32.0) * (5.0 / 9.0))
        if "MAX AIR TEMP" in df.columns:
            df = df.with_columns((pl.col("MAX AIR TEMP") - 32.0) * (5.0 / 9.0))

        # Add any missing req columns
        for req_col in ["SWE", "SNOWDEPTH", "PRECIPITATION", "AVG AIR TEMP", "MIN AIR TEMP", "MAX AIR TEMP"]:
            if req_col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(req_col))

        return df
