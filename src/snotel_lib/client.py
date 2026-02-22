import io
import logging
import typing
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pandera.typing import DataFrame
from pandera.typing.geopandas import GeoDataFrame

from .config import METADATA_CACHE_DAYS, STATION_CACHE_DAYS, get_cache_dir
from .schemas import SnotelDataSchema, StationMetadataSchema

logger = logging.getLogger(__name__)

EGAGLI_GEOJSON_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson"
EGAGLI_STATION_CSV_BASE = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station_id}.csv"


class SnotelClient:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or get_cache_dir()

    def get_stations_metadata(self, force_update: bool = False) -> GeoDataFrame[StationMetadataSchema]:
        """
        Fetch metadata for all SNOTEL stations.
        Caches the result locally as GeoParquet.
        """
        cache_path = self.cache_dir / "all_stations.parquet"

        if not force_update and cache_path.exists():
            # Check if cache is still valid
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(days=METADATA_CACHE_DAYS):
                logger.info(f"Retrieving metadata from local GeoParquet cache: {cache_path}")
                df = gpd.read_parquet(cache_path).set_index("code")
                logger.debug(f"Columns in metadata: {df.columns.tolist()}")
                return typing.cast(GeoDataFrame[StationMetadataSchema], StationMetadataSchema.validate(df))

        # Fetch and cache
        logger.info(f"Fetching metadata from internet: {EGAGLI_GEOJSON_URL}")
        response = requests.get(EGAGLI_GEOJSON_URL)
        response.raise_for_status()

        # Load raw geojson and save as geoparquet
        df = gpd.read_file(io.BytesIO(response.content))
        df.to_parquet(cache_path)

        df = df.set_index("code")
        logger.debug(f"Columns in fetched metadata: {df.columns.tolist()}")
        return typing.cast(GeoDataFrame[StationMetadataSchema], StationMetadataSchema.validate(df))

    def get_station_data(
        self,
        station_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        force_update: bool = False,
    ) -> DataFrame[SnotelDataSchema]:
        """
        Fetch daily SNOTEL data for a specific station.
        Caches the result locally as Parquet.

        Parameters:
        -----------
        station_id : str
            The station ID (e.g., '679_WA_SNTL')
        start_date : str, optional
            ISO format string 'YYYY-MM-DD'
        end_date : str, optional
            ISO format string 'YYYY-MM-DD'
        force_update : bool, optional
            If True, re-fetches even if cached data exists.
        """
        cache_path = self.cache_dir / f"{station_id}.parquet"

        if not force_update and cache_path.exists():
            # Check cache validity
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(days=STATION_CACHE_DAYS):
                logger.info(f"Retrieving data for {station_id} from local Parquet cache: {cache_path}")
                df = pd.read_parquet(cache_path)
                return self._filter_and_process(df, start_date, end_date)

        # Fetch and cache
        url = EGAGLI_STATION_CSV_BASE.format(station_id=station_id)
        logger.info(f"Fetching data for {station_id} from internet: {url}")
        response = requests.get(url)
        response.raise_for_status()

        # Load raw CSV and save as parquet
        df = pd.read_csv(io.BytesIO(response.content), index_col="datetime", parse_dates=True)
        # Note: We save the full dataframe before filtering to keep the cache complete
        df.to_parquet(cache_path)

        return self._filter_and_process(df, start_date, end_date)

    def _filter_and_process(
        self, df: pd.DataFrame, start_date: str | None, end_date: str | None
    ) -> DataFrame[SnotelDataSchema]:
        """Rename columns and filter by date range."""
        # Rename based on egagli schema to more readable names
        column_map = {
            "WTEQ": "swe_m",
            "SNWD": "snow_depth_m",
            "PRCPSA": "precip_m",
            "TMIN": "tmin_c",
            "TMAX": "tmax_c",
            "TAVG": "tavg_c",
        }
        df = df.rename(columns=column_map)
        logger.debug(f"Columns after renaming: {df.columns.tolist()}")

        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]

        # Coerce and validate
        return SnotelDataSchema.validate(df)
