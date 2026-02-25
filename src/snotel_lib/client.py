import io
import logging
import lzma
import tarfile
import time
import typing
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import polars as pl
import requests
from pandera.typing.geopandas import GeoDataFrame

from .config import METADATA_CACHE_DAYS, STATION_CACHE_DAYS, get_cache_dir
from .schemas import SNOTEL_DATA_DTYPES, StationMetadataSchema

logger = logging.getLogger(__name__)

EGAGLI_GEOJSON_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson"
EGAGLI_STATION_CSV_BASE = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station_id}.csv"
EGAGLI_ALL_STATIONS_TAR_URL = (
    "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/all_station_data.tar.lzma"
)

METADATA_COLUMN_MAP = {
    "code": "code",
    "name": "name",
    "network": "network",
    "elevation_m": "elevation_m",
    "latitude": "latitude",
    "longitude": "longitude",
    "state": "state",
    "HUC": "huc",
    "mgrs": "mgrs",
    "mountainRange": "mountain_range",
    "beginDate": "begin_date",
    "endDate": "end_date",
    "csvData": "csv_data",
}

STATION_DATA_COLUMN_MAP = {
    "WTEQ": "swe_m",
    "SNWD": "snow_depth_m",
    "PRCPSA": "precip_m",
    "TMIN": "tmin_c",
    "TMAX": "tmax_c",
    "TAVG": "tavg_c",
}


class SnotelClient:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or get_cache_dir()

    def get_stations_metadata(self, force_update: bool = False) -> GeoDataFrame[StationMetadataSchema]:
        """Fetch metadata for all SNOTEL stations, with caching."""
        start_time = time.perf_counter()
        cache_path = self.cache_dir / "all_stations.parquet"

        if not force_update and self._is_cache_valid(cache_path, METADATA_CACHE_DAYS):
            logger.info(f"Retrieving metadata from local cache: {cache_path}")
            df = gpd.read_parquet(cache_path)
            if df.index.name != "code" and "code" in df.columns:
                df = df.set_index("code")
            res = typing.cast(GeoDataFrame[StationMetadataSchema], StationMetadataSchema.validate(df))
            logger.info(f"Metadata retrieval took {time.perf_counter() - start_time:.2f}s (cache hit)")
            return res

        res = self._fetch_and_cache_metadata(cache_path)
        logger.info(f"Metadata retrieval took {time.perf_counter() - start_time:.2f}s (cache miss)")
        return res

    def _is_cache_valid(self, path: Path, days: int) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=days)

    def _fetch_and_cache_metadata(self, cache_path: Path) -> GeoDataFrame[StationMetadataSchema]:
        logger.info(f"Fetching metadata from internet: {EGAGLI_GEOJSON_URL}")
        response = requests.get(EGAGLI_GEOJSON_URL)
        response.raise_for_status()

        df = gpd.read_file(io.BytesIO(response.content))
        df = df.rename(columns=METADATA_COLUMN_MAP)
        if "code" in df.columns:
            df = df.set_index("code")

        validated_df = typing.cast(GeoDataFrame[StationMetadataSchema], StationMetadataSchema.validate(df))
        validated_df.to_parquet(cache_path)
        return validated_df

    def get_station_data(
        self,
        station_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        force_update: bool = False,
    ) -> pl.DataFrame:
        """Fetch daily SNOTEL data for a specific station, with caching."""
        start_time = time.perf_counter()
        cache_path = self.cache_dir / f"{station_id}.parquet"

        if not force_update and self._is_cache_valid(cache_path, STATION_CACHE_DAYS):
            logger.info(f"Retrieving data for {station_id} from local cache: {cache_path}")
            df = pl.read_parquet(cache_path)
            res = self._filter_and_process(df, start_date, end_date)
            logger.info(f"Data retrieval for {station_id} took {time.perf_counter() - start_time:.2f}s (cache hit)")
            return res

        res = self._fetch_and_cache_station_data(station_id, cache_path, start_date, end_date)
        logger.info(f"Data retrieval for {station_id} took {time.perf_counter() - start_time:.2f}s (cache miss)")
        return res

    def _fetch_and_cache_station_data(
        self, station_id: str, cache_path: Path, start_date: str | None, end_date: str | None
    ) -> pl.DataFrame:
        url = EGAGLI_STATION_CSV_BASE.format(station_id=station_id)
        logger.info(f"Fetching data for {station_id} from internet: {url}")

        response = requests.get(url)
        response.raise_for_status()

        df = pl.read_csv(
            io.BytesIO(response.content),
            try_parse_dates=True,
            null_values=["", "NaN", "NA", "null"],
        )

        df = self._process_raw_polars_data(df)
        df.write_parquet(cache_path)

        return self._filter_and_process(df, start_date, end_date)

    def _process_raw_polars_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rename and cast columns to target schema types."""
        cols_to_rename = {k: v for k, v in STATION_DATA_COLUMN_MAP.items() if k in df.columns}
        df = df.rename(cols_to_rename)

        cast_exprs = [pl.col(col).cast(dtype) for col, dtype in SNOTEL_DATA_DTYPES.items() if col in df.columns]
        df = df.with_columns(cast_exprs)
        return df

    def _filter_and_process(self, df: pl.DataFrame, start_date: str | None, end_date: str | None) -> pl.DataFrame:
        """Ensure column naming and apply date filtering."""
        df = self._process_raw_polars_data(df)

        if start_date:
            df = df.filter(pl.col("datetime") >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            df = df.filter(pl.col("datetime") <= pl.lit(end_date).cast(pl.Date))

        return df

    def get_all_station_data(self, force_update: bool = False) -> pl.DataFrame:
        """Fetch combined daily SNOTEL data for all stations."""
        start_time = time.perf_counter()
        cache_path = self.cache_dir / "all_station_data.parquet"

        if not force_update and self._is_cache_valid(cache_path, STATION_CACHE_DAYS):
            logger.info(f"Retrieving combined data from local cache: {cache_path}")
            res = pl.read_parquet(cache_path)
            logger.info(f"Combined data retrieval took {time.perf_counter() - start_time:.2f}s (cache hit)")
            return res

        logger.info(f"Fetching combined data from internet: {EGAGLI_ALL_STATIONS_TAR_URL}")

        response = requests.get(EGAGLI_ALL_STATIONS_TAR_URL, stream=True)
        response.raise_for_status()

        dfs = self._parse_tar_to_dataframes(response.content)
        combined_df = pl.concat(dfs, how="vertical_relaxed")

        combined_df = self._process_raw_polars_data(combined_df)

        # Add station_id if not already correctly cast
        if "station_id" in combined_df.columns:
            combined_df = combined_df.with_columns(pl.col("station_id").cast(pl.String))

        combined_df.write_parquet(cache_path)
        logger.info(f"Combined data retrieval took {time.perf_counter() - start_time:.2f}s (cache miss)")
        return combined_df

    def _parse_tar_to_dataframes(self, content: bytes) -> list[pl.DataFrame]:
        """Extract CSV files from tar.lzma into a list of DataFrames."""
        dfs = []
        with lzma.open(io.BytesIO(content)) as lzma_file:
            with tarfile.open(fileobj=lzma_file, mode="r:") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".csv"):
                        station_id = member.name.split("/")[-1].replace(".csv", "")
                        f = tar.extractfile(member)
                        if f:
                            csv_bytes = f.read()
                            if not csv_bytes:
                                continue
                            df = pl.read_csv(
                                csv_bytes,
                                try_parse_dates=True,
                                null_values=["", "NaN", "NA", "null"],
                            )
                            df = df.with_columns(pl.lit(station_id).alias("station_id"))
                            dfs.append(df)
        return dfs
