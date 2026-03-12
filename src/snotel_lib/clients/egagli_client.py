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

from ..config import METADATA_CACHE_DAYS, STATION_CACHE_DAYS, get_cache_dir
from ..io import cast_to_schema
from ..schemas import AllSnotelDataSchema, SnotelDataSchema, StationMetadataSchema
from .base import BaseSnotelClient

logger = logging.getLogger(__name__)

EGAGLI_GEOJSON_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson"
EGAGLI_STATION_CSV_BASE = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station_id}.csv"
EGAGLI_ALL_STATIONS_TAR_URL = (
    "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/all_station_data.tar.lzma"
)

METADATA_COLUMN_MAP = {
    "code": StationMetadataSchema.code,
    "name": StationMetadataSchema.name,
    "network": StationMetadataSchema.network,
    "elevation_m": StationMetadataSchema.elevation_m,
    "latitude": StationMetadataSchema.latitude,
    "longitude": StationMetadataSchema.longitude,
    "state": StationMetadataSchema.state,
    "HUC": StationMetadataSchema.huc,
    "mgrs": StationMetadataSchema.mgrs,
    "mountainRange": StationMetadataSchema.mountain_range,
    "beginDate": StationMetadataSchema.begin_date,
    "endDate": StationMetadataSchema.end_date,
    "csvData": StationMetadataSchema.csv_data,
}

STATION_DATA_COLUMN_MAP = {
    "WTEQ": SnotelDataSchema.swe_m,
    "SNWD": SnotelDataSchema.snow_depth_m,
    "PRCPSA": SnotelDataSchema.precip_m,
    "TMIN": SnotelDataSchema.tmin_c,
    "TMAX": SnotelDataSchema.tmax_c,
    "TAVG": SnotelDataSchema.tavg_c,
}


class EgagliClient(BaseSnotelClient):
    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        self.cache_dir = cache_dir or get_cache_dir()

    def get_stations_metadata(self, force_update: bool = False) -> GeoDataFrame[StationMetadataSchema]:
        """Fetch metadata for all SNOTEL stations, with caching."""
        start_time = time.perf_counter()
        cache_path = self.cache_dir / "all_stations.parquet"

        if not force_update and self._is_cache_valid(cache_path, METADATA_CACHE_DAYS):
            logger.info(f"Retrieving metadata from local cache: {cache_path}")
            df = gpd.read_parquet(cache_path)
            if df.index.name != StationMetadataSchema.code and StationMetadataSchema.code in df.columns:
                df = df.set_index(StationMetadataSchema.code)
            res = typing.cast(GeoDataFrame[StationMetadataSchema], StationMetadataSchema.validate(df))
            logger.info(
                f"Metadata retrieval took {time.perf_counter() - start_time:.2f}s (cache hit, {len(res)} stations)"
            )
            return res

        res = self._fetch_and_cache_metadata(cache_path)
        logger.info(
            f"Metadata retrieval took {time.perf_counter() - start_time:.2f}s (cache miss, {len(res)} stations)"
        )
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
        if StationMetadataSchema.code in df.columns:
            df = df.set_index(StationMetadataSchema.code)

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

    def _process_raw_polars_data(self, df: pl.DataFrame, is_all_stations: bool = False) -> pl.DataFrame:
        """Rename columns from source names, cast to schema dtypes, validate, and compute accumulated precip."""
        schema = AllSnotelDataSchema if is_all_stations else SnotelDataSchema
        df = cast_to_schema(df, schema, column_map=STATION_DATA_COLUMN_MAP)

        sort_cols = (
            [AllSnotelDataSchema.station_id, SnotelDataSchema.datetime]
            if is_all_stations
            else [SnotelDataSchema.datetime]
        )
        df = df.sort(sort_cols)

        # The egagli data seems to do precip per day, but SNOTEL naturally uses accumulated precipitation by water year (starting Oct 1)
        water_year_expr = pl.col(SnotelDataSchema.datetime).dt.year() + (
            pl.col(SnotelDataSchema.datetime).dt.month() >= 10
        ).cast(pl.Int32)
        partition_cols = [AllSnotelDataSchema.station_id, "water_year"] if is_all_stations else ["water_year"]

        return (
            df.with_columns(water_year=water_year_expr)
            .with_columns(
                pl.col(SnotelDataSchema.precip_m)
                .fill_null(0.0)
                .cum_sum()
                .over(partition_cols)
                .cast(pl.Float32)
                .alias(SnotelDataSchema.precip_m)
            )
            .drop("water_year")
        )

    def _filter_and_process(self, df: pl.DataFrame, start_date: str | None, end_date: str | None) -> pl.DataFrame:
        """Ensure column naming and apply date filtering."""
        df = self._process_raw_polars_data(df, is_all_stations=False)

        if start_date:
            df = df.filter(pl.col(SnotelDataSchema.datetime) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            df = df.filter(pl.col(SnotelDataSchema.datetime) <= pl.lit(end_date).cast(pl.Date))

        return df

    def get_all_station_data(self, force_update: bool = False) -> pl.DataFrame:
        """Fetch combined daily SNOTEL data for all stations."""
        start_time = time.perf_counter()
        cache_path = self.cache_dir / "all_station_data.parquet"

        if not force_update and self._is_cache_valid(cache_path, STATION_CACHE_DAYS):
            logger.info(f"Retrieving combined data from local cache: {cache_path}")
            res = pl.read_parquet(cache_path)
            logger.info(
                f"Combined data retrieval took {time.perf_counter() - start_time:.2f}s "
                f"(cache hit, {len(res)} rows, {res.estimated_size('mb'):.1f} MB)"
            )
            return res

        logger.info(f"Fetching combined data from internet: {EGAGLI_ALL_STATIONS_TAR_URL}")

        response = requests.get(EGAGLI_ALL_STATIONS_TAR_URL, stream=True)
        response.raise_for_status()

        dfs = self._parse_tar_to_dataframes(response.content)
        combined_df = pl.concat(dfs, how="vertical_relaxed")

        # Add station_id if not already correctly cast
        if AllSnotelDataSchema.station_id in combined_df.columns:
            combined_df = combined_df.with_columns(pl.col(AllSnotelDataSchema.station_id).cast(pl.String))

        combined_df = self._process_raw_polars_data(combined_df, is_all_stations=True)

        combined_df.write_parquet(cache_path)
        logger.info(
            f"Combined data retrieval took {time.perf_counter() - start_time:.2f}s "
            f"(cache miss, {len(combined_df)} rows, {combined_df.estimated_size('mb'):.1f} MB)"
        )
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
                            df = df.with_columns(pl.lit(station_id).alias(AllSnotelDataSchema.station_id))
                            dfs.append(df)
        return dfs
