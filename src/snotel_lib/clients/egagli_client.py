import io
import logging
import lzma
import tarfile
import time
import typing
from pathlib import Path

import geopandas as gpd
import polars as pl
import requests
from pandera.typing.geopandas import GeoDataFrame
from pandera.typing.polars import DataFrame

from ..calculation import accumulate_precip_by_water_year
from ..constants import METADATA_CACHE_DAYS, STATION_CACHE_DAYS
from ..io import (
    get_all_station_data_cache_path,
    get_default_cache_dir,
    get_egagli_station_cache_path,
    get_metadata_cache_path,
)
from ..schemas import (
    AllSnotelDataSchema,
    SnotelDataSchema,
    StationMetadataSchema,
    cast_to_schema,
)
from .base import BaseSnotelClient

logger = logging.getLogger(__name__)

EGAGLI_GEOJSON_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson"
EGAGLI_STATION_CSV_BASE = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station_id}.csv"
EGAGLI_ALL_STATIONS_TAR_URL = (
    "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/all_station_data.tar.lzma"
)

METADATA_COLUMN_MAP = {
    "code": StationMetadataSchema.station_id,
    "name": StationMetadataSchema.station_name,
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
        self.cache_dir = cache_dir or get_default_cache_dir()

    def get_stations_metadata(self, force_update: bool = False) -> GeoDataFrame[StationMetadataSchema]:
        """Fetch metadata for all SNOTEL stations, with caching."""
        start_time = time.perf_counter()
        cache_path = get_metadata_cache_path(self.cache_dir)

        cached = self._read_cache_if_valid(cache_path, METADATA_CACHE_DAYS, force_update, gpd.read_parquet, "metadata")
        if cached is not None:
            res = typing.cast(GeoDataFrame[StationMetadataSchema], StationMetadataSchema.validate(cached))
            logger.info(
                f"Metadata retrieval took {time.perf_counter() - start_time:.2f}s (cache hit, {len(res)} stations)"
            )
            return res

        res = self._fetch_and_cache_metadata(cache_path)
        logger.info(
            f"Metadata retrieval took {time.perf_counter() - start_time:.2f}s (cache miss, {len(res)} stations)"
        )
        return res

    def get_station_data(
        self,
        station_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        force_update: bool = False,
    ) -> DataFrame[SnotelDataSchema]:
        """Fetch daily SNOTEL data for a specific station, with caching."""
        start_time = time.perf_counter()
        cache_path = get_egagli_station_cache_path(self.cache_dir, station_id)

        cached = self._read_cache_if_valid(
            cache_path, STATION_CACHE_DAYS, force_update, pl.read_parquet, f"station data for {station_id}"
        )
        if cached is not None:
            res = self._filter_and_process(cached, start_date, end_date)
            logger.info(
                f"Data retrieval for {station_id} took {time.perf_counter() - start_time:.2f}s (cache hit, {len(res)} rows)"
            )
            return typing.cast(DataFrame[SnotelDataSchema], res)

        res = self._fetch_and_cache_station_data(station_id, cache_path, start_date, end_date)
        logger.info(
            f"Data retrieval for {station_id} took {time.perf_counter() - start_time:.2f}s (cache miss, {len(res)} rows)"
        )
        return typing.cast(DataFrame[SnotelDataSchema], res)

    def get_all_station_data(self, force_update: bool = False) -> DataFrame[AllSnotelDataSchema]:
        """Fetch combined daily SNOTEL data for all stations."""
        start_time = time.perf_counter()
        cache_path = get_all_station_data_cache_path(self.cache_dir)

        cached = self._read_cache_if_valid(
            cache_path, STATION_CACHE_DAYS, force_update, pl.read_parquet, "combined station data"
        )
        if cached is not None:
            logger.info(
                f"Combined data retrieval took {time.perf_counter() - start_time:.2f}s (cache hit, {len(cached)} rows, {cached.estimated_size('mb'):.1f} MB)"
            )
            return typing.cast(DataFrame[AllSnotelDataSchema], cached)

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
        return typing.cast(DataFrame[AllSnotelDataSchema], combined_df)

    def _fetch_and_cache_metadata(self, cache_path: Path) -> GeoDataFrame[StationMetadataSchema]:
        logger.info(f"Fetching metadata from internet: {EGAGLI_GEOJSON_URL}")
        response = requests.get(EGAGLI_GEOJSON_URL)
        response.raise_for_status()

        df = gpd.read_file(io.BytesIO(response.content))
        df = df.rename(columns=METADATA_COLUMN_MAP)

        validated_df = typing.cast(GeoDataFrame[StationMetadataSchema], StationMetadataSchema.validate(df))
        validated_df.to_parquet(cache_path)
        return validated_df

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

    def _process_raw_polars_data(
        self, df: pl.DataFrame, is_all_stations: bool = False
    ) -> pl.DataFrame:  # returns AllSnotelDataSchema when is_all_stations=True
        """Rename columns from source names, cast to schema dtypes, validate, and compute accumulated precip."""
        schema = AllSnotelDataSchema if is_all_stations else SnotelDataSchema
        df = cast_to_schema(df, schema, column_map=STATION_DATA_COLUMN_MAP)

        sort_cols = (
            [AllSnotelDataSchema.station_id, SnotelDataSchema.datetime]
            if is_all_stations
            else [SnotelDataSchema.datetime]
        )
        df = df.sort(sort_cols)
        return accumulate_precip_by_water_year(df, is_all_stations=is_all_stations)

    def _filter_and_process(self, df: pl.DataFrame, start_date: str | None, end_date: str | None) -> pl.DataFrame:
        """Ensure column naming and apply date filtering."""
        df = self._process_raw_polars_data(df, is_all_stations=False)
        return super()._filter_and_process(df, start_date, end_date)

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
