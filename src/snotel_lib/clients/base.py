import abc
import logging
import typing
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from pandera.typing.geopandas import GeoDataFrame
from pandera.typing.polars import DataFrame

from ..schemas import AllSnotelDataSchema, SnotelDataSchema, StationMetadataSchema

logger = logging.getLogger(__name__)

T = typing.TypeVar("T")


class BaseSnotelClient(abc.ABC):
    """Abstract base class for SNOTEL data clients."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the client, optionally with a custom cache directory."""
        from ..io import get_default_cache_dir

        self.cache_dir = cache_dir or get_default_cache_dir()

    @abc.abstractmethod
    def get_stations_metadata(self, force_update: bool = False) -> GeoDataFrame[StationMetadataSchema]:
        """Fetch metadata for all SNOTEL stations.

        Args:
            force_update: If True, bypass caching and force a fresh download.

        Returns:
            GeoDataFrame of station metadata conforming to StationMetadataSchema.
        """
        pass

    @abc.abstractmethod
    def get_station_data(
        self,
        station_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        force_update: bool = False,
    ) -> DataFrame[SnotelDataSchema]:
        """Fetch daily SNOTEL data for a specific station.

        Args:
            station_id: The unique identifier for the station (e.g., '1000:CO:SNTL').
            start_date: Optional start date filtering (format 'YYYY-MM-DD').
            end_date: Optional end date filtering (format 'YYYY-MM-DD').
            force_update: If True, bypass caching and force a fresh download.

        Returns:
            A Polars DataFrame conforming to SnotelDataSchema.
        """
        pass

    @abc.abstractmethod
    def get_all_station_data(self, force_update: bool = False) -> DataFrame[AllSnotelDataSchema]:
        """Fetch combined daily SNOTEL data for all stations.

        Args:
            force_update: If True, bypass caching and force a fresh download.

        Returns:
            A Polars DataFrame conforming to AllSnotelDataSchema.
        """
        pass

    def _is_cache_valid(self, path: Path, days: int) -> bool:
        """Return True if a cached file exists and is younger than `days` days."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=days)

    def _read_cache_if_valid(
        self,
        cache_path: Path,
        max_days: int,
        force_update: bool,
        read_func: typing.Callable[[Path], T],
        log_label: str,
    ) -> T | None:
        """Read and return cached data if the cache is still valid; otherwise return None.

        Args:
            cache_path: Path to the cached file.
            max_days: Maximum age of the cache in days before it is considered stale.
            force_update: If True, treat the cache as stale regardless of age.
            read_func: Callable that accepts a Path and returns the cached data object.
            log_label: Human-readable label used in the cache-hit log message.

        Returns:
            The cached data if valid, or None on a cache miss.
        """
        if force_update or not self._is_cache_valid(cache_path, max_days):
            return None
        logger.info(f"Cache hit — reading {log_label} from {cache_path}")
        return read_func(cache_path)

    def _filter_and_process(self, df: pl.DataFrame, start_date: str | None, end_date: str | None) -> pl.DataFrame:
        """Apply date filtering to an already-processed dataframe."""
        if start_date:
            df = df.filter(pl.col(SnotelDataSchema.datetime) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            df = df.filter(pl.col(SnotelDataSchema.datetime) <= pl.lit(end_date).cast(pl.Date))

        return df
