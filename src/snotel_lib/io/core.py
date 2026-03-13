from __future__ import annotations

from pathlib import Path

import pandera.polars as pl_pa
import polars as pl
from platformdirs import user_cache_dir

from ..schemas import cast_to_schema

DEFAULT_CACHE_DIR = Path(user_cache_dir(appname="snotel_data"))


def get_default_cache_dir() -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR


def get_metadata_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "all_stations.parquet"


def get_all_station_data_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "all_station_data.parquet"


def get_egagli_station_cache_path(cache_dir: Path, station_id: str) -> Path:
    return cache_dir / f"{station_id}.parquet"


def get_metloom_station_cache_path(cache_dir: Path, station_id: str) -> Path:
    safe_station_id = station_id.replace(":", "_")
    return cache_dir / f"metloom_{safe_station_id}.parquet"


def read_validated_csv(
    source: str | Path | bytes,
    schema: type[pl_pa.DataFrameModel],
    column_map: dict[str, str] | None = None,
    **pl_read_kwargs,
) -> pl.DataFrame:
    """
    Read a CSV source and return a schema-validated, correctly-typed DataFrame.

    Args:
        source: File path, URL, or raw bytes understood by ``pl.read_csv``.
        schema: Pandera ``DataFrameModel`` to validate and cast against.
        column_map: Optional rename map applied before casting (see ``cast_to_schema``).
        **pl_read_kwargs: Extra keyword arguments forwarded to ``pl.read_csv``.

    Returns:
        A validated ``pl.DataFrame`` conforming to *schema*.
    """
    df = pl.read_csv(source, **pl_read_kwargs)
    return cast_to_schema(df, schema, column_map=column_map)


def read_validated_parquet(
    source: str | Path,
    schema: type[pl_pa.DataFrameModel],
    column_map: dict[str, str] | None = None,
    **pl_read_kwargs,
) -> pl.DataFrame:
    """
    Read a Parquet file and return a schema-validated, correctly-typed DataFrame.

    Args:
        source: File path understood by ``pl.read_parquet``.
        schema: Pandera ``DataFrameModel`` to validate and cast against.
        column_map: Optional rename map applied before casting (see ``cast_to_schema``).
        **pl_read_kwargs: Extra keyword arguments forwarded to ``pl.read_parquet``.

    Returns:
        A validated ``pl.DataFrame`` conforming to *schema*.
    """
    df = pl.read_parquet(source, **pl_read_kwargs)
    return cast_to_schema(df, schema, column_map=column_map)
