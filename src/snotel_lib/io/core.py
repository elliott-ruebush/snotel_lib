"""
Schema-aware IO utilities for snotel_lib.

Provides generic helpers for extracting Polars dtypes from pandera DataFrameModels,
casting DataFrames to a schema, and reading validated CSV/Parquet files.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pandera.polars as pl_pa
import polars as pl


def dtypes_from_schema(schema: type[pl_pa.DataFrameModel]) -> dict[str, pl.DataType]:
    """
    Extract a {column_name: polars_dtype} mapping from a pandera Polars DataFrameModel.

    Handles both bare dtype values (e.g. pl.Float32) and the generic
    Series[pl.Float32] annotation form used by pandera.
    """
    result: dict[str, pl.DataType] = {}
    for col, annotation in schema.__annotations__.items():
        dtype = _extract_pl_dtype(annotation)
        if dtype is not None:
            result[col] = dtype
    return result


def cast_to_schema(
    df: pl.DataFrame,
    schema: type[pl_pa.DataFrameModel],
    column_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """
    Optionally rename columns via *column_map*, cast to the dtypes declared in
    *schema*, then pandera-validate and return the result.

    Args:
        df: Input DataFrame with raw column names / dtypes.
        schema: A pandera ``DataFrameModel`` subclass describing the desired output.
        column_map: Optional ``{source_col: target_col}`` rename map applied before
            casting.  Only columns present in ``df`` are renamed.

    Returns:
        A validated ``pl.DataFrame`` conforming to *schema*.
    """
    if column_map:
        rename = {k: v for k, v in column_map.items() if k in df.columns}
        if rename:
            df = df.rename(rename)

    dtypes = dtypes_from_schema(schema)
    cast_exprs = [pl.col(col).cast(dtype) for col, dtype in dtypes.items() if col in df.columns]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return typing.cast(pl.DataFrame, schema.validate(df))


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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_pl_dtype(annotation) -> pl.DataType | None:
    """
    Safely extract a Polars DataType from a pandera annotation.

    Handles:
    - Bare instances: ``pl.Float32``
    - Bare classes: ``pl.Float32`` (the class itself, not an instance)
    - Generic aliases: ``Series[pl.Float32]``
    """
    # Bare instance (e.g. pl.Float32() )
    if isinstance(annotation, pl.DataType):
        return annotation

    # Bare class (e.g. pl.Float32 the class)
    try:
        if issubclass(annotation, pl.DataType):
            return annotation()
    except TypeError:
        pass

    # Generic alias like Series[pl.Float32]
    args = getattr(annotation, "__args__", None)
    if args:
        for arg in args:
            dtype = _extract_pl_dtype(arg)
            if dtype is not None:
                return dtype

    return None
