"""Tests for snotel_lib.io."""

from datetime import date

import pandera.polars as pl_pa
import polars as pl
import pytest
from pandera.typing import Series

from snotel_lib.io import (
    cast_to_schema,
    dtypes_from_schema,
    read_validated_csv,
    read_validated_parquet,
)
from snotel_lib.schemas import SnotelDataSchema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_df() -> pl.DataFrame:
    """A minimal raw DataFrame matching SnotelDataSchema columns."""
    return pl.DataFrame(
        {
            "datetime": [date(2024, 1, 1), date(2024, 1, 2)],
            "swe_m": [0.1, 0.2],
            "snow_depth_m": [0.5, 0.6],
            "precip_m": [None, 0.01],
            "tavg_c": [-1.0, 0.0],
            "tmin_c": [-3.0, -1.0],
            "tmax_c": [1.0, 2.0],
        }
    )


def _raw_csv_bytes() -> bytes:
    lines = [
        "datetime,swe_m,snow_depth_m,precip_m,tavg_c,tmin_c,tmax_c",
        "2024-01-01,0.1,0.5,,-1.0,-3.0,1.0",
        "2024-01-02,0.2,0.6,0.01,0.0,-1.0,2.0",
    ]
    return "\n".join(lines).encode()


# ---------------------------------------------------------------------------
# dtypes_from_schema
# ---------------------------------------------------------------------------


class TestDtypesFromSchema:
    def test_extracts_all_snotel_fields(self):
        dtypes = dtypes_from_schema(SnotelDataSchema)
        assert "datetime" in dtypes
        assert "swe_m" in dtypes
        assert "snow_depth_m" in dtypes

    def test_correct_dtype_instances(self):
        dtypes = dtypes_from_schema(SnotelDataSchema)
        assert isinstance(dtypes["swe_m"], pl.Float32)
        assert isinstance(dtypes["datetime"], pl.Date)

    def test_custom_schema(self):
        class MySchema(pl_pa.DataFrameModel):
            id: Series[pl.Int64] = pl_pa.Field()
            value: Series[pl.Float64] = pl_pa.Field(nullable=True)

        dtypes = dtypes_from_schema(MySchema)
        assert isinstance(dtypes["id"], pl.Int64)
        assert isinstance(dtypes["value"], pl.Float64)


# ---------------------------------------------------------------------------
# cast_to_schema
# ---------------------------------------------------------------------------


class TestCastToSchema:
    def test_basic_cast_validates(self):
        raw = _make_raw_df()
        result = cast_to_schema(raw, SnotelDataSchema)
        assert result.schema["swe_m"] == pl.Float32

    def test_rename_then_cast(self):
        """column_map is applied before casting."""
        raw = pl.DataFrame(
            {
                "WTEQ": [0.1, 0.2],
                "datetime": [date(2024, 1, 1), date(2024, 1, 2)],
            }
        )
        column_map = {"WTEQ": "swe_m"}

        class TinySchema(pl_pa.DataFrameModel):
            datetime: Series[pl.Date] = pl_pa.Field()
            swe_m: Series[pl.Float32] = pl_pa.Field(nullable=True)

            class Config:
                strict = False
                coerce = True

        result = cast_to_schema(raw, TinySchema, column_map=column_map)
        assert "swe_m" in result.columns
        assert "WTEQ" not in result.columns
        assert result.schema["swe_m"] == pl.Float32

    def test_schema_violation_raises(self):
        """A DataFrame missing a required non-nullable column should fail validation."""

        class StrictSchema(pl_pa.DataFrameModel):
            required_col: Series[pl.String] = pl_pa.Field()

            class Config:
                strict = True
                coerce = True

        raw = pl.DataFrame({"other_col": ["a", "b"]})
        with pytest.raises((Exception, pl_pa.errors.SchemaError)):
            cast_to_schema(raw, StrictSchema)


# ---------------------------------------------------------------------------
# read_validated_csv
# ---------------------------------------------------------------------------


class TestReadValidatedCsv:
    def test_round_trip(self, tmp_path):
        raw = _make_raw_df()
        csv_path = tmp_path / "test.csv"
        raw.write_csv(csv_path)

        result = read_validated_csv(
            csv_path,
            SnotelDataSchema,
            try_parse_dates=True,
            null_values=["", "NaN"],
        )
        assert result.height == 2
        assert result.schema["swe_m"] == pl.Float32


# ---------------------------------------------------------------------------
# read_validated_parquet
# ---------------------------------------------------------------------------


class TestReadValidatedParquet:
    def test_round_trip(self, tmp_path):
        raw = _make_raw_df()
        # Write with correct types so parquet stores them accurately
        typed = raw.with_columns(
            [
                pl.col("swe_m").cast(pl.Float32),
                pl.col("snow_depth_m").cast(pl.Float32),
                pl.col("precip_m").cast(pl.Float32),
                pl.col("tavg_c").cast(pl.Float32),
                pl.col("tmin_c").cast(pl.Float32),
                pl.col("tmax_c").cast(pl.Float32),
            ]
        )
        pq_path = tmp_path / "test.parquet"
        typed.write_parquet(pq_path)

        result = read_validated_parquet(pq_path, SnotelDataSchema)
        assert result.height == 2
        assert result.schema["swe_m"] == pl.Float32
