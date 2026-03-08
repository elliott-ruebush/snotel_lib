from dataclasses import dataclass
from typing import Literal

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, Series


class QCLogSchema(pa.DataFrameModel):
    station_id: Series[pl.String] = pa.Field()
    datetime: Series[pl.Date] = pa.Field()
    variable: Series[pl.String] = pa.Field()
    qc_type: Series[pl.String] = pa.Field()
    reason: Series[pl.String] = pa.Field()
    explanation: Series[pl.String] = pa.Field()


@dataclass(frozen=True)
class QCResult:
    """Result of running QC checks containing the cleaned data and discrete logs."""

    data: pl.DataFrame
    qc: DataFrame[QCLogSchema]


@dataclass(frozen=True)
class QCCheck:
    """A quality control check definition to flag or filter anomalies in data."""

    variable: str
    qc_type: Literal["FILTER", "FLAG"]
    reason: str
    expr: pl.Expr
    explanation: str = ""
