import typing
from dataclasses import dataclass, field

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, Series


class QCLogSchema(pa.DataFrameModel):
    station_id: Series[pl.String] = pa.Field()
    datetime: Series[pl.Date] = pa.Field()
    name: Series[pl.String] = pa.Field()
    explanation: Series[pl.String] = pa.Field()


@dataclass(frozen=True)
class QCResult:
    """Result of running QC checks containing the cleaned data and discrete logs."""

    data: pl.DataFrame
    filter_log: DataFrame[QCLogSchema]
    flag_log: DataFrame[QCLogSchema]


@dataclass(frozen=True)
class FilterCheck:
    """A quality control check that filters out rows from the data."""

    name: str
    expr: pl.Expr
    explanation: str = ""

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Apply the filter and return (cleaned_df, log)."""
        issue_df = df.filter(self.expr)
        log = issue_df.select(["station_id", "datetime"]).with_columns(
            pl.lit(self.name).alias("name"),
            pl.lit(self.explanation).alias("explanation"),
        )
        return df.filter(~self.expr), log.select(list(QCLogSchema.to_schema().columns.keys()))


@dataclass(frozen=True)
class FlagCheck:
    """A quality control check that flags rows in the data without removing them."""

    name: str
    expr: pl.Expr
    explanation: str = ""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the flag check and return the log."""
        issue_df = df.filter(self.expr)
        log = issue_df.select(["station_id", "datetime"]).with_columns(
            pl.lit(self.name).alias("name"),
            pl.lit(self.explanation).alias("explanation"),
        )
        return log.select(list(QCLogSchema.to_schema().columns.keys()))


@dataclass(frozen=True)
class FilterList:
    """Ordered container for FilterChecks."""

    checks: list[FilterCheck] = field(default_factory=list)

    def __add__(self, other: "FilterList") -> "FilterList":
        if not isinstance(other, FilterList):
            return NotImplemented
        return FilterList(self.checks + other.checks)

    def exclude(self, check: FilterCheck) -> "FilterList":
        """Exclude a check by identity."""
        return FilterList([c for c in self.checks if c is not check])

    def __iter__(self) -> typing.Iterator[FilterCheck]:
        return iter(self.checks)


@dataclass(frozen=True)
class FlagList:
    """Ordered container for FlagChecks."""

    checks: list[FlagCheck] = field(default_factory=list)

    def __add__(self, other: "FlagList") -> "FlagList":
        if not isinstance(other, FlagList):
            return NotImplemented
        return FlagList(self.checks + other.checks)

    def exclude(self, check: FlagCheck) -> "FlagList":
        """Exclude a check by identity."""
        return FlagList([c for c in self.checks if c is not check])

    def __iter__(self) -> typing.Iterator[FlagCheck]:
        return iter(self.checks)
