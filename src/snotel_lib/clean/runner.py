import datetime

import polars as pl

from .models import QCCheck, QCLogSchema, QCResult


def run_qc(df: pl.DataFrame, station_id: str, checks: list[QCCheck]) -> QCResult:
    """Run a sequence of quality control checks over a DataFrame."""
    qc_logs: list[pl.DataFrame] = []
    filter_exprs: list[pl.Expr] = []

    # Get expected schema struct
    log_columns = list(QCLogSchema.to_schema().columns.keys())

    for check in checks:
        # evaluate the expression
        issue_df = df.filter(check.expr)

        # If any rows were found by this check, generate the logs
        if len(issue_df) > 0:
            # Use existing station_id column if present (multi-station processing)
            # otherwise fall back to the provided station_id string
            output_cols = (
                [QCLogSchema.datetime, QCLogSchema.station_id]
                if QCLogSchema.station_id in issue_df.columns
                else [QCLogSchema.datetime]
            )

            log_df = (
                issue_df.select(output_cols)
                .with_columns(
                    pl.lit(station_id).cast(pl.String).alias(QCLogSchema.station_id)
                    if QCLogSchema.station_id not in issue_df.columns
                    else pl.col(QCLogSchema.station_id),
                    pl.lit(check.variable).cast(pl.String).alias(QCLogSchema.variable),
                    pl.lit(check.qc_type).cast(pl.String).alias(QCLogSchema.qc_type),
                    pl.lit(check.reason).cast(pl.String).alias(QCLogSchema.reason),
                    pl.lit(check.explanation).cast(pl.String).alias(QCLogSchema.explanation),
                )
                .select(pl.col(log_columns))
            )
            qc_logs.append(log_df)

        if check.qc_type == "FILTER":
            filter_exprs.append(check.expr)

    # Accumulate filters and remove corresponding rows from pure data
    clean_df = df
    if filter_exprs:
        combined_filter = filter_exprs[0]
        for expr in filter_exprs[1:]:
            combined_filter = combined_filter | expr
        clean_df = df.filter(~combined_filter)

    # Combine all QC logs
    if qc_logs:
        qc_df = pl.concat(qc_logs)
    else:
        # Empty DataFrame with string/date schema
        # Workaround because pandera DataFrameModel.to_schema().columns dtype doesn't have a direct Polars translate sometimes.
        # But we know they're Int, String, etc.
        # Instead, just construct one with dummy values and filter it.
        dummy_df = pl.DataFrame(
            {
                QCLogSchema.station_id: [""],
                QCLogSchema.datetime: [datetime.date.today()],
                QCLogSchema.variable: [""],
                QCLogSchema.qc_type: [""],
                QCLogSchema.reason: [""],
                QCLogSchema.explanation: [""],
            }
        ).clear()
        qc_df = dummy_df

    validated_qc_df = QCLogSchema.validate(qc_df)
    return QCResult(data=clean_df, qc=validated_qc_df)  # type: ignore[arg-type]
