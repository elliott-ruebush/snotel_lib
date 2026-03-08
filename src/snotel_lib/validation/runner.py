import polars as pl

from .models import FilterList, FlagList, QCLogSchema, QCResult


def run_qc(df: pl.DataFrame, filters: FilterList, flags: FlagList) -> QCResult:
    """Run a sequence of quality control checks over a DataFrame."""
    filter_logs: list[pl.DataFrame] = []

    # TODO: Ensure data is sorted by station and date for window functions (diffs)
    # Currently frontend generate does this as before passing in the dataframe.
    # Ideally we would avoid adding the specific snotel logic into run_qc
    clean_df = df.sort(["station_id", "datetime"])
    for f_check in filters:
        clean_df, log = f_check.apply(clean_df)
        if len(log) > 0:
            filter_logs.append(log)

    flag_logs: list[pl.DataFrame] = []
    for fl_check in flags:
        log = fl_check.apply(clean_df)
        if len(log) > 0:
            flag_logs.append(log)

    filter_qc_df = pl.concat(filter_logs) if filter_logs else QCLogSchema.empty()
    flag_qc_df = pl.concat(flag_logs) if flag_logs else QCLogSchema.empty()

    return QCResult(
        data=clean_df,
        filter_log=QCLogSchema.validate(filter_qc_df),
        flag_log=QCLogSchema.validate(flag_qc_df),
    )
