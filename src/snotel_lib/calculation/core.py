import datetime as dt

import geopandas as gpd
import polars as pl

from snotel_lib.schemas import AllSnotelDataSchema, SnotelDataSchema, StationMetadataSchema

from ..constants import WATER_YEAR_START_MONTH


def format_rows(
    df: pl.DataFrame,
    metric_col: str,
    round_digits: int = 4,
    extra_cols: list[str] | None = None,
) -> list[dict]:
    res = []
    for r in df.to_dicts():
        row = {
            "station_id": r.get(AllSnotelDataSchema.station_id),
            "name": r.get("station_name", "Unknown"),
            "state": r.get(StationMetadataSchema.state, "Unknown"),
            "elevation_m": r.get(StationMetadataSchema.elevation_m),
            "value": round(r.get(metric_col, 0), round_digits) if r.get(metric_col) is not None else None,
        }
        dt = r.get(SnotelDataSchema.datetime)
        if dt is not None:
            row["data_date"] = str(dt)

        if r.get("is_flagged"):
            row["is_flagged"] = True
            row["qc_flags"] = r.get("qc_flags")
        if extra_cols:
            for ecol in extra_cols:
                if ecol.endswith("_year"):
                    row[ecol] = r.get(ecol)
                else:
                    row[ecol] = round(r.get(ecol, 0), round_digits) if r.get(ecol) is not None else None
        res.append(row)
    return res


def get_top_bot(
    df: pl.DataFrame,
    col: str,
    top_n: int = 10,
    bot_n: int = 5,
    sort_by: str | None = None,
    **kwargs,
) -> dict:
    valid_df = df.drop_nulls([col])
    if sort_by is None:
        sort_by = col

    total = valid_df.height
    top = valid_df.sort(sort_by, descending=True).head(top_n)
    bot = valid_df.sort(sort_by, descending=True).tail(bot_n)
    return {
        "top": format_rows(top, col, **kwargs),
        "bottom": format_rows(bot, col, **kwargs),
        "total_count": total,
    }


def accumulate_precip_by_water_year(df: pl.DataFrame, is_all_stations: bool = False) -> pl.DataFrame:
    """
    SNOTEL naturally uses accumulated precipitation by water year (starting Oct 1).
    This function computes that cumulative sum from daily precipitation values.
    """
    water_year_expr = pl.col(SnotelDataSchema.datetime).dt.year() + (
        pl.col(SnotelDataSchema.datetime).dt.month() >= WATER_YEAR_START_MONTH
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


def get_min_and_max_rows(station_metadata: gpd.GeoDataFrame, column_name: str) -> gpd.GeoDataFrame:
    """Find the rows with the minimum and maximum values in a specific column for stations active in the last 2 days."""
    t_minus_two = dt.date.today() - dt.timedelta(days=2)
    current_stations = station_metadata[
        (station_metadata["end_date"] > t_minus_two) & (station_metadata["end_date"] <= dt.date.today())
    ].index.unique()
    current_stations_metadata = station_metadata.loc[current_stations]
    max_column_idx = current_stations_metadata[column_name].dropna().idxmax()
    min_column_idx = current_stations_metadata[column_name].dropna().idxmin()
    return current_stations_metadata.loc[[max_column_idx, min_column_idx]]
