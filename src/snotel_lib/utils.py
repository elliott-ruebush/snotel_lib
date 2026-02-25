import datetime as dt

import geopandas as gpd


def get_min_and_max_rows(station_metadata: gpd.GeoDataFrame, column_name: str) -> gpd.GeoDataFrame:
    t_minus_two = dt.date.today() - dt.timedelta(days=2)
    current_stations = station_metadata[
        (station_metadata["end_date"] > t_minus_two) & (station_metadata["end_date"] <= dt.date.today())
    ].index.unique()
    current_stations_metadata = station_metadata.loc[current_stations]
    max_column_idx = current_stations_metadata[column_name].dropna().idxmax()
    min_column_idx = current_stations_metadata[column_name].dropna().idxmin()
    return current_stations_metadata.loc[[max_column_idx, min_column_idx]]
