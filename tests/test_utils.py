import datetime as dt

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from snotel_lib.utils import get_min_and_max_rows


def test_get_min_and_max_rows():
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    data = {
        "code": ["S1", "S2", "S3"],
        "name": ["Station 1", "Station 2", "Station 3"],
        "end_date": [today, yesterday, today - dt.timedelta(days=5)],
        "val": [10.0, 20.0, 5.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
    }
    gdf = gpd.GeoDataFrame(data).set_index("code")
    gdf["end_date"] = pd.to_datetime(gdf["end_date"]).dt.date

    result = get_min_and_max_rows(gdf, "val")  # ty: ignore[invalid-argument-type]

    # Only S1 and S2 are within last 2 days
    assert len(result) == 2
    assert "S2" in result.index  # Max
    assert "S1" in result.index  # Min
    assert result.loc["S2", "val"] == 20.0
    assert result.loc["S1", "val"] == 10.0
