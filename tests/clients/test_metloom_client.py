import pandas as pd
import pytest

from snotel_lib.clients import MetloomClient
from snotel_lib.schemas import SnotelDataSchema, StationMetadataSchema


def test_metloom_client_init(tmp_path):
    client = MetloomClient(cache_dir=tmp_path)
    assert client.cache_dir == tmp_path


def test_metadata_caching(mocker, tmp_path):
    client = MetloomClient(cache_dir=tmp_path)

    # Needs to match Egagli GeoDataFrame format
    import geopandas as gpd
    from shapely.geometry import Point

    mock_df = pd.DataFrame(
        {
            StationMetadataSchema.station_id: ["123"],
            StationMetadataSchema.station_name: ["Test Station"],
            StationMetadataSchema.network: ["SNTL"],
            StationMetadataSchema.elevation_m: [1000.0],
            StationMetadataSchema.state: ["CO"],
            StationMetadataSchema.huc: ["TEST_HUC"],
            StationMetadataSchema.mountain_range: ["Rainier"],
            StationMetadataSchema.latitude: [0.0],
            StationMetadataSchema.longitude: [0.0],
        }
    )

    gdf = gpd.GeoDataFrame(mock_df, geometry=[Point(0, 0)], crs="EPSG:4326")

    # Mock the underlying egagli client method to simply return the perfectly formatted metadata
    mocker.patch("snotel_lib.clients.egagli_client.EgagliClient.get_stations_metadata", return_value=gdf)

    metadata = client.get_stations_metadata()
    assert "123" in metadata[StationMetadataSchema.station_id].values


def test_station_data_caching(mocker, tmp_path):
    client = MetloomClient(cache_dir=tmp_path)

    import geopandas as gpd
    from shapely.geometry import Point

    # Mock SnotelPointData.get_daily_data
    mock_df = pd.DataFrame(
        {
            "SWE": [10.0],  # inches
            "SNOWDEPTH": [50.0],  # inches
            "PRECIPITATION": [15.0],  # inches
            "AVG AIR TEMP": [32.0],  # fahrenheit
            "SWE_units": ["in"],
            "datasource": ["NRCS"],
            # Ensure we test the missing columns logic as well (MIN AIR TEMP and MAX AIR TEMP aren't here)
        }
    )
    mock_df["datetime"] = pd.to_datetime(["2023-01-01"]).tz_localize("UTC")
    mock_df["site"] = ["679:WA:SNTL"]
    mock_df = mock_df.set_index(["datetime", "site"])

    mock_gdf = gpd.GeoDataFrame(mock_df, geometry=[Point(0, 0)], crs="EPSG:4326")

    mock_snotel_point = mocker.Mock()
    mock_snotel_point.get_daily_data.return_value = mock_gdf
    mocker.patch("snotel_lib.clients.metloom_client.SnotelPointData", return_value=mock_snotel_point)

    # First call
    df = client.get_station_data("679:WA:SNTL")

    # 10 inches * 0.0254 = 0.254 meters
    assert df.select(SnotelDataSchema.swe_m).item(0, 0) == pytest.approx(0.254)
    # 32F = 0C
    assert df.select(SnotelDataSchema.tavg_c).item(0, 0) == 0.0
    # Missing columns should have been added as nulls
    assert df.select(SnotelDataSchema.tmin_c).item(0, 0) is None

    assert (tmp_path / "metloom_679_WA_SNTL.parquet").exists()


def test_get_all_station_data_unsupported():
    client = MetloomClient()
    with pytest.raises(NotImplementedError):
        client.get_all_station_data()
