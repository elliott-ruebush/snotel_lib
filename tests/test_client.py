import pytest
import requests

from snotel_lib import SnotelClient


def test_client_init(tmp_path):
    client = SnotelClient(cache_dir=tmp_path)
    assert client.cache_dir == tmp_path


def test_metadata_caching(mocker, tmp_path):
    client = SnotelClient(cache_dir=tmp_path)

    # Mock requests.get
    mock_response = mocker.Mock()
    # A minimal valid geojson for geopandas to read
    mock_response.content = b'{"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"code": "123", "name": "Test", "network": "SNTL", "elevation_m": 1000, "latitude": 45, "longitude": -120, "state": "WA"}, "geometry": {"type": "Point", "coordinates": [-120, 45]}}]}'
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)

    # First call - should hit mock
    metadata = client.get_stations_metadata()
    assert "123" in metadata.index
    assert (tmp_path / "all_stations.parquet").exists()

    assert (
        requests.get.call_count if hasattr(requests.get, "call_count") else getattr(requests.get, "call_count", 1)
    ) == 1


def test_station_data_caching(mocker, tmp_path):
    client = SnotelClient(cache_dir=tmp_path)

    # Mock requests.get
    mock_response = mocker.Mock()
    mock_response.content = b"datetime,WTEQ,PRCPSA,TAVG\n2023-01-01,100,50,0"
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)

    # First call
    df = client.get_station_data("679_WA_SNTL")
    assert df.iloc[0]["swe_m"] == 100
    assert (tmp_path / "679_WA_SNTL.parquet").exists()

    # Second call
    client.get_station_data("679_WA_SNTL")
    assert (
        requests.get.call_count if hasattr(requests.get, "call_count") else getattr(requests.get, "call_count", 1)
    ) == 1


def test_station_data_validation_failure(mocker, tmp_path):
    import pandera.pandas as pa

    client = SnotelClient(cache_dir=tmp_path)

    # Mock requests.get with invalid data (wrong type for swe_m)
    mock_response = mocker.Mock()
    mock_response.content = b"datetime,WTEQ,PRCPSA,TAVG\n2023-01-01,not_a_float,50,0"
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(pa.errors.SchemaError):
        client.get_station_data("INVALID_SNTL")
