import io
import json
import lzma
import tarfile

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
    # A minimal valid geojson for geopandas to read, including all schema columns
    geojson_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "code": "123",
                    "name": "Test",
                    "network": "SNTL",
                    "elevation_m": 1000,
                    "latitude": 45,
                    "longitude": -120,
                    "state": "WA",
                    "HUC": "12345",
                    "mgrs": "ABC",
                    "mountainRange": "Rainier",
                    "beginDate": "1980-01-01",
                    "endDate": "2023-01-01",
                    "csvData": True,
                },
                "geometry": {"type": "Point", "coordinates": [-120, 45]},
            }
        ],
    }
    mock_response.content = json.dumps(geojson_content).encode()
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)

    # First call - should hit mock
    metadata = client.get_stations_metadata()
    assert "123" in metadata.index
    assert "mountain_range" in metadata.columns
    assert metadata.loc["123", "mountain_range"] == "Rainier"
    assert (tmp_path / "all_stations.parquet").exists()

    assert (
        requests.get.call_count if hasattr(requests.get, "call_count") else getattr(requests.get, "call_count", 1)
    ) == 1


def test_station_data_caching(mocker, tmp_path):
    client = SnotelClient(cache_dir=tmp_path)

    # Mock requests.get
    mock_response = mocker.Mock()
    # Now that we have a stricter schema, we must include all columns or accept NaNs if allowed
    # Our schema specifies swe_m, snow_depth_m, precip_m, tavg_c, tmin_c, tmax_c are required but nullable.
    # However, Pandera's DataFrameModel requires the column to exist if typed as Series[float].
    header = "datetime,WTEQ,SNWD,PRCPSA,TAVG,TMIN,TMAX"
    row = "2023-01-01,100,50,0,1,2,3"
    mock_response.content = f"{header}\n{row}".encode()
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)

    # First call
    df = client.get_station_data("679_WA_SNTL")
    assert df.select("swe_m").item(0, 0) == 100
    assert "snow_depth_m" in df.columns
    assert (tmp_path / "679_WA_SNTL.parquet").exists()

    # Second call
    client.get_station_data("679_WA_SNTL")
    assert (
        requests.get.call_count if hasattr(requests.get, "call_count") else getattr(requests.get, "call_count", 1)
    ) == 1

    # Test filtering
    df_filtered = client.get_station_data("679_WA_SNTL", start_date="2023-01-01", end_date="2023-01-01")
    assert len(df_filtered) == 1


def test_station_data_validation_failure(mocker, tmp_path):
    import polars.exceptions as ple

    client = SnotelClient(cache_dir=tmp_path)

    # Mock requests.get with invalid data (wrong type for swe_m)
    mock_response = mocker.Mock()
    header = "datetime,WTEQ,SNWD,PRCPSA,TAVG,TMIN,TMAX"
    row = "2023-01-01,not_a_float,50,0,1,2,3"
    mock_response.content = f"{header}\n{row}".encode()
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises((ple.SchemaError, ple.ComputeError, ple.PolarsError)):
        client.get_station_data("INVALID_SNTL")


def test_get_all_station_data(mocker, tmp_path):
    client = SnotelClient(cache_dir=tmp_path)

    # Create a small tar.lzma in memory
    buf = io.BytesIO()
    with lzma.open(buf, "wb") as lzma_file:
        with tarfile.open(fileobj=lzma_file, mode="w:") as tar:
            csv_content = b"datetime,WTEQ,SNWD\n2023-01-01,100,50"
            tarinfo = tarfile.TarInfo(name="679_WA_SNTL.csv")
            tarinfo.size = len(csv_content)
            tar.addfile(tarinfo, io.BytesIO(csv_content))

    mock_response = mocker.Mock()
    mock_response.content = buf.getvalue()
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)

    df = client.get_all_station_data()
    assert "station_id" in df.columns
    assert df.select("station_id").item(0, 0) == "679_WA_SNTL"
    assert df.select("swe_m").item(0, 0) == 100
    assert (tmp_path / "all_station_data.parquet").exists()
