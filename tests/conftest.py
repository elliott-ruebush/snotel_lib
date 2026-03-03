import pytest


@pytest.fixture
def mock_geojson():
    return {
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


@pytest.fixture
def mock_station_csv():
    header = "datetime,WTEQ,SNWD,PRCPSA,TAVG,TMIN,TMAX"
    row = "2023-01-01,100,50,0,1,2,3"
    return f"{header}\n{row}"
