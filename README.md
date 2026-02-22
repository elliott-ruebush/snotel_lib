# SNOTEL Lib

A Python client for fetching, caching, and validating SNOTEL data.

All credit to [egagli](https://github.com/egagli) for their implementation which backs this repo (see [egagli/snotel_ccss_stations](https://github.com/egagli/snotel_ccss_stations)) and to all the folks/organizations listed in the acknowledgments section of that repo.

More info on SNOTEL (an awesome, cost-effective, and critical network of sensors!): https://www.nrcs.usda.gov/state-offices/nevada/what-is-a-snotel-station

## Installation

Add it to your project using `uv`:

```bash
uv add --editable /path/to/snotel_lib
```

## Quick Start

See [notebooks/snotel_demo.ipynb](notebooks/snotel_demo.ipynb) for a quick demo

## Data Schema

The library standardizes column names:
- `swe_m`: Snow Water Equivalent (meters)
- `snow_depth_m`: Snow Depth (meters)
- `precip_m`: Accumulated Precipitation (meters)
- `tavg_c`, `tmin_c`, `tmax_c`: Temperatures (Celsius)

Validation schemas are defined in `src/snotel_lib/schemas.py`.

## Development

Run tests with:
```bash
uv run pytest
```
