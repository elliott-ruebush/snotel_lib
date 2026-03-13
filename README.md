# SNOTEL Lib (WIP)

A Python client for fetching, caching, and validating SNOTEL data.

Currently used to power:
- SNOTEL Leaderboard: https://elliott-ruebush.github.io/snotel_leaderboard/ (GitHub: https://github.com/elliott-ruebush/snotel_leaderboard)
- More fun projects coming soon...?

All credit to [egagli](https://github.com/egagli) for their data fetching implementation which backs this repo (see [egagli/snotel_ccss_stations](https://github.com/egagli/snotel_ccss_stations)) and to all the folks/organizations listed in the acknowledgments section there.

More info on SNOTEL (an awesome, cost-effective, and critical network of sensors!):
- https://www.nrcs.usda.gov/state-offices/nevada/what-is-a-snotel-station
- https://www.drought.gov/data-maps-tools/snow-telemetry-snotel-snow-depth-and-snow-water-equivalent-products

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

Run a suite of pre-commit checks for formatting/linting/type checking with
```bash
uv run prek --all-files
```
