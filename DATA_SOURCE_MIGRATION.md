From Gemini 3.1 Pro LLM model

# Data Source Migration Exploration

This document tracks thoughts and research around potentially migrating the underlying data ingestion layer in `snotel_lib` away from the current `egagli/snotel_ccss_stations` repository and towards a more native or professionally maintained API.

## Motivation for Migration
The current GitHub repository source is excellent for prototyping, but has a few limitations that make a migration appealing for long-term robustness:
1. **Data Quirks:** As discovered, the existing source pre-processes the precipitation values (`PRCPSA`) into daily differential amounts rather than maintaining the native SNOTEL Accumulated Precipitation format. (We currently work around this in `snotel_lib` by performing cumulative sums partitioned by water year, but native data is preferred).
2. **Frequency/Liveliness:** A GitHub Actions cron-scraped dataset will always be delayed relative to hitting an active API. For true "live leaderboards," a direct connection is required.
3. **Multi-Network Expansion:** If we want to evaluate Canadian datasets (e.g., BC River Forecast Centre) or California specific Department of Water Resources (CDEC) snow sensors, we need a library or source that abstracts these.

## Potential Upstream Data Sources/Libraries

### 1. Metloom
* **Repository:** [https://github.com/M3Works/metloom](https://github.com/M3Works/metloom)
* **Overview:** A Python library specifically built to abstract point-measurement data from various networks (NRCS SNOTEL, CDEC, MesoWest).
* **Pros:**
  * Actively maintained and built exactly for this use case.
  * Handles the SNOTEL (NRCS AWDB) SOAP API natively.
  * Extensible (can pull CDEC for California sensors right out of the box).
  * Returns GeoDataFrames making spatial analysis trivial.
* **Cons:**
  * Adds a heavier dependency footprint.

### 2. Synoptic API
* **Website:** [https://synopticdata.com/weatherapi/](https://synopticdata.com/weatherapi/)
* **Overview:** A commercial (but with free tiers/academic access) API that aggregates thousands of weather and mesonets, including SNOTEL.
* **Pros:** Exceptionally fast, standardized JSON outputs natively, highly reliable.
* **Cons:** Requires dealing with API keys, rate limits, and potentially paid tiers if usage scales significantly.

### 3. Direct NRCS AWDB API (National Water and Climate Center)
* **Website:** [https://www.nrcs.usda.gov/wps/portal/wcc/home/dataAccessHelp/webService/](https://www.nrcs.usda.gov/wps/portal/wcc/home/dataAccessHelp/webService/)
* **Overview:** The canonical upstream source for all SNOTEL data.
* **Pros:** The source of truth. No middle-man interpretation of the data. Free to use.
* **Cons:** It is a legacy SOAP API, which is notoriously frustrating to interact with in modern Python stacks compared to REST/JSON. `zeep` is typically required. (This is exactly what `metloom` abstracts away).

### 4. ulmo (Deprecated)
* **Repository:** [https://github.com/ulmo-dev/ulmo](https://github.com/ulmo-dev/ulmo)
* **Note:** Formerly a popular tool for this, but appears largely unmaintained. Should likely be avoided.

## Implementation Path inside `snotel_lib`
Because `snotel_lib/client.py` isolates the data extraction via `SnotelClient.get_station_data()` and `SnotelClient.get_all_station_data()`, migrating sources only requires rewriting this single class.

The rest of the library (`clean/`, computations, schemas) operates strictly on the parsed `pl.DataFrame` defined by `SnotelDataSchema`. As long as the new client outputs this exact Polars schema, the backend and frontend are entirely decoupled from the migration.

## Status Update: Metloom Integration (March 2026)

A dual-client architecture was implemented in `snotel_lib/clients/`, abstracting behind a `BaseSnotelClient` interface.
1. **`EgagliClient`**: The original GitHub-scraping logic. **Best for massive bulk downloads** (e.g. all 900+ stations) because it fetches a single pre-aggregated tarball in seconds. Currently in use by `generate_leaderboard.py` to power the nightly build.
2. **`MetloomClient`**: A new wrapper integrating `metloom` as a native interface to the NRCS AWDB API. This maps imperial NRCS units (inches/Fahrenheit) back to the metric `SnotelDataSchema` standard.
   * *Limitation:* Metloom/AWDB does not easily support a single, fast endpoint to download the historical context for *all* stations simultaneously. Hitting the API sequentially/concurrently for a massive backend generation job is too slow.
   * *Use Case:* Perfectly suited for a future dynamic API where a user queries a *single station* for up-to-the-minute live data, bypassing the delay of the `egagli` GitHub Action.
