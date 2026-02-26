from pathlib import Path

from platformdirs import user_cache_dir

# Default cache directory for snotel data
DEFAULT_CACHE_DIR = Path(user_cache_dir("snotel_data", "eruebush"))

# Cache durations in days
METADATA_CACHE_DAYS = 1
STATION_CACHE_DAYS = 1


def get_cache_dir() -> Path:
    """Returns the cache directory, creating it if it doesn't exist."""
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR
