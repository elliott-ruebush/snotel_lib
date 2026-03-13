import logging
import shutil
from pathlib import Path

from snotel_lib.io import get_default_cache_dir

logger = logging.getLogger(__name__)


def format_size(size_bytes_int: int) -> str:
    """Formats a size in bytes to a human-readable string."""
    size_in_clearest_unit = size_bytes_int
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_clearest_unit < 1024:
            return f"{size_in_clearest_unit:.2f} {unit}"
        size_in_clearest_unit = size_in_clearest_unit // 1024
    return f"{size_in_clearest_unit:.2f} PB"


def clean_cache_dir(cache_dir: Path | None = None, force: bool = False) -> None:
    """
    Clears all files and subdirectories from the cache directory.

    Args:
        cache_dir: Optional custom cache directory. Defaults to the standard app cache.
        force: If True, skip the confirmation prompt.
    """
    target = cache_dir or get_default_cache_dir()
    if not target.exists():
        logger.info(f"Cache directory {target} does not exist. Nothing to clean.")
        return

    items = list(target.iterdir())
    if not items:
        logger.info(f"Cache directory {target} is already empty.")
        return

    total_size = 0
    logger.info(f"\nFiles in cache directory ({target}):")
    for item in items:
        if item.is_file():
            size = item.stat().st_size
            total_size += size
            logger.info(f"  - {item.name: <40} {format_size(size): >10}")
        elif item.is_dir():
            logger.info(f"  - {item.name}/ (directory)")

    logger.info(f"\nTotal cache size: {format_size(total_size)}")

    if not force:
        try:
            response = input("\nAre you sure you want to clear the cache? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Abort.")
                return
        except EOFError:
            # Handle non-interactive environments by defaulting to Abort
            logger.info("\nNo input received. Aborting deletion.")
            return

    logger.info(f"Cleaning cache directory: {target}")

    for item in items:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    logger.info("Cache cleared successfully.")


def main() -> None:
    """CLI entry point for cleaning the cache."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    clean_cache_dir()


if __name__ == "__main__":
    main()
