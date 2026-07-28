import logging
import logging.handlers
import os
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"
_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    _LOG_DIR.mkdir(exist_ok=True)
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                _LOG_FILE,
                maxBytes=10_000_000,  # 10 MB per file
                backupCount=5,
                encoding="utf-8",
            ),
        ],
    )
    logging.getLogger(__name__).info("logging configured: level=%s file=%s", level, _LOG_FILE)
