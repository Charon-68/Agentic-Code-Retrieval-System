"""Utility helpers for the sample repository."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

ERROR_CODE_INVALID_CONFIG = "ERR_CONFIG_001"
ERROR_CODE_CONNECTION_TIMEOUT = "ERR_CONN_002"
ERROR_CODE_AUTH_FAILURE = "ERR_AUTH_003"


def slugify(value: str, max_length: int = 64) -> str:
    """Convert arbitrary text into a URL-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:max_length]


def generate_request_id(prefix: str = "req") -> str:
    """Generate a unique request identifier with a timestamp component."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha256(f"{prefix}-{timestamp}".encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{digest}"


def validate_config(config: dict[str, Any], required_keys: list[str]) -> Optional[str]:
    """
    Validate that all required keys exist in a configuration dictionary.

    Returns an error message string on failure, or None if valid.
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        return f"{ERROR_CODE_INVALID_CONFIG}: missing keys {missing}"
    return None


def retry_with_backoff(
    func: Any,
    max_attempts: int = 3,
    base_delay: float = 0.5,
) -> Any:
    """Execute a callable with exponential backoff retries."""
    import time

    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.2fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"{ERROR_CODE_CONNECTION_TIMEOUT}: all {max_attempts} attempts failed"
    ) from last_error
