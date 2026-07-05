"""Database connection utilities with retry logic for the sample repository."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from sample_repo.utils import ERROR_CODE_AUTH_FAILURE, ERROR_CODE_CONNECTION_TIMEOUT

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConnectionState(Enum):
    """Lifecycle states for a database connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class DatabaseConfig:
    """Configuration for establishing a database connection."""

    host: str = "localhost"
    port: int = 5432
    database: str = "app_db"
    username: str = "app_user"
    password: str = ""
    ssl_enabled: bool = True
    connect_timeout_seconds: float = 10.0
    max_retries: int = 3


class RetryPolicy:
    """Configurable retry policy for transient connection failures."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 30.0,
        exponential_base: float = 2.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.exponential_base = exponential_base

    def compute_delay(self, attempt: int) -> float:
        """Calculate backoff delay for a given attempt number (1-indexed)."""
        delay = self.base_delay_seconds * (self.exponential_base ** (attempt - 1))
        return min(delay, self.max_delay_seconds)

    def execute(self, operation: Callable[[], T], description: str = "operation") -> T:
        """Run an operation with retries according to this policy."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                delay = self.compute_delay(attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.2fs",
                    description,
                    attempt,
                    self.max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise ConnectionError(
            f"{ERROR_CODE_CONNECTION_TIMEOUT}: {description} failed after "
            f"{self.max_attempts} attempts"
        ) from last_error


@dataclass
class DatabaseConnection:
    """Simulated database connection with retry-aware connect/disconnect."""

    config: DatabaseConfig
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    state: ConnectionState = field(default=ConnectionState.DISCONNECTED)
    _session_id: Optional[str] = field(default=None, repr=False)

    def connect(self) -> str:
        """Establish a connection, using retry policy for transient failures."""
        self.state = ConnectionState.CONNECTING
        logger.info(
            "Connecting to %s:%d/%s",
            self.config.host,
            self.config.port,
            self.config.database,
        )

        def _do_connect() -> str:
            if not self.config.username:
                raise PermissionError(
                    f"{ERROR_CODE_AUTH_FAILURE}: username is required"
                )
            session_id = f"sess_{self.config.host}_{int(time.time())}"
            self._session_id = session_id
            self.state = ConnectionState.CONNECTED
            return session_id

        session = self.retry_policy.execute(_do_connect, description="database connect")
        logger.info("Connected with session_id=%s", session)
        return session

    def disconnect(self) -> None:
        """Close the active connection."""
        if self.state == ConnectionState.CONNECTED:
            logger.info("Disconnecting session %s", self._session_id)
        self._session_id = None
        self.state = ConnectionState.DISCONNECTED

    def execute_query(self, sql: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Execute a read query against the simulated database."""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.debug("Executing query: %s | params=%s", sql[:80], params)
        return [{"result": "ok", "rows_affected": 0}]

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED
