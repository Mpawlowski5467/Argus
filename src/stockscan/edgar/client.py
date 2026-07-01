"""Throttled, retrying SEC EDGAR HTTP client.

SEC enforces a hard 10 req/s per IP and requires a descriptive User-Agent that
includes a contact address; violations return 403/429 and a ~10-minute IP block.
We stay at <=8 req/s, always send the User-Agent, and back off on transient errors.

For the historical build we prefer bulk downloads (the quarterly Financial
Statement Data Sets and the nightly bulk zips) over per-CIK crawling; this client
is for those small JSON/index pulls and same-day incremental deltas.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx

from ..config import EDGAR_MAX_RPS, EDGAR_USER_AGENT


class _RateLimiter:
    """Thread-safe minimum-interval limiter (token-bucket of size 1)."""

    def __init__(self, max_rps: float):
        self._min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class EdgarClient:
    """Minimal EDGAR client: throttled GET with retries, JSON/bytes helpers."""

    DATA_HOST = "https://data.sec.gov"
    WWW_HOST = "https://www.sec.gov"

    def __init__(
        self,
        user_agent: str = EDGAR_USER_AGENT,
        max_rps: float = EDGAR_MAX_RPS,
        timeout: float = 30.0,
        max_retries: int = 5,
    ):
        if "@" not in user_agent:
            raise ValueError(
                "EDGAR User-Agent must include a contact email (SEC requirement). "
                f"Got: {user_agent!r}"
            )
        self._limiter = _RateLimiter(max_rps)
        self._max_retries = max_retries
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
        )

    # -- lifecycle -------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- core ------------------------------------------------------------------
    def _get(self, url: str) -> httpx.Response:
        backoff = 1.0
        last_exc: Exception | None = None
        resp: httpx.Response | None = None
        for _ in range(self._max_retries):
            self._limiter.wait()
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as exc:  # network/timeout — retry
                last_exc = exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                time.sleep(backoff)  # throttled/blocked/transient — back off and retry
                backoff = min(backoff * 2, 60.0)
                continue
            resp.raise_for_status()  # 4xx we won't recover from
        if last_exc is not None:
            raise last_exc
        status = resp.status_code if resp is not None else "unknown"
        raise RuntimeError(
            f"EDGAR request failed after {self._max_retries} retries "
            f"(last status={status}): {url}"
        )

    def get_json(self, url: str) -> Any:
        return self._get(url).json()

    def get_bytes(self, url: str) -> bytes:
        return self._get(url).content

    # -- convenience -----------------------------------------------------------
    def company_tickers(self) -> dict:
        """The full ticker <-> CIK map. Small JSON; doubles as a connectivity check."""
        return self.get_json(f"{self.WWW_HOST}/files/company_tickers.json")
