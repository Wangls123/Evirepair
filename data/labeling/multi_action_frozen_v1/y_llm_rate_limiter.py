from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

@dataclass
class LabelingRunMetrics:

    total_api_calls: int = 0
    total_retries: int = 0
    rate_limit_429_count: int = 0
    total_latency_sec: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_call(self, *, latency_sec: float, is_429: bool = False) -> None:
        with self._lock:
            self.total_api_calls += 1
            self.total_latency_sec += latency_sec
            if is_429:
                self.rate_limit_429_count += 1

    def record_retries(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self.total_retries += n

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            calls = self.total_api_calls
            avg_lat = self.total_latency_sec / calls if calls else 0.0
            return {
                "total_api_calls": calls,
                "retry_count": self.total_retries,
                "rate_limit_429_count": self.rate_limit_429_count,
                "average_latency_sec": round(avg_lat, 4),
            }

class GlobalRateLimiter:

    def __init__(self, target_rpm: float, *, window_sec: float = 60.0) -> None:
        self.target_rpm = max(0.0, float(target_rpm))
        self.window_sec = window_sec
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        if self.target_rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_sec
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.target_rpm:
                    self._timestamps.append(now)
                    return
                wait_until = self._timestamps[0] + self.window_sec
                sleep_for = max(0.001, wait_until - now)
            time.sleep(sleep_for)

class RateLimitedLLMClient:

    def __init__(
        self,
        inner: Any,
        *,
        rate_limiter: GlobalRateLimiter | None,
        metrics: LabelingRunMetrics | None = None,
    ) -> None:
        self._inner = inner
        self._rate_limiter = rate_limiter
        self._metrics = metrics
        self.model = getattr(inner, "model", None)

    def chat(self, *, system: str, user: str) -> dict:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        start = time.monotonic()
        is_429 = False
        try:
            return self._inner.chat(system=system, user=user)
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate limit" in err.lower():
                is_429 = True
            raise
        finally:
            if self._metrics is not None:
                self._metrics.record_call(
                    latency_sec=time.monotonic() - start,
                    is_429=is_429,
                )

def make_worker_llm_client(
    *,
    rate_limiter: GlobalRateLimiter | None,
    metrics: LabelingRunMetrics | None,
) -> Any:

    from smarthome_mdf.multi_action_frozen_v1.y_llm_config import get_y_llm_config
    from smarthome_mdf.v4_synthesis.llm_schema import LLMClient

    ycfg = get_y_llm_config()
    base = LLMClient(
        api_key=ycfg["api_key"],
        base_url=ycfg["base_url"],
        model=ycfg["model"],
        temperature=ycfg["temperature"],
        top_p=ycfg["top_p"],
    )
    if rate_limiter is None and metrics is None:
        return base
    return RateLimitedLLMClient(base, rate_limiter=rate_limiter, metrics=metrics)
