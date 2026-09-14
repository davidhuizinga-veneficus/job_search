"""Simple Gemini rate-limit guard for the job-search pipeline."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GeminiRateLimiter:
    """Throttle requests and input size to stay under the provider's limits."""

    requests_per_minute: int = 15
    max_input_tokens: int = 250_000
    window_seconds: int = 60
    _timestamps: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        env_rpm = os.environ.get("GEMINI_REQUESTS_PER_MINUTE")
        env_tokens = os.environ.get("GEMINI_MAX_INPUT_TOKENS")
        if env_rpm:
            try:
                self.requests_per_minute = int(env_rpm)
            except ValueError:
                pass
        if env_tokens:
            try:
                self.max_input_tokens = int(env_tokens)
            except ValueError:
                pass

    def acquire(self) -> None:
        """Wait until the API rate window has capacity."""
        with self._lock:
            if not isinstance(self._timestamps, deque):
                self._timestamps = deque(self._timestamps)

            while True:
                now = time.monotonic()
                window_start = now - self.window_seconds
                while self._timestamps and self._timestamps[0] <= window_start:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return

                oldest = self._timestamps[0]
                wait_for = max(0.0, (oldest + self.window_seconds) - now)
                if wait_for <= 0:
                    self._timestamps.popleft()
                    continue
                time.sleep(wait_for)

    def estimate_tokens(self, text: str) -> int:
        """Cheap conservative token estimate for budgeting long prompts."""
        if not text:
            return 0
        return max(1, len(text.split()) // 3)

    def ensure_prompt_within_budget(self, text: str, *, safety_margin: float = 0.9) -> str:
        """Trim a prompt to the configured input-token budget and preserve the most useful tail."""
        budget = max(1, int(self.max_input_tokens * safety_margin))
        if self.estimate_tokens(text) <= budget:
            return text

        words = text.split()
        keep = []
        running_words = 0
        for word in reversed(words):
            running_words += 1
            keep.append(word)
            if self.estimate_tokens(" ".join(reversed(keep))) > budget:
                keep.pop()
                if not keep:
                    return ""
                break

        return " ".join(reversed(keep))


GLOBAL_GEMINI_RATE_LIMITER = GeminiRateLimiter()


def get_shared_gemini_limiter() -> GeminiRateLimiter:
    return GLOBAL_GEMINI_RATE_LIMITER


def reset_shared_gemini_limiter() -> GeminiRateLimiter:
    global GLOBAL_GEMINI_RATE_LIMITER
    GLOBAL_GEMINI_RATE_LIMITER = GeminiRateLimiter(
        requests_per_minute=15,
        max_input_tokens=250_000,
        window_seconds=60,
    )
    return GLOBAL_GEMINI_RATE_LIMITER
