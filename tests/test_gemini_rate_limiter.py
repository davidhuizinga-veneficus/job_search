import time
import unittest
from unittest.mock import patch

from job_search.gemini_limiter import GeminiRateLimiter


class TestGeminiRateLimiter(unittest.TestCase):
    def test_waits_when_minute_limit_is_reached(self):
        limiter = GeminiRateLimiter(requests_per_minute=2, max_input_tokens=1000)
        now = time.monotonic()
        limiter._timestamps = [now - 30, now - 5]

        with patch("job_search.gemini_limiter.time.monotonic", side_effect=[now, now + 60]), patch(
            "job_search.gemini_limiter.time.sleep"
        ) as sleep_mock:
            limiter.acquire()

        sleep_mock.assert_called_once()

    def test_truncates_prompt_when_forecast_exceeds_token_budget(self):
        limiter = GeminiRateLimiter(requests_per_minute=2, max_input_tokens=10)
        text = "word " * 50

        safe = limiter.ensure_prompt_within_budget(text)

        self.assertLess(len(safe), len(text))
        self.assertLessEqual(limiter.estimate_tokens(safe), 10)


if __name__ == "__main__":
    unittest.main()
