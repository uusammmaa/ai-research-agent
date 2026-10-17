import pytest

from agent.reliability import Deadline, RetryPolicy, RunTimeoutError, retry_async


class Flaky:
    def __init__(self, failures: list[Exception], result="ok"):
        self.failures = list(failures)
        self.calls = 0
        self.result = result

    async def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


class Retryable(Exception):
    pass


class Fatal(Exception):
    pass


def is_retryable(exc):
    return isinstance(exc, Retryable)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


async def test_retries_until_success_with_exponential_backoff():
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    fn = Flaky([Retryable(), Retryable()])
    policy = RetryPolicy(max_attempts=3, base_delay=0.5, jitter=False)
    assert await retry_async(fn, policy=policy, is_retryable=is_retryable, sleep=sleep) == "ok"
    assert fn.calls == 3
    assert sleeps == [0.5, 1.0]


async def test_non_retryable_error_raises_immediately():
    fn = Flaky([Fatal()])
    with pytest.raises(Fatal):
        await retry_async(fn, policy=RetryPolicy(), is_retryable=is_retryable, sleep=_no_sleep)
    assert fn.calls == 1


async def test_exhausted_attempts_reraise_last_error():
    fn = Flaky([Retryable("1"), Retryable("2"), Retryable("3")])
    with pytest.raises(Retryable, match="3"):
        await retry_async(
            fn, policy=RetryPolicy(max_attempts=3), is_retryable=is_retryable, sleep=_no_sleep
        )
    assert fn.calls == 3


async def test_on_retry_reports_attempt_and_delay():
    seen = []
    fn = Flaky([Retryable()])
    await retry_async(
        fn,
        policy=RetryPolicy(base_delay=0.25, jitter=False),
        is_retryable=is_retryable,
        on_retry=lambda attempt, delay, exc: seen.append((attempt, delay, type(exc))),
        sleep=_no_sleep,
    )
    assert seen == [(1, 0.25, Retryable)]


async def test_retry_stops_when_backoff_would_overrun_deadline():
    clock = FakeClock()
    deadline = Deadline(1.0, clock=clock)
    clock.now = 0.9  # 0.1s left, next backoff is 0.5s
    fn = Flaky([Retryable(), Retryable()])
    with pytest.raises(Retryable):
        await retry_async(
            fn,
            policy=RetryPolicy(base_delay=0.5, jitter=False),
            is_retryable=is_retryable,
            deadline=deadline,
            sleep=_no_sleep,
        )
    assert fn.calls == 1


def test_delay_is_capped_and_jitter_stays_in_range():
    policy = RetryPolicy(base_delay=1.0, max_delay=4.0, jitter=False)
    assert [policy.delay_for(n) for n in (1, 2, 3, 4, 5)] == [1.0, 2.0, 4.0, 4.0, 4.0]
    jittered = RetryPolicy(base_delay=1.0, max_delay=4.0)
    assert all(0 <= jittered.delay_for(3) <= 4.0 for _ in range(100))


def test_deadline_caps_timeouts_and_raises_when_spent():
    clock = FakeClock()
    deadline = Deadline(10.0, clock=clock)
    assert deadline.cap(30) == 10.0
    clock.now = 7.0
    assert deadline.cap(30) == pytest.approx(3.0)
    assert deadline.cap(1) == 1
    clock.now = 10.0
    assert deadline.expired()
    with pytest.raises(RunTimeoutError):
        deadline.cap(5)


async def _no_sleep(_):
    return None
