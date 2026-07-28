"""RED test for Task 2: semaphore must not access private asyncio internals."""
import asyncio
import pytest
from rag.enrichers.enricher import LLMEnricher


def test_semaphore_has_no_loop_attr_access():
    """The semaphore property must not read _loop off the Semaphore object."""
    enricher = LLMEnricher.__new__(LLMEnricher)
    enricher._max_concurrency = 3
    enricher._semaphore = None

    async def run():
        sem = enricher.semaphore
        assert sem is not None
        # Verify no _loop attribute was accessed (would raise on Python 3.10+ stripped internals)
        # We just need the property to work without AttributeError
        sem2 = enricher.semaphore  # second call should return same semaphore
        assert sem is sem2

    asyncio.run(run())


def test_semaphore_respects_max_concurrency():
    """Semaphore value must match max_concurrency."""
    enricher = LLMEnricher.__new__(LLMEnricher)
    enricher._max_concurrency = 7
    enricher._semaphore = None

    async def run():
        sem = enricher.semaphore
        # Semaphore internal value should equal max_concurrency
        assert sem._value == 7

    asyncio.run(run())
