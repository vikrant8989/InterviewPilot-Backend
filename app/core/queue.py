from __future__ import annotations

# Temporarily disable RQ for Windows compatibility
# from rq import Queue
from redis import Redis

from app.core.config import settings


def make_redis() -> Redis | None:
    if not settings.redis_url:
        return None
    return Redis.from_url(settings.redis_url)


def make_transcription_queue():
    # Return None for now - will fix later
    return None

