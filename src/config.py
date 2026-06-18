from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class MongoSettings:
    uri: str
    database: str
    collection: str


def load_mongo_settings() -> MongoSettings:
    """Load MongoDB connection settings from environment variables."""
    load_dotenv()

    return MongoSettings(
        uri=_required_env("MONGODB_URI"),
        database=_required_env("MONGODB_DATABASE"),
        collection=_required_env("MONGODB_COLLECTION"),
    )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
