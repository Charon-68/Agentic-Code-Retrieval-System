"""LLM provider configuration loaded safely from environment variables."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

load_dotenv()


class LLMSettings(BaseModel):
    """Validated settings for LLM API clients."""

    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Default OpenAI chat model")
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Default Anthropic chat model",
    )


@lru_cache(maxsize=1)
def get_settings() -> LLMSettings:
    """Load and cache LLM settings from the environment."""
    return LLMSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def get_openai_client() -> OpenAI:
    """Return a synchronous OpenAI client, raising if the API key is missing."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=settings.openai_api_key)


def get_async_openai_client() -> AsyncOpenAI:
    """Return an asynchronous OpenAI client, raising if the API key is missing."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_anthropic_client() -> Anthropic:
    """Return a synchronous Anthropic client, raising if the API key is missing."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def get_async_anthropic_client() -> AsyncAnthropic:
    """Return an asynchronous Anthropic client, raising if the API key is missing."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


def is_openai_configured() -> bool:
    """Check whether OpenAI credentials are available."""
    return bool(get_settings().openai_api_key)


def is_anthropic_configured() -> bool:
    """Check whether Anthropic credentials are available."""
    return bool(get_settings().anthropic_api_key)
