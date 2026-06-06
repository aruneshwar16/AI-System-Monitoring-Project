"""Gemini API client wrapper using the Google GenAI SDK."""

import os
from typing import Optional

from dotenv import load_dotenv

from utils.constants import GEMINI_MODEL
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class GeminiClient:
    """Thin wrapper around the google-genai Client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def _get_client(self):
        if self._client is None:
            if not self.is_configured:
                raise ValueError(
                    "GEMINI_API_KEY not set. Add it to .env or environment variables."
                )
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate text content from a prompt."""
        client = self._get_client()
        response = client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()
