"""OpenAI completion service implementation."""

import logging
from typing import Any, Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from app.core.config import Settings
from app.services.llm.base import CompletionService

logger = logging.getLogger(__name__)


class OpenAICompletionService(CompletionService):
    """OpenAI completion service implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key,base_url=settings.openai_base_url)
        else:
            self.client = None  # type: ignore
            logger.warning(
                "OpenAI API key is not set. LLM features will be disabled."
            )

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Generate a completion from the language model."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping generation."
            )
            return None

        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        logger.info(f"Generated response: {content}")

        if content is None:
            logger.warning("Received None response from OpenAI")
            return None

        try:
            # Parse the response into the expected model
            parsed_response = response_model.model_validate_json(content)
            return parsed_response
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    async def decompose_query(self, query: str) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping decomposition."
            )
            return {"sub_queries": [query]}

        # TODO: Implement the actual decomposition logic here
        return {"sub_queries": [query]}
