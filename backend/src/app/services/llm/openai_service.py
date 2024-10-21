"""Service for interacting with OpenAI models."""

import logging
from typing import Any, List, Optional, Type

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel

from app.core.config import Settings
from app.services.llm.base import LLMService

logger = logging.getLogger(__name__)


class OpenAIService(LLMService):
    """Service for interacting with OpenAI models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model
            )
        else:
            self.client = None  # type: ignore
            self.embeddings = None  # type: ignore
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

        response = self.client.beta.chat.completions.parse(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
        )

        parsed_response = response.choices[0].message.parsed
        logger.info(f"Generated response: {parsed_response}")

        if parsed_response is None:
            logger.warning("Received None response from OpenAI")
            return None

        try:
            validated_response = response_model(**parsed_response.model_dump())
            if all(
                value is None
                for value in validated_response.model_dump().values()
            ):
                logger.info("All fields in the response are None")
                return None
            return validated_response
        except ValueError as e:
            logger.error(f"Error validating response: {e}")
            return None

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for text."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping embeddings."
            )
            return []

        return self.embeddings.embed_documents(texts)

    async def decompose_query(self, query: str) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping decomposition."
            )
            return {"sub_queries": [query]}

        # TODO: Implement the actual decomposition logic here
        return {"sub_queries": [query]}
