"""
LLM Client (Google Gemini via LangChain).

Uses ChatGoogleGenerativeAI and with_structured_output for Pydantic outputs.
"""

import os
from typing import Optional, Type, TypeVar, Any, Dict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Gemini LLM client for text and structured (Pydantic) generation."""

    def __init__(self, provider: str = "gemini", model: Optional[str] = None) -> None:
        """
        Args:
            provider: Must be "gemini" or "google".
            model: Gemini model name (default: gemini-2.0-flash).
        """
        provider = provider.lower()
        if provider not in {"gemini", "google"}:
            raise ValueError("Only Gemini is supported. Got provider={!r}.".format(provider))

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")

        self.model = model or "gemini-2.0-flash"
        self._llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            google_api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Simple text generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate free-form text."""
        if system_prompt:
            messages = [
                SystemMessage(content=system_prompt),
                ("human", prompt),
            ]
            resp = self._llm.invoke(messages)
        else:
            resp = self._llm.invoke(prompt)

        return resp.content if hasattr(resp, "content") else str(resp)

    # ------------------------------------------------------------------
    # Structured / Pydantic output
    # ------------------------------------------------------------------
    def generate_structured(
        self,
        model: Type[T],
        system_prompt: str,
        user_template: str,
        variables: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
    ) -> T:
        """Generate a Pydantic object via with_structured_output."""
        variables = variables or {}

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                ("human", user_template),
            ]
        )

        structured_llm = self._llm.with_structured_output(model)
        chain = prompt | structured_llm

        result: T = chain.invoke(variables)
        return result
