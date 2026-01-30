"""
LLM Client Wrapper (Gemini + LangChain)

This project now uses ONLY Google Gemini via LangChain, and uses
`with_structured_output` wherever structured Pydantic models are needed.
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
    """Unified LLM client interface (Gemini-only)."""

    def __init__(self, provider: str = "gemini", model: Optional[str] = None) -> None:
        """
        Initialize Gemini client.

        Args:
            provider: Kept for backward compatibility, must be "gemini".
            model: Gemini model name (default: "gemini-2.0-flash").
        """
        provider = provider.lower()
        if provider not in {"gemini", "google"}:
            raise ValueError(
                f"Only Gemini is supported now. Got provider={provider!r}."
            )

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")

        self.model = model or "gemini-2.0-flash"
        # Base LLM instance (temperature fixed at 0 for stability / structured outputs)
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
        temperature: float = 0.7,  # kept for API compatibility (currently ignored)
        max_tokens: Optional[int] = None,  # kept for API compatibility (currently ignored)
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate free-form text from a prompt."""
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
        temperature: float = 0.1,  # kept for API compatibility (Gemini temp fixed at init)
    ) -> T:
        """
        Generate a structured Pydantic object using LangChain's
        `with_structured_output`.

        Args:
            model: Pydantic BaseModel subclass describing the output schema.
            system_prompt: System instructions for the assistant.
            user_template: Templated user message (used with `.format(**variables)`).
            variables: Variables for the user template.
            temperature: Sampling temperature (low = more deterministic).
        """
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
