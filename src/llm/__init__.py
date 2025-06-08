import os
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.output_parsers.json import SimpleJsonOutputParser

json_parser = SimpleJsonOutputParser()


@lru_cache(maxsize=None)
def get_llm(provider: str, model: str):
    """
    Return a cached LLM instance based on provider and model.
    Supported providers: openai, groq, openrouter
    """
    timeout = 30
    max_retries = 3

    if provider == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "groq":
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


__all__ = ["json_parser", "get_llm"]

