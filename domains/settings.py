from pydantic_settings import BaseSettings
from typing import ClassVar
import os


class Settings(BaseSettings):
    # openai
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_CHAT_BASE_URL: str = os.environ.get(
        "OPENAI_CHAT_BASE_URL", "https://api.openai.com/v1/chat/completions"
    )
    THREASHOLD_MESSAGE_TO_SUMMARIZE: int = int(
        os.environ.get("THREASHOLD_MESSAGE_TO_SUMMARIZE", 10)
    )
    OPENAI_EMBEDDING_MODEL_NAME: str = os.environ.get(
        "OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small"
    )
    SUMMARIZE_LLM_MODEL: str = os.environ.get("SUMMARIZE_LLM_MODEL", "gpt-4o")

    API_HOSTNAME: str = os.environ.get("API_HOSTNAME", "https://dummyjson.com/c")
    VECTOR_DATABASE_TO_USE: str = os.environ.get("VECTOR_DATABASE_TO_USE","pinecone")

    MAX_TOKEN_LIMIT: int = os.environ.get("MAX_TOKEN_LIMIT", 1500)

    # pinecone
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_ENV: str = os.environ.get("PINECONE_ENV", "")
    PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX", "cloud202-dev")
    PINECONE_INDEX_METRIC_TYPE: str = os.environ.get("PINECONE_METRIC_TYPE", "cosine")
    PINECONE_INDEX_CLOUD_NAME: str = os.environ.get("PINECONE_INDEX_CLOUD_NAME", "aws")
    PINECONE_INDEX_REGION_NAME: str = os.environ.get("PINECONE_INDEX_REGION_NAME", "us-east-1")
    PINECONE_DEFAULT_DEV_NAMESPACE: str = os.environ.get("PINCEONE_DEFAULT_DEV_NAMESPACE", "default_dev")
    PINECONE_DROP_INDEX_NAME_STATUS: bool = os.environ.get("PINECONE_DROP_INDEX_NAME_STATUS", False)
    PINECONE_TOTAL_DOCS_TO_RETRIEVE: int = os.environ.get(
        "PINECONE_TOTAL_DOCS_TO_RETRIEVE", 10
    )

    # citation
    CITATIONS_TOGGLE: bool = os.environ.get(
        "CITATIONS_TOGGLE", False
    )

    # chunk setting
    CHUNK_SIZE: int = os.environ.get("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP: int = os.environ.get("CHUNK_OVERLAP", 200)

    # classification
    CLASSIFICATION_MODEL: str = os.environ.get("CLASSIFICATION_MODEL", "gpt-4o")


    # conversion memory key
    CONVERSATIONAL_BUFFER_WINDOW_MEMORY_KEY: str = os.environ.get(
        "CONVERSATIONAL_BUFFER_WINDOW_MEMORY_KEY", "chat_history"
    )
    LANGCHAIN_MEMORY_BUFFER_WINDOW: int = os.environ.get(
        "LANGCHAIN_MEMORY_BUFFER_WINDOW", 10
    )
    CONVERSATIONAL_BUFFER_WINDOW_INPUT_KEY: str = os.environ.get(
        "CONVERSATIONAL_BUFFER_WINDOW_INPUT_KEY", "question"
    )
    CHAT_CONTEXT_HUMAN_MESSAGE_KEY: str = os.environ.get(
        "CHAT_CONTEXT_HUMAN_MESSAGE_KEY", "human"
    )
    CHAT_CONTEXT_AI_MESSAGE_KEY: str = os.environ.get(
        "CHAT_CONTEXT_AI_MESSAGE_KEY", "ai"
    )

    # optimized question
    OPTIMIZED_QUESTION_MODEL: str = os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o-mini")
    MINIMUM_SCORE: float = float(os.environ.get("MINIMUM_SCORE", 0.5))

    UPLOAD_FOLDER: str = os.environ.get("UPLOAD_FOLDER", "uploads")
    LOGS_FOLDER: str = os.environ.get("LOGS_FOLDER", "logs")

    LLM_SERVICE: str = os.environ.get(
        "LLM_SERVICE", "openai"
    )

    MAX_TOKENS: int = os.environ.get("MAX_TOKENS", 1500)
    # Modular Model Names
    LLMS: ClassVar[dict] = {
        "OPENAI_CHAT_MODEL_NAME": os.environ.get("OPENAI_CHAT_MODEL_NAME", "gpt-4o"),
        "SUMMARIZE_LLM_MODEL": os.environ.get("SUMMARIZE_LLM_MODEL", "gpt-4o"),
        "OPENAI_EMBEDDING_MODEL_NAME": os.environ.get(
            "OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        "CLASSIFICATION_MODEL": os.environ.get("CLASSIFICATION_MODEL", "gpt-4o"),
        "OPTIMIZED_QUESTION_MODEL": os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o"),
        "OPENAI_CHAT": os.environ.get("OPENAI_CHAT", "gpt-4o"),
    }

    AZURE_OPENAI_SETTINGS: ClassVar[dict] = {
        "LLM_MODEL_NAME": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_LLM_MODEL_NAME", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_LLM_MODEL_NAME", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_LLM_MODEL_NAME", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_LLM_MODEL_NAME", ""),
            "OPTIMIZED_QUESTION_MODEL": os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o-mini"),
        },
        "MODEL_NAME_GPT_4": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_MODEL_NAME_GPT_4", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_MODEL_NAME_GPT_4", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_MODEL_NAME_GPT_4", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_MODEL_NAME_GPT_4", ""),
        }
    }

config_settings = Settings()