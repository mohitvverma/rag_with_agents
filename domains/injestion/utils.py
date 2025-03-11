from langchain_openai import OpenAIEmbeddings
from domains.settings import config_settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from domains.models import RequestStatus
from domains.status_util import call_update_status_api

def split_text(text: list[Document], CHUNK_SIZE: int, CHUNK_OVERLAP: int) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(text)


def get_embeddings(
        model_key: str
):
    if model_key == "EMBEDDING_MODEL":
        return OpenAIEmbeddings(
            model=config_settings.LLMS.get("OPENAI_EMBEDDING_MODEL_NAME"),
            api_key=config_settings.OPENAI_API_KEY,
        )

    elif model_key == "AZURE_EMBEDDING_MODEL":
        return OpenAIEmbeddings(
            model=config_settings.LLMS.get("SUMMARIZE_LLM_MODEL"),
            api_key=config_settings.OPENAI_API_KEY,
        )


def update_status(api_path: str, request_status: RequestStatus, token: str=None) -> None:
    if api_path:
        call_update_status_api(api_path, request_status)


