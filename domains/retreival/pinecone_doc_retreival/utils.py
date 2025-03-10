from domains.settings import config_settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone
from domains.injestion.utils import get_embeddings
from loguru import logger

from typing import Any


def load_index(index_name: str, namespace: str | None = None) -> Pinecone:
    # load a pinecone index
    return Pinecone.from_existing_index(
        index_name=index_name,
        embedding=get_embeddings(model_key="EMBEDDING_MODEL"),
        namespace=namespace,
    )

from langchain_pinecone import PineconeVectorStore
from domains.injestion.utils import get_embeddings

def get_related_docs_without_context(
    index_name: str,
    namespace: str,
    question: str,
    total_docs_to_retrieve: int = 10
) ->list[tuple[Document, float]]:
    try:
        docsearch = load_index(index_name)

        logger.info("Getting related docs from vector DB without context")

        # Perform similarity search without a filter
        related_docs_with_score = docsearch.similarity_search_with_score(
            query=question,
            k=total_docs_to_retrieve or config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE,
            namespace=namespace,
        )
        logger.info(f"Related docs without context: {related_docs_with_score}")
        return related_docs_with_score

    except Exception as e:
        logger.error(f"Failed to get related docs without context: {e}")
        return []


if __name__ == "__main__":
    get_related_docs_without_context(
        config_settings.PINECONE_INDEX_NAME,
        config_settings.PINECONE_DEFAULT_DEV_NAMESPACE,
        "candidate name",
        config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE
    )
