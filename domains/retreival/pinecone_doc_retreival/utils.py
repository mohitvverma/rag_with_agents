import asyncio

from domains.settings import config_settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone
from typing import Tuple, List
from domains.injestion.utils import get_embeddings
from loguru import logger

from contextlib import asynccontextmanager
from functools import lru_cache

from langchain_pinecone import PineconeVectorStore
from domains.injestion.utils import get_embeddings


@lru_cache(maxsize=32)
def load_index(index_name: str, namespace: str | None = None) -> Pinecone:
    # load a pinecone index
    return Pinecone.from_existing_index(
        index_name=index_name,
        embedding=get_embeddings(model_key="EMBEDDING_MODEL"),
        namespace=namespace,
    )


@asynccontextmanager
async def get_docsearch(index_name: str) -> PineconeVectorStore:
    """
    Context manager for handling document search initialization.
    """
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=get_embeddings(model_key="EMBEDDING_MODEL"),
        )
        yield docsearch
    except Exception as e:
        logger.error(f"Failed to initialize document search: {e}")
        raise


async def get_related_docs_with_score(
    index_name: str,
    namespace: str,
    question: str,
    total_docs_to_retrieve: int = 10
) ->list[tuple[Document, float]]:
    try:
        docsearch = load_index(index_name=index_name)

        # Perform similarity search without a filter
        related_docs_with_score = await docsearch.asimilarity_search_with_relevance_scores(
            query=question,
            namespace=namespace,
        )
        return related_docs_with_score

    except Exception as e:
        logger.error(f"Failed to get related docs without context: {e}")
        return []


async def get_related_docs_without_context(
        index_name: str,
        namespace: str,
        question: str,
        total_docs_to_retrieve: int = 10
) -> List[Tuple[Document, float]]:
    """
    Retrieve related documents using PineconeVectorStore retriever.
    """
    try:
        async with get_docsearch(index_name) as docsearch:
            retriever = docsearch.as_retriever(
                search_kwargs={
                    "k": total_docs_to_retrieve,
                    "namespace": namespace
                }
            )

            related_docs = await retriever.ainvoke(input=question)
            logger.info(f"Retrieved {len(related_docs)} documents")
            return related_docs

    except Exception as e:
        logger.error(f"Error in get_related_docs_without_context: {str(e)}")
        return []


async def main() -> None:
    """
    Example usage of the retrieval functions.
    """
    result = await get_related_docs_with_score(
        index_name=config_settings.PINECONE_INDEX_NAME,
        namespace=config_settings.PINECONE_DEFAULT_DEV_NAMESPACE,
        question="candidate name",
        total_docs_to_retrieve=config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE
    )
    logger.info(f"Retrieved {len(result)} documents")


if __name__ == "__main__":
    asyncio.run(main())
