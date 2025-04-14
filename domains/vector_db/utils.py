from pinecone import Pinecone, ServerlessSpec
from domains.injestion.utils import get_embeddings
from domains.settings import config_settings
from loguru import logger
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone.exceptions import PineconeApiException

from domains.handler import retry_with_custom_backoff


def initialize_pinecone() -> Pinecone:
    try:
        pc = Pinecone(api_key=config_settings.PINECONE_API_KEY)
        logger.info("Successfully initialized Pinecone")
        return pc
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise


@retry_with_custom_backoff()
def validate_and_create_index(
        index_name: str,
        drop_index: bool=config_settings.PINECONE_DROP_INDEX_NAME_STATUS
) -> bool:
    try:
        pc = initialize_pinecone()
        indexes = [index.get("name", None) for index in pc.list_indexes()]

        logger.info(f"Existing indexes: {indexes}")
        def create_index(index_name: str) -> None:
            try:
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric=config_settings.PINECONE_INDEX_METRIC_TYPE,
                    spec=ServerlessSpec(
                        cloud=config_settings.PINECONE_INDEX_CLOUD_NAME,
                        region=config_settings.PINECONE_INDEX_REGION_NAME
                    )
                )
                logger.info(f"Successfully created index: {index_name}")
            except PineconeApiException as e:
                logger.error(f"Pinecone API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to create index: {e}")
                raise

        for idx in indexes:
            if idx is not None and idx == index_name:
                if drop_index:
                    try:
                        logger.info(f"Deleting index: {index_name}")
                        pc.delete_index(index_name)
                        create_index(index_name)
                        logger.info(f"Successfully deleted index: {index_name}")
                        return True

                    except PineconeApiException as e:
                        logger.error(f"Pinecone API error: {e}")
                        return False
                    except Exception as e:
                        logger.error(f"Failed to delete index: {e}")
                        return False
                else:
                    logger.info(f"Index already exists: {index_name}")
                    return True
        create_index(index_name)
        return True

    except Exception as e:
        logger.error(f"Failed to validate and create index: {e}")
        return False


def push_to_database(
        texts: list,
        index_name: str = config_settings.PINECONE_INDEX_NAME,
        namespace: str = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE,
        drop_namespace=config_settings.DELETE_NAMESPACE_STATUS,
):
    try:
        meta_datas = [text.metadata for text in texts]

        if namespace is None:
            namespace = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE

        try:
            if config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
                if drop_namespace:
                    pinecone_vs = initialize_pinecone()
                    loaded_index = pinecone_vs.Index(index_name)

                    if loaded_index is None:
                        logger.error(f"Index {index_name} not found")
                        return False

                    list_namespaces = list(loaded_index.describe_index_stats(namespace=namespace)["namespaces"].keys())


                    logger.info(
                                f"Namespaces in index: {index_name}: {list_namespaces}"
                    )

                    if namespace in list_namespaces:
                        loaded_index.delete(
                            delete_all=True,
                            namespace=namespace,
                        )
                        logger.info(
                            f"Successfully deleted namespace: {namespace} from index: {index_name}"
                        )

                    logger.info(f"Pushing data to Pinecone index: {index_name} and namespace: {namespace}")

                    PineconeVectorStore.from_texts(
                        [t.page_content for t in texts],
                        get_embeddings(model_key="EMBEDDING_MODEL_NAME"),
                        meta_datas,
                        index_name=index_name,
                        namespace=namespace,
                    )
                    logger.info("Successfully pushed data to Pinecone")

                else:
                    logger.info(f"Pushing data to Pinecone index: {index_name} and namespace: {namespace}")
                    PineconeVectorStore.from_texts(
                        [t.page_content for t in texts],
                        get_embeddings(model_key="EMBEDDING_MODEL_NAME"),
                        meta_datas,
                        index_name=index_name,
                        namespace=namespace,
                    )
                    logger.info("Successfully pushed data to Pinecone")


        except Exception as e:
            logger.error(f"Failed to push data to Pinecone: {str(e)}")
            raise Exception(f"Pinecone ingestion failed: {str(e)}")

        logger.info("Vectors have been pushed to database successfully")
        return True

    except Exception as e:
        logger.exception(f"Failed to push vectors to database: {str(e)}")
        raise Exception(f"Vector database operation failed: {str(e)}")
