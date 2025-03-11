# import pprint
#
# import fastapi
# from pydantic import BaseModel
#
# from domains.retreival.rag_util import send_message_over_websocket
# from domains import retreival
# from domains.retreival.utils import transform_user_query_for_retreival
# from fastapi import WebSocket
# from langchain_core.documents import Document
#
# from domains.retreival.pinecone_doc_retreival.utils import get_related_docs_without_context
# from typing import Optional, List, Dict, Callable
# from domains.retreival.initialize_memory import initialise_memory_from_chat_context
# from domains.settings import config_settings
#
# from domains.retreival.utils import get_chat_model_with_streaming
# from langchain.chains import question_answering as q_a
#
# from loguru import logger
# from domains.retreival.models import RagUseCase, RAGGenerationResponse
# from domains.retreival.models import Message
# from domains.retreival.prompts import PROMPT_PREFIX_QNA, PROMPT_SUFFIX, initialise_doc_search_prompt_template
# from langchain.prompts import PromptTemplate
#
#
#
# def run_rag(
#     question: str,
#     language: str,
#     chat_context: Optional[List[Message]],
#     websocket: fastapi.WebSocket,
#     namespace: str,
# ):
#     DEFAULT_MIN_SCORE = 0.8
#     # initialise minimum score
#     minimum_score = (
#         config_settings.MINIMUM_SCORE
#         or DEFAULT_MIN_SCORE
#     )
#
#     # initialise prompt
#     prompt_qna_ask_question = initialise_doc_search_prompt_template(
#         PROMPT_PREFIX_QNA, PROMPT_SUFFIX
#     )
#
#     # initialise memory
#     memory = initialise_memory_from_chat_context(
#         chat_context
#     )
#
#     if not namespace or namespace == "":
#         namespace = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE
#
#     return rag_with_streaming(
#         websocket=websocket,
#         language=language,
#         question=question,
#         minimum_score=minimum_score,
#         prompt_template_ask_question=prompt_qna_ask_question,
#         memory=memory,
#         namespace=namespace,
#     )
#
#
# async def rag_with_streaming(
#     websocket: fastapi.WebSocket,
#     question: str,
#     language: str,
#     minimum_score: float,
#     prompt_template_ask_question: PromptTemplate,
#     memory,
#     namespace: str,
#     use_case: RagUseCase = RagUseCase.DEFAULT,
#     citations_count=0,
# ):
#     try:
#         if not citations_count:
#             citations_count = config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE
#
#         # citations_toggle = config_settings.CITATIONS_TOGGLE
#         index_name = config_settings.PINECONE_INDEX_NAME
#
#         await send_message_over_websocket(
#             websocket, "", retreival.MESSAGE_TYPE_START
#         )
#
#         await send_message_over_websocket(
#             websocket,
#             "",
#             retreival.MESSAGE_TYPE_START,
#             content_type=retreival.CONTENT_TYPE_OPTIMISED_QUESTION,
#         )
#
#         retreival_query = await transform_user_query_for_retreival(
#             question, "OPTIMIZED_QUESTION_MODEL"
#         )
#
#         logger.debug(f"Retreival query: {retreival_query}")
#
#         related_docs_with_score = []
#         if retreival_query or retreival_query != "None":
#             related_docs_with_score = await get_related_docs_without_context(
#                 index_name,
#                 namespace,
#                 retreival_query,
#             )
#
#             # logger.debug(
#             #     "\n".join(
#             #         [
#             #             f'File Name: {doc.metadata.get("file_name", "")}, Score: {score}, Page Number: {doc.metadata.get("page", 0)}'
#             #             for doc, score in related_docs_with_score
#             #         ]
#             #     )
#             # )
#         logger.info(f"Related docs without context: {related_docs_with_score}")
#         await send_message_over_websocket(
#             websocket,
#             "",
#             retreival.MESSAGE_TYPE_END,
#             content_type=retreival.CONTENT_TYPE_OPTIMISED_QUESTION,
#         )
#
#         route = RagUseCase.DEFAULT
#
#         rag_generation: RAGGenerationResponse = await generator_routing(
#             memory,
#             language,
#             retreival_query,
#             prompt_template_ask_question,
#             websocket,
#             route,
#             citations_count,
#             minimum_score,
#             related_docs_with_score,
#         )
#
#         await send_message_over_websocket(
#             websocket, "", retreival.MESSAGE_TYPE_END, content_type=retreival.CONTENT_TYPE_ANSWER
#         )
#
#
#         await send_message_over_websocket(
#             websocket, "", retreival.MESSAGE_TYPE_END
#         )
#
#     except Exception as e:
#         print(f"Error: {e}")
#         await send_message_over_websocket(
#             websocket, f"Error: {e}", retreival.MESSAGE_TYPE_ERROR
#         )
#         logger.error(f"Error: {e}")
#         return
#
#
# async def generator_routing(
#     memory,
#     language: str,
#     optimised_question: str,
#     prompt_template_ask_question: PromptTemplate,
#     websocket: WebSocket,
#     route: str,
#     citations_count: int,
#     minimum_score: float,
#     related_docs_with_score: list[tuple[Document, float]] = [],
# ) -> RAGGenerationResponse:
#
#     if route == RagUseCase.DEFAULT:
#         response = await run_doc_retrieval_flow(
#             memory,
#             optimised_question,
#             prompt_template_ask_question,
#             related_docs_with_score[:citations_count],
#             websocket,
#             minimum_score,
#             language,
#         )
#
#     return response
#
# from langchain_core.output_parsers import StrOutputParser
#
# async def run_doc_retrieval_flow(
#     memory,
#     optimised_question: str,
#     prompt_template_ask_question: PromptTemplate,
#     related_docs_with_score: list[tuple[Document, float]],
#     websocket: WebSocket,
#     minimum_score: float,
#     language: str,
# ) -> RAGGenerationResponse:
#
#     # document_count = len(
#     #     [doc for doc in related_docs_with_score if doc[1] > minimum_score]
#     # )
#
#     document_count = len(related_docs_with_score)
#
#     llm =get_chat_model_with_streaming(
#         websocket, model_key=config_settings.LLMS.get("OPENAI_CHAT")
#     )
#     if llm:
#         llm_chain = prompt_template_ask_question | llm | StrOutputParser()
#
#         response = await llm_chain.ainvoke(
#             {
#                 "question": optimised_question,
#                 "chat_history": memory.buffer_as_str,
#                 "doc_count": str(document_count),
#                 "context": related_docs_with_score,
#                 "language": language
#             },
#         )
#         logger.info(f"Response: {response}")
#
#         return RAGGenerationResponse(
#             answer=response
#         )
from enum import Enum
from typing import List, Optional, Tuple, Any
import asyncio
from fastapi import WebSocket, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from domains.retreival.rag_util import send_message_over_websocket
from domains.retreival.utils import (
    transform_user_query_for_retreival,
    get_chat_model_with_streaming,
)
from domains.retreival.pinecone_doc_retreival.utils import get_related_docs_without_context
from domains.retreival.initialize_memory import initialise_memory_from_chat_context
from domains.settings import config_settings
from domains.retreival.models import RagUseCase, RAGGenerationResponse, Message
from domains.retreival.prompts import (
    PROMPT_PREFIX_QNA,
    PROMPT_SUFFIX,
    initialise_doc_search_prompt_template,
)
from domains import retreival


class RAGError(Exception):
    """Base exception for RAG-related errors"""
    pass


class WebSocketConnectionError(RAGError):
    """Raised when WebSocket communication fails"""
    pass


class DocumentRetrievalError(RAGError):
    """Raised when document retrieval fails"""
    pass


async def run_rag(
        question: str,
        language: str,
        chat_context: Optional[List[Message]] = None,
        websocket: Optional[WebSocket] = None,
        namespace: Optional[str] = None,
) -> RAGGenerationResponse:
    """
    Main RAG pipeline function.

    Args:
        question: User's question
        language: Response language
        chat_context: Previous chat history
        websocket: WebSocket connection for streaming
        namespace: Pinecone namespace

    Returns:
        RAGGenerationResponse object

    Raises:
        RAGError: If the RAG pipeline fails
    """
    try:
        # Configuration
        minimum_score = config_settings.MINIMUM_SCORE or 0.8
        namespace = namespace or config_settings.PINECONE_DEFAULT_DEV_NAMESPACE

        # Initialize components
        prompt_qna = initialise_doc_search_prompt_template(
            PROMPT_PREFIX_QNA,
            PROMPT_SUFFIX
        )
        memory = initialise_memory_from_chat_context(chat_context or [])

        return await rag_with_streaming(
            websocket=websocket,
            language=language,
            question=question,
            minimum_score=minimum_score,
            prompt_template_ask_question=prompt_qna,
            memory=memory,
            namespace=namespace,
        )
    except Exception as e:
        logger.exception("RAG pipeline failed")
        raise RAGError(f"RAG pipeline failed: {str(e)}")


async def rag_with_streaming(
        websocket: Optional[WebSocket],
        question: str,
        language: str,
        minimum_score: float,
        prompt_template_ask_question: PromptTemplate,
        memory: Any,
        namespace: str,
        use_case: RagUseCase = RagUseCase.DEFAULT,
        citations_count: int = None,
) -> RAGGenerationResponse:
    """
    RAG pipeline with streaming support.
    """
    citations_count = citations_count or config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE
    index_name = config_settings.PINECONE_INDEX_NAME

    try:
        if websocket:
            await send_message_over_websocket(
                websocket, "", retreival.MESSAGE_TYPE_START
            )

        # Get optimized retrieval query
        retreival_query = await transform_user_query_for_retreival(
            question, "OPTIMIZED_QUESTION_MODEL"
        )

        if not retreival_query or retreival_query == "None":
            logger.warning("Empty retrieval query")
            return RAGGenerationResponse(answer="")

        # Retrieve related documents
        related_docs = await get_related_docs_without_context(
            index_name,
            namespace,
            retreival_query,
        )

        logger.debug(f"Retrieved {len(related_docs)} documents")

        # Generate response
        response = await generator_routing(
            memory=memory,
            language=language,
            optimised_question=retreival_query,
            prompt_template_ask_question=prompt_template_ask_question,
            websocket=websocket,
            route=RagUseCase.DEFAULT,
            citations_count=citations_count,
            minimum_score=minimum_score,
            related_docs_with_score=related_docs,
        )

        if websocket:
            await send_message_over_websocket(
                websocket, "", retreival.MESSAGE_TYPE_END
            )

        return response

    except Exception as e:
        logger.exception("Streaming RAG pipeline failed")
        if websocket:
            await send_message_over_websocket(
                websocket, str(e), retreival.MESSAGE_TYPE_ERROR
            )
        raise RAGError(f"Streaming RAG failed: {str(e)}")


async def generator_routing(
        memory: Any,
        language: str,
        optimised_question: str,
        prompt_template_ask_question: PromptTemplate,
        websocket: Optional[WebSocket],
        route: RagUseCase,
        citations_count: int,
        minimum_score: float,
        related_docs_with_score: List[Tuple[Document, float]],
) -> RAGGenerationResponse:
    """
    Routes the generation request to appropriate handler based on use case.
    """
    if route != RagUseCase.DEFAULT:
        raise ValueError(f"Unsupported route: {route}")

    return await run_doc_retrieval_flow(
        memory=memory,
        optimised_question=optimised_question,
        prompt_template_ask_question=prompt_template_ask_question,
        related_docs_with_score=related_docs_with_score[:citations_count],
        websocket=websocket,
        minimum_score=minimum_score,
        language=language,
    )


async def run_doc_retrieval_flow(
        memory: Any,
        optimised_question: str,
        prompt_template_ask_question: PromptTemplate,
        related_docs_with_score: List[Tuple[Document, float]],
        websocket: Optional[WebSocket],
        minimum_score: float,
        language: str,
) -> RAGGenerationResponse:
    """
    Executes the document retrieval and response generation flow.
    """
    try:
        document_count = len(related_docs_with_score)

        llm = get_chat_model_with_streaming(
            websocket,
            model_key=config_settings.LLMS.get("OPENAI_CHAT")
        )
        if not llm:
            raise ValueError("Failed to initialize language model")

        llm_chain = prompt_template_ask_question | llm | StrOutputParser()

        response = await llm_chain.ainvoke({
            "question": optimised_question,
            "chat_history": memory.buffer_as_str,
            "doc_count": str(document_count),
            "context": related_docs_with_score,
            "language": language
        })

        logger.debug(f"Generated response length: {len(response)}")

        return RAGGenerationResponse(answer=response)

    except Exception as e:
        logger.exception("Document retrieval flow failed")
        raise DocumentRetrievalError(f"Document retrieval failed: {str(e)}")