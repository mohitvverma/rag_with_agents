import fastapi

from domains.retreival.models import RagUseCase
from domains.retreival.rag_util import send_message_over_websocket
from domains import retreival
from domains.retreival.utils import transform_user_query_for_retreival
from fastapi import WebSocket
from langchain_core.documents import Document

from domains.retreival.pinecone_doc_retreival.utils import get_related_docs_without_context
from typing import Optional, List, Dict, Callable
from pydantic import BaseModel
from domains.retreival.initialize_memory import initialise_memory_from_chat_context
from domains.settings import config_settings

from domains.retreival.utils import get_chat_model_with_streaming
from langchain.chains import question_answering as q_a

from loguru import logger
from domains.retreival.models import RagUseCase, RAGGenerationResponse
from domains.retreival.models import Message


PROMPT_PREFIX_QNA = """You are an AI assistant for a data research team that provides comprehensive answers based on the given context.\nUse the following pieces of context to answer the users question.\nIf you don't know the answer, just say that 'I tried to look for this information, but could not find something highly relevant to your query.', don't try to make up an answer.
"""

PROMPT_SUFFIX = """
Respond in "{language}" only, without mixing any other language.
If you don't find relevant answer, respond like 'I tried to look for this information, but could not find something highly relevant to your query.', and ask user for some other query.
"""

DEFAULT_PROMPT_POST_SUFFIX = """{chat_history}
Human: {question}
AI:"""

from langchain.prompts import PromptTemplate


def initialise_doc_search_prompt_template(prefix, suffix):
    prefix = prefix or config_settings.DEFAULT_PROMPT_PREFIX
    suffix = suffix or config_settings.DEFAULT_PROMPT_SUFFIX

    context_arg = """ {doc_count} documents found\n\n
    Related documents : \n {context}
    """

    input_variables = ["chat_history", "question", "doc_count", "context"]

    prompt_template = "\n\n".join([prefix, context_arg, suffix, DEFAULT_PROMPT_POST_SUFFIX])

    return PromptTemplate(template=prompt_template, input_variables=input_variables)


def run_rag(
    question: str,
    language: str,
    chat_context: Optional[List[Message]],
    websocket: fastapi.WebSocket,
    namespace: str,
):
    DEFAULT_MIN_SCORE = 0.8
    # initialise minimum score
    minimum_score = (
        config_settings.MINIMUM_SCORE
        or DEFAULT_MIN_SCORE
    )

    # initialise prompt
    prompt_qna_ask_question = initialise_doc_search_prompt_template(
        PROMPT_PREFIX_QNA, PROMPT_SUFFIX
    )

    # initialise memory
    memory = initialise_memory_from_chat_context(
        chat_context
    )

    if not namespace or namespace == "":
        namespace = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE

    return rag_with_streaming(
        websocket=websocket,
        language=language,
        question=question,
        minimum_score=minimum_score,
        prompt_template_ask_question=prompt_qna_ask_question,
        memory=memory,
        namespace=namespace,
    )


async def rag_with_streaming(
    websocket: fastapi.WebSocket,
    question: str,
    language: str,
    minimum_score: float,
    prompt_template_ask_question: PromptTemplate,
    memory,
    namespace: str,
    use_case: RagUseCase = RagUseCase.DEFAULT,
    citations_count=0,
):
    try:
        if not citations_count:
            citations_count = config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE

        # citations_toggle = config_settings.CITATIONS_TOGGLE
        index_name = config_settings.PINECONE_INDEX_NAME

        await send_message_over_websocket(
            websocket, "", retreival.MESSAGE_TYPE_START
        )

        await send_message_over_websocket(
            websocket,
            "",
            retreival.MESSAGE_TYPE_START,
            content_type=retreival.CONTENT_TYPE_OPTIMISED_QUESTION,
        )

        retreival_query = await transform_user_query_for_retreival(
            question, "OPTIMIZED_QUESTION_MODEL"
        )

        related_docs_with_score = []
        if retreival_query or retreival_query != "None":
            related_docs_with_score = get_related_docs_without_context(
                index_name,
                namespace,
                retreival_query,
            )

            logger.debug(
                "\n".join(
                    [
                        f'File Name: {doc.metadata.get("file_name", "")}, Score: {score}, Page Number: {doc.metadata.get("page", 0)}'
                        for doc, score in related_docs_with_score
                    ]
                )
            )
        logger.info(f"Related docs without context: {related_docs_with_score}")
        await send_message_over_websocket(
            websocket,
            "",
            retreival.MESSAGE_TYPE_END,
            content_type=retreival.CONTENT_TYPE_OPTIMISED_QUESTION,
        )

        route = RagUseCase.DEFAULT

        rag_generation: RAGGenerationResponse = await generator_routing(
            memory,
            language,
            retreival_query,
            prompt_template_ask_question,
            websocket,
            route,
            citations_count,
            minimum_score,
            related_docs_with_score,
        )

        await send_message_over_websocket(
            websocket, "", retreival.MESSAGE_TYPE_END, content_type=retreival.CONTENT_TYPE_ANSWER
        )


        await send_message_over_websocket(
            websocket, "", retreival.MESSAGE_TYPE_END
        )

    except Exception as e:
        print(f"Error: {e}")
        await send_message_over_websocket(
            websocket, f"Error: {e}", retreival.MESSAGE_TYPE_ERROR
        )
        logger.error(f"Error: {e}")
        return


async def generator_routing(
    memory,
    language: str,
    optimised_question: str,
    prompt_template_ask_question: PromptTemplate,
    websocket: WebSocket,
    route: str,
    citations_count: int,
    minimum_score: float,
    related_docs_with_score: list[tuple[Document, float]] = [],
) -> RAGGenerationResponse:

    if route == RagUseCase.DEFAULT:
        response = await run_doc_retrieval_flow(
            memory,
            optimised_question,
            prompt_template_ask_question,
            related_docs_with_score[:citations_count],
            websocket,
            minimum_score,
            language,
        )

    return response


async def run_doc_retrieval_flow(
    memory,
    optimised_question: str,
    prompt_template_ask_question: PromptTemplate,
    related_docs_with_score: list[tuple[Document, float]],
    websocket: WebSocket,
    minimum_score: float,
    language: str,
) -> RAGGenerationResponse:

    document_count = len(
        [doc for doc in related_docs_with_score if doc[1] > minimum_score]
    )

    llm =get_chat_model_with_streaming(
        websocket, model_key=config_settings.LLMS.get("OPENAI_CHAT")
    )
    logger.info(f"LLM: {llm.invoke("Hellomwolrd")}")
    if llm:
        llm_chain = q_a.load_qa_chain(
            llm=llm,
            prompt=prompt_template_ask_question,
            verbose=True,
        )

        response = await llm_chain.ainvoke(
            {
                "question": optimised_question,
                "chat_history": memory.buffer_as_str,
                "doc_count": str(document_count),
                "input_documents": related_docs_with_score,
                "language": language
            },
        )
        logger.info(f"Response: {response}")

        return RAGGenerationResponse(
            answer=response.get("output_text","")
        )