import asyncio

from langchain_community.tools.tavily_search import TavilySearchResults

import asyncio
from typing import List, Tuple
from loguru import logger
from langchain_core.documents import Document
from domains.agents.utils import (
    initialize_doc_parser_chain,
    length_function,
    reduce_summary_chain,
    collect_summaries,
    should_collapse,
    map_summaries
)
from domains.agents.models import SummaryState, OverallState, QueryRequest
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langgraph.graph import END, START, StateGraph
from domains.settings import config_settings

from domains.settings import config_settings
from domains.retreival.pinecone_doc_retreival.utils import get_related_docs_with_score
from langchain_core.documents import Document
from domains.retreival.utils import transform_user_query_for_retreival


async def qna_tool(
        request: QueryRequest
) -> List[Document]:
    """
    This function retrieves documents related to a given query from a Pinecone index and filters them based on a minimum relevance score.

    Parameters:
    request (QueryRequest): An object containing the user query and namespace information.

    Returns:
    A list of tuples where each tuple contains:
    A Document object representing a relevant document.
    A float value indicating the relevance score of the document.
    Returns an empty list if no documents meet the minimum relevance score.

    Exceptions:
    None explicitly handled, but failures in retrieval may raise runtime errors.
    :param request:
    :return:
    """

    minimum_score = config_settings.MINIMUM_SCORE
    transformed_query = await transform_user_query_for_retreival(request.query)

    related_docs_with_score = await get_related_docs_with_score(
        index_name=config_settings.PINECONE_INDEX_NAME,
        namespace=request.namespace,
        question=transformed_query,
    )

    documents = [doc for doc in related_docs_with_score if doc[1] > minimum_score]

    final_documents = []

    logger.debug(f"Documents: {documents}")
    if len(documents) != 0:
        for doc in documents:
            final_documents.append(
                Document(
                    metadata=doc[0].metadata,
                    page_content=doc[0].page_content
                )
            )

        return final_documents
        logger.info(f"Final Document {final_documents}")

    else:
        return final_documents


async def information_extraction_tool(query: str) -> list[Document]:
    """
    This function performs a web search using the Tavily search API and extracts relevant content from the search results from web.

    Parameters:
    query (str): The input query string to search for information.

    Returns:
    A list of Document objects, where each document contains:
    metadata (dict): A dictionary with a "url" key storing the URL of the source.
    page_content (str): The textual content extracted from the search result.
    Returns an empty list if no search results are found.

    Exceptions:
    None explicitly handled, but failures in the Tavily API call may raise runtime errors.

    """
    tavily_tool = TavilySearchResults(max_results=2)

    response = await tavily_tool.ainvoke(query)

    final_response = []
    if response:
        for result in response:
            final_response.append(
                Document(
                    metadata={
                        'url': result.get("url", None)
                    },
                    page_content=result.get("content", None)
                )
            )

        return final_response

    else:
        return final_response


async def summarize_content_tool(content: List[Document]) -> str:
    """
    Description:
    This function summarizes a list of documents using a state-based summarization pipeline. It extracts summaries from individual documents, merges them iteratively, and generates a final summary.

    Parameters:
    content (List[Document]): A list of Document objects to be summarized.

    Returns:
    A str containing the final summarized content.
    """
    async def generate_summary(state: SummaryState):
        try:
            logger.info("Generating summary for content")
            response = await initialize_doc_parser_chain().ainvoke(state["content"])
            logger.info("Generated summary successfully")
            return {"summaries": [response]}

        except Exception as e:
            logger.exception("Failed to generate summary")
            raise e

    async def collapse_summaries(state: OverallState):
        try:
            logger.info("Collapsing summaries")
            doc_lists = split_list_of_docs(
                state["collapsed_summaries"], length_function, config_settings.MAX_TOKENS
            )
            results = []
            for doc_list in doc_lists:
                results.append(await acollapse_docs(doc_list, reduce_summary_chain().ainvoke))
            logger.info("Collapsed summaries successfully")
            return {"collapsed_summaries": results}

        except Exception as e:
            logger.exception("Failed to collapse summaries")
            raise e

    async def generate_final_summary(state: OverallState):
        try:
            logger.info("Generating final summary")
            response = await reduce_summary_chain().ainvoke(state["collapsed_summaries"])
            logger.info("Generated final summary successfully")
            return {"final_summary": response}
        except Exception as e:
            logger.exception("Failed to generate final summary")
            raise e

    try:
        graph = StateGraph(OverallState)
        graph.add_node("generate_summary", generate_summary)
        graph.add_node("collect_summaries", collect_summaries)
        graph.add_node("collapse_summaries", collapse_summaries)
        graph.add_node("generate_final_summary", generate_final_summary)

        graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", should_collapse)
        graph.add_conditional_edges("collapse_summaries", should_collapse)
        graph.add_edge("generate_final_summary", END)

        app = graph.compile()

        async for step in app.astream(
                {"contents": [doc.page_content for doc in content]},
                {"recursion_limit": 10},
        ):
            if "generate_final_summary" in step:
                return step['generate_final_summary']["final_summary"]

        else:
            return step

    except Exception as e:
        logger.exception("Summarization tool failed")
        raise e


if __name__ == "__main__":
    res = asyncio.run(
        information_extraction_tool(
            "What is the capital of France?"
        )
    )
    print(res)