import pprint
import asyncio
from fastapi import FastAPI, HTTPException
from loguru import logger

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from fastapi import APIRouter, BackgroundTasks

from domains.utils import get_chat_model
from domains.agents.models import QueryRequest, OverallState
from domains.agents.tools import qna_tool, information_extraction_tool, summarize_content_tool
from domains.agents.models import QueryRequest as QueryRequestModel


router = APIRouter(tags=["run-agents"])


@router.post("/run_agents")
async def run_agents_api(request: QueryRequestModel):
    try:
        result = await run_agents(request.query, request.id)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_agents(query: str, id):
    memory = MemorySaver()
    agent_executor = create_react_agent(
        model=get_chat_model(model_key="OPENAI_CHAT"),
        tools=[orchestrator_agent],
        checkpointer=memory,
    )
    config = {"configurable": {"thread_id": id}}
    for step in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]},
            config,
            stream_mode="values",
    ):
        final_result = step["messages"][-1].content

    logger.info(final_result)
    return final_result

from typing import Annotated, List, Literal, TypedDict, Optional

from pydantic import BaseModel
from typing import Annotated, List, Literal, TypedDict, Optional
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.managed import IsLastStep, RemainingSteps
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from loguru import logger
from fastapi import APIRouter, HTTPException
from domains.utils import get_chat_model
from domains.agents.models import QueryRequest

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


async def run_agents_2(query: str, id: str):
    # Define custom state schema
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]  # Chat messages
        query: str  # Original query
        result: Optional[str]  # Orchestrator result
        is_last_step: IsLastStep
        remaining_steps: RemainingSteps

    # Create memory for state persistence
    memory = MemorySaver()

    # Create tool node for orchestrator
    async def run_orchestrator(state: AgentState) -> AgentState:
        """Execute orchestrator and update state with result"""
        try:
            result = await orchestrator_agent(state["query"])
            # Handle case where result is a dictionary
            if isinstance(result, dict) and "contents" in result:
                content = result["contents"][0] if isinstance(result["contents"], list) else str(result["contents"])
            else:
                content = str(result)

            return {
                "messages": [AIMessage(content=content)],
                "result": content
            }

        except Exception as e:
            logger.exception("Orchestrator execution failed")
            error_msg = f"Orchestrator failed: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)],
                "result": error_msg
            }
    # Create workflow graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("orchestrator", run_orchestrator)

    # Set entry point and flow
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", END)

    # Compile graph
    agent_executor = workflow.compile(checkpointer=memory)

    # Execute graph
    config = {"configurable": {"thread_id": id}}
    final_result = None

    # Stream results
    async for step in agent_executor.astream(
            {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "result": None
            },
            config
    ):
        if "result" in step:
            final_result = step["result"]

    logger.info(f"Agent result: {final_result}")
    return final_result

async def run_qna_tool(state: OverallState):
    """Fetches documents from the vector database."""
    try:
        request = QueryRequest(query=state['query'])
        documents = await qna_tool(request)
        state["documents"] = documents
        return {"documents": documents}
    except Exception as e:
        logger.exception("Failed to run qna_tool")
        raise e


async def run_information_extraction_tool(state: OverallState):
    """Runs information extraction if vector database results are insufficient."""
    try:
        extracted_docs = await information_extraction_tool(state["query"])
        return {"documents": state["documents"] + extracted_docs}
    except Exception as e:
        logger.exception("Failed to run information_extraction_tool")
        raise e


async def run_summarize_content_tool(state: OverallState):
    """Summarizes the final set of documents."""
    try:
        summary = await summarize_content_tool(state["documents"])
        return {"final_summary": summary}
    except Exception as e:
        logger.exception("Failed to run summarize_content_tool")
        raise e


async def orchestrator_agent(query: str) -> str:
    """
    Orchestrates the workflow based on vector database results.

    - This search the vector database first and if the infromation not found then fetches from internet and summarize them and give the
    consise summary of the documents.

    - All query needs to be handled by this agent.
    - It will call qna_tool, information_extraction_tool, and summarize_content_tool.
    """
    try:
        graph = StateGraph(OverallState)
        graph.add_node("run_qna_tool", run_qna_tool)
        graph.add_node("run_information_extraction_tool", run_information_extraction_tool)
        graph.add_node("run_summarize_content_tool", run_summarize_content_tool)

        graph.add_edge(START, "run_qna_tool")
        graph.add_conditional_edges(
            "run_qna_tool",
            lambda state: "run_summarize_content_tool" if len(
                state["documents"]) >= 5 else "run_information_extraction_tool"
        )
        graph.add_edge("run_information_extraction_tool", "run_summarize_content_tool")
        graph.add_edge("run_summarize_content_tool", END)

        app = graph.compile()
        async for step in app.astream({"query": query, "documents": []}, {"recursion_limit": 10}):
            if "final_summary" in step:
                return step["final_summary"]

        raise Exception("Failed to generate final summary")
    except Exception as e:
        logger.exception("Orchestrator agent failed")
        raise e



if __name__ == "__main__":
    rr = asyncio.run(
        run_agents_2(
            query="Which is the best car in the world and candiate name",
            id="123"
        )
    )
    print(rr)