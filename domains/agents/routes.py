import pprint
import asyncio
from fastapi import FastAPI, HTTPException
from loguru import logger

from loguru import logger
from fastapi import APIRouter, HTTPException
from domains.utils import get_chat_model
from domains.agents.models import QueryRequest

from langchain_core.documents import Document
from langgraph.graph.message import add_messages

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
async def react_orchestrator(query: str, id: str):
    # Create the tools
    tools = [qna_tool, information_extraction_tool, summarize_content_tool]

    # Get the model
    model = get_chat_model(model_key="OPENAI_CHAT")

    # Create memory saver
    memory = MemorySaver()

    # Define the system prompt
    system_prompt = """
    You are an orchestrator agent responsible for handling all queries using a structured retrieval and summarization process. Follow this workflow for every query:

    1. Search the vector database tool for relevant answers.
    2. If no relevant answer is found, fetch information from the internet using information tool.
    3. Summarize the retrieved documents into a concise and clear response.

    Additionally:
    - Always specify which tools were used to retrieve and summarize the information.
    - Do not make assumptions—base responses strictly on retrieved data.
    Ensure accuracy, relevance, and brevity in all responses.
    """

    # Create the agent with built-in flow
    agent_executor = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=system_prompt,
        checkpointer=memory,
    )

    # Execute with config
    config = {"configurable": {"thread_id": id}}
    final_result = None

    # Stream results
    async for step in agent_executor.astream(
            {
                "messages": [HumanMessage(content=query)],
            },
            config
    ):
        if step.get("agent"):
            final_result = step.get("agent", {}).get("messages", [])[-1].content

    logger.info(f"Agent result: {final_result}")
    return final_result
