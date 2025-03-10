import typing
import uuid

import langchain.callbacks.base
import langchain.schema.output
from loguru import logger

from domains.retreival.chat_response import ChatResponse
from langchain_core.messages import BaseMessage
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult
from fastapi import WebSocket
import datetime


import typing
import uuid
import datetime
from loguru import logger
from fastapi import WebSocket
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult
from langchain_core.messages import BaseMessage

class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket_internal: WebSocket):
        self.websocket = websocket_internal

    async def on_chat_model_start(
            self,
            serialized: typing.Dict[str, typing.Any],
            messages: typing.List[typing.List[BaseMessage]],
            *,
            run_id: uuid.UUID,
            parent_run_id: uuid.UUID | None = None,
            tags: typing.List[str] | None = None,
            metadata: typing.Dict[str, typing.Any] | None = None,
            **kwargs: typing.Any,
    ) -> typing.Any:
        logger.info(f"LLM chain chat model start with serialized: {serialized}\nmessages: {messages}\nkwargs: {kwargs}")

    async def on_llm_start(self,
                           serialized: typing.Dict[str, typing.Any],
                           prompts: typing.List[str],
                           *,
                           run_id: uuid.UUID,
                           parent_run_id: typing.Optional[uuid.UUID] = None,
                           tags: typing.Optional[typing.List[str]] = None,
                           metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
                           **kwargs: typing.Any) -> None:
        logger.info(f'LLM chain started with prompts: {prompts} and kwargs: {kwargs}')

    async def on_llm_new_token(self, token: str, **kwargs: typing.Any) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        resp = {
            "message": token,
            "type": "stream",
            "content_type": None
        }
        formatted_resp = f'{resp}\n{timestamp}'
        await self.websocket.send_json(formatted_resp)

    async def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID,
                         parent_run_id: typing.Optional[uuid.UUID] = None,
                         tags: typing.Optional[typing.List[str]] = None, **kwargs: typing.Any) -> None:
        logger.info(f'LLM chain ended with response: {response}')
#
# class StreamingLLMCallbackHandler(AsyncCallbackHandler):
#     """Callback handler for streaming LLM responses."""
#
#     def __init__(self, websocket_internal: WebSocket):
#         self.websocket = websocket_internal
#
#     async def on_chat_model_start(
#             self,
#             serialized: typing.Dict[str, typing.Any],
#             messages: typing.List[typing.List[BaseMessage]],
#             *,
#             run_id: uuid.UUID,
#             parent_run_id: uuid.UUID | None = None,
#             tags: typing.List[str] | None = None,
#             metadata: typing.Dict[str, typing.Any] | None = None,
#             **kwargs: typing.Any,
#     ) -> typing.Any:
#         logger.info(f"LLM chain chat model start with serialized: {serialized}\nmessages: {messages}\nkwargs: {kwargs}")
#
#     async def on_llm_start(self,
#                            serialized: typing.Dict[str, typing.Any],
#                            prompts: typing.List[str],
#                            *,
#                            run_id: uuid.UUID,
#                            parent_run_id: typing.Optional[uuid.UUID] = None,
#                            tags: typing.Optional[typing.List[str]] = None,
#                            metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
#                            **kwargs: typing.Any) -> None:
#         logger.info(f'LLM chain started with prompts: {prompts} and kwargs: {kwargs}')
#
#     async def on_llm_new_token(self, token: str, **kwargs: typing.Any) -> None:
#         timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
#         resp = {
#             "message": token,
#             "type": "stream",
#             "content_type": None
#         }
#         formatted_resp = f'{resp}\n{timestamp}'
#         await self.websocket.send_json(formatted_resp)
#
#     async def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID,
#                          parent_run_id: typing.Optional[uuid.UUID] = None,
#                          tags: typing.Optional[typing.List[str]] = None, **kwargs: typing.Any) -> None:
#         logger.info(f'LLM chain ended with response: {response}')


# class StreamingLLMCallbackHandler(langchain.callbacks.base.AsyncCallbackHandler):
#     """Callback handler for streaming LLM responses."""
#
#     def __init__(self, websocket_internal):
#         self.websocket = websocket_internal
#
#     async def on_chat_model_start(
#             self,
#             serialized: typing.Dict[str, typing.Any],
#             messages: typing.List[typing.List[BaseMessage]],
#             *,
#             run_id: uuid.UUID,
#             parent_run_id: uuid.UUID | None = None,
#             tags: typing.List[str] | None = None,
#             metadata: typing.Dict[str, typing.Any] | None = None,
#             **kwargs: typing.Any,
#     ) -> typing.Any:
#         logger.info(f"LLM chain chat model start with serialized: {serialized}\nmessages: {messages}\nkwargs: {kwargs}")
#
#     async def on_llm_start(self,
#                            serialized: typing.Dict[str, typing.Any],
#                            prompts: typing.List[str],
#                            *,
#                            run_id: uuid.UUID,
#                            parent_run_id: typing.Optional[uuid.UUID] = None,
#                            tags: typing.Optional[typing.List[str]] = None,
#                            metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
#                            **kwargs: typing.Any) -> None:
#         logger.info(f'LLM chain started with prompts: {prompts} and kwargs: {kwargs}')
#
#     async def on_llm_new_token(self, token: str, **kwargs: typing.Any) -> None:
#         timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
#         resp = {
#             "message": token,
#             "type": "stream",
#             "content_type": None
#         }
#         formatted_resp = f'{resp}\n{timestamp}'
#         await self.websocket.send_json(formatted_resp)
#
#     async def on_llm_end(self, response: langchain.schema.output.LLMResult, *, run_id: uuid.UUID,
#                          parent_run_id: typing.Optional[uuid.UUID] = None,
#                          tags: typing.Optional[typing.List[str]] = None, **kwargs: typing.Any) -> None:
#         logger.info(f'LLM chain ended with response: {response}')

# class StreamingLLMCallbackHandler(langchain.callbacks.base.AsyncCallbackHandler):
#     """Callback handler for streaming LLM responses."""
#
#     def __init__(self, websocket_internal):
#         self.websocket = websocket_internal
#
#     async def on_chat_model_start(
#             self,
#             serialized: typing.Dict[str, typing.Any],
#             messages: typing.List[typing.List[BaseMessage]],
#             *,
#             run_id: uuid.UUID,
#             parent_run_id: uuid.UUID | None = None,
#             tags: typing.List[str] | None = None,
#             metadata: typing.Dict[str, typing.Any] | None = None,
#             **kwargs: typing.Any,
#     ) -> typing.Any:
#         logger.info(f"LLM chain chat model start with serialized: {serialized}\nmessages: {messages}\nkwargs: {kwargs}")
#
#     async def on_llm_start(self,
#                            serialized: typing.Dict[str, typing.Any],
#                            prompts: typing.List[str],
#                            *,
#                            run_id: uuid.UUID,
#                            parent_run_id: typing.Optional[uuid.UUID] = None,
#                            tags: typing.Optional[typing.List[str]] = None,
#                            metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
#                            **kwargs: typing.Any) -> None:
#         logger.info(f'LLM chain started with prompts: {prompts} and kwargs: {kwargs}')
#
#     async def on_llm_new_token(self, token: str, **kwargs: typing.Any) -> None:
#         resp = ChatResponse(message=token, type="stream")
#         await self.websocket.send_json(resp.dict())
#
#     async def on_llm_end(self, response: langchain.schema.output.LLMResult, *, run_id: uuid.UUID,
#                          parent_run_id: typing.Optional[uuid.UUID] = None,
#                          tags: typing.Optional[typing.List[str]] = None, **kwargs: typing.Any) -> None:
#         logger.info(f'LLM chain ended with response: {response}')