# from fastapi import BackgroundTasks, Header
# from domains.settings import config_settings
#
# import fastapi
# from typing import Optional
#
# import uvicorn
#
# from domains.injestion.routes import router as injestion_router
from langchain.vectorstores.base import VectorStore
#
# app = fastapi.FastAPI()
# vectorstore: Optional[VectorStore] = None
#
#
# app.include_router(
#     injestion_router,
#     prefix="/runner-gpt",
#     #dependencies=[fastapi.Depends(validate_request_token)],
# )
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8081)


# from fastapi import BackgroundTasks, Header, WebSocket, WebSocketDisconnect
# from domains.settings import config_settings
# import fastapi
# from typing import Optional, List
# import uvicorn
# from domains.injestion.routes import router as injestion_router
# from domains.retreival.routes import run_rag, RagUseCase, Message
# from fastapi.middleware.cors import CORSMiddleware
#
# app = fastapi.FastAPI()
# vectorstore: Optional[VectorStore] = None
#
# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["Sec-WebSocket-Accept"],  # Expose WebSocket headers explicitly
# )
#
# app.include_router(
#     injestion_router,
#     prefix="/runner-gpt",
#     #dependencies=[fastapi.Depends(validate_request_token)],
# )
#
# @app.websocket("/ws/run_rag")
# async def websocket_run_rag(
#     websocket: WebSocket,
#     question: str,
#     language: str,
#     chat_context: Optional[List[Message]],
#     namespace: str,
# ):
#     await websocket.accept()
#     try:
#         await run_rag(
#             language=language,
#             chat_context=chat_context,
#             websocket=websocket,
#             namespace=namespace,
#             question=question,
#         )
#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print(f"Error: {e}")
#         await websocket.close(code=1000)


import fastapi
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from domains.settings import config_settings
from domains.injestion.routes import router as injestion_router
from domains.retreival.routes import run_rag, RagUseCase, Message

app = fastapi.FastAPI()
vectorstore: Optional[VectorStore] = None

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Sec-WebSocket-Accept"],  # Ensure WebSocket headers are exposed
)

# Include other API routes
app.include_router(
    injestion_router,
    prefix="/runner-gpt",
)

@app.websocket("/ws/run_rag")
async def websocket_run_rag(websocket: WebSocket):
    """WebSocket endpoint for running RAG model queries."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()  # Receive parameters as JSON

        # Optional: Validate Authentication Token
        # token = data.get("token", None)
        # if token != "your_secret_token":
        #     await websocket.close(code=1008)  # 1008 = Policy Violation
        #     return

        # Run the RAG model
        await run_rag(
            language=data.get("language", "en"),
            chat_context=data.get("chat_context", []),
            websocket=websocket,
            namespace=data.get("namespace", config_settings.PINECONE_DEFAULT_DEV_NAMESPACE),
            question=data.get("question", ""),
        )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close(code=1011)  # 1011 = Internal Server Error


if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8081)

