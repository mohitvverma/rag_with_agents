import fastapi
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores.base import VectorStore
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

