# api_server.py
import uuid
import threading
from typing import Dict, List, Optional

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage

from generate_response import initialize_resources, generate_response
from user_profile import delete_profile

# Generate FastAPI app
app = FastAPI(title="Car Chatbot API", version="1.0.0")
# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Global resources (load ONCE) =====
llm = None
vector_store = None

# In-memory session store (NOTE: phù hợp dev / 1 worker)
_histories: Dict[str, List[BaseMessage]] = {}
_lock = threading.Lock()


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)
    reset: bool = False  # nếu true thì reset session trước khi chat


class ChatResponse(BaseModel):
    session_id: str
    answer: str


@app.on_event("startup")
def _startup():
    global llm, vector_store
    try:
        llm, vector_store = initialize_resources()
    except Exception as e:
        # nếu init fail thì server vẫn chạy nhưng /chat sẽ báo lỗi rõ ràng
        print(f"[STARTUP ERROR] initialize_resources failed: {e}")


@app.get("/health")
def health():
    ready = (llm is not None) and (vector_store is not None)
    return {"ok": True, "ready": ready}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global llm, vector_store

    if llm is None or vector_store is None:
        raise HTTPException(status_code=500, detail="Server not ready: LLM/VectorStore not initialized.")

    # 1) resolve session id
    session_id = req.session_id or str(uuid.uuid4())

    # 2) optional reset
    if req.reset:
        with _lock:
            _histories.pop(session_id, None)
        delete_profile(session_id)

    # 3) get history
    with _lock:
        history = _histories.get(session_id, []).copy()

    # 4) run sync pipeline in a worker thread
    try:
        answer, updated_history = await anyio.to_thread.run_sync(
            generate_response,
            llm,
            vector_store,
            history,
            req.message,
            session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate_response failed: {e}")

    # 5) persist history
    with _lock:
        _histories[session_id] = updated_history

    return ChatResponse(session_id=session_id, answer=answer)


@app.post("/reset/{session_id}")
def reset(session_id: str):
    with _lock:
        _histories.pop(session_id, None)
    delete_profile(session_id)
    return {"ok": True, "session_id": session_id}


