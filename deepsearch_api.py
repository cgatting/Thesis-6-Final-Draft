from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from functools import lru_cache
from typing import List
import json
import asyncio
from DEEPSEARCH import DocumentRefiner, DEFAULT_SETTINGS


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()


class BroadcastingView:
    def show_error(self, msg: str):
        asyncio.create_task(manager.broadcast(json.dumps({
            "type": "error",
            "message": msg
        })))

    def update_progress(self, percent: float, status: str):
        asyncio.create_task(manager.broadcast(json.dumps({
            "type": "progress",
            "progress": percent,
            "message": status
        })))


class RefineRequest(BaseModel):
    manuscriptText: str


class RefineResponse(BaseModel):
    processedText: str
    bibliographyText: str
    bibtex: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global refiner instance isn't ideal for concurrency if settings change, 
# but for this local tool it's fine. We'll recreate it per request 
# or use a shared one with the broadcasting view.
def get_refiner():
    settings = DEFAULT_SETTINGS
    view = BroadcastingView()
    return DocumentRefiner(settings, view)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/refine", response_model=RefineResponse)
async def refine(req: RefineRequest):
    refiner = get_refiner()
    processed = await refiner.refine_document(req.manuscriptText)
    bibliography_text = refiner.generate_bibliography_text()
    bibtex = refiner.generate_bibtex_content()
    return RefineResponse(
        processedText=processed,
        bibliographyText=bibliography_text,
        bibtex=bibtex,
    )
