from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from functools import lru_cache
from typing import List
import json
import asyncio
import os
from contextlib import asynccontextmanager
from DEEPSEARCH import DocumentRefiner, DEFAULT_SETTINGS, NLPProcessor


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


# Global NLP Processor to avoid reloading models on every request
nlp_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp_processor
    print("Loading ML models...")
    # Initialize NLP Processor once
    nlp_processor = NLPProcessor(DEFAULT_SETTINGS)
    print("ML models loaded.")
    yield
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_refiner():
    settings = DEFAULT_SETTINGS
    view = BroadcastingView()
    # Pass the global NLP processor
    return DocumentRefiner(settings, view, nlp_processor=nlp_processor)


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
    refiner.view.update_progress(0.0, "Starting DeepSearch refinement...")
    processed = await refiner.refine_document(req.manuscriptText)
    refiner.view.update_progress(0.98, "Generating bibliography and BibTeX...")
    bibliography_text = refiner.generate_bibliography_text()
    bibtex = refiner.generate_bibtex_content()
    refiner.view.update_progress(1.0, "DeepSearch complete")
    return RefineResponse(
        processedText=processed,
        bibliographyText=bibliography_text,
        bibtex=bibtex,
    )

# Serve static files for React frontend
if os.path.exists("dist"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        potential_path = os.path.join("dist", full_path)
        if os.path.isfile(potential_path):
            return FileResponse(potential_path)
        return FileResponse("dist/index.html")
