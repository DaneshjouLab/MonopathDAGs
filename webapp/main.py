from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import asyncio

from .app_code.background_tasks import background_worker

# Import shared graph state correctly
from .shared_graph_state import get_graph, set_graph

app = FastAPI()

# Handle relative paths
BASE_DIR = os.path.dirname(__file__)        # /path/to/your/webapp
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

@app.get("/graph-data", response_class=JSONResponse)
async def serve_graph_data():
    return get_graph()

@app.post("/graph-data", response_class=JSONResponse)
async def update_graph_data(request: Request):
    payload = await request.json()
    set_graph(payload)
    return {"status": "ok"}

@app.on_event("startup")
async def start_background_worker():
    asyncio.create_task(background_worker())
