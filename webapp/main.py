from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, select
from functools import wraps
import os
import json
import asyncio
import re

print("✅ BACKEND MODULE LOADED")

DATABASE_URL = "sqlite+aiosqlite:///./webapp/.data/localdata.db"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ----------------- Database Model -----------------

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)

class Evaluation(Base):
    __tablename__="evaluations"
    id = Column(Integer, primary_key=True)
    username=Column(String)
    graph_id=Column(String)
    element_id=Column(String)
    order=Column(String)
    accuracy=Column(String)
    
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# ----------------- Login Decorator -----------------

def require_login_html(route_handler):
    @wraps(route_handler)
    async def wrapper(request: Request, *args, **kwargs):
        user_id = request.cookies.get("user_id")
        print(user_id)
        if not user_id:
            return RedirectResponse(url="/")
        return await route_handler(request, *args, **kwargs)
    return wrapper

# ----------------- Routes -----------------

@app.get("/", response_class=HTMLResponse)
async def serve_root(request: Request):
    user_id = request.cookies.get("user_id")
    if user_id:
        index_path = os.path.join(STATIC_DIR, "index.html")
    else:
        index_path = os.path.join(STATIC_DIR, "login.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())




@app.get("/node-evaluation", response_class=HTMLResponse)
@require_login_html
async def serve_node_evaluation(request: Request):
    file_path = os.path.join(STATIC_DIR, "node-evaluation.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/synthetic-evaluation", response_class=HTMLResponse)
@require_login_html
async def serve_synthetic_evaluation(request: Request):
    file_path = os.path.join(STATIC_DIR, "synthetic-evaluation.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/graph-visualization", response_class=HTMLResponse)
@require_login_html
async def serve_graph_visualization(request: Request):
    file_path = os.path.join(STATIC_DIR, "graph-visualization.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/graph-data", response_class=JSONResponse)
async def serve_graph_data():
    return get_graph()

@app.post("/graph-data", response_class=JSONResponse)
async def update_graph_data(request: Request):
    payload = await request.json()
    set_graph(payload)
    return {"status": "ok"}

@app.post("/api/login", response_class=JSONResponse)
async def api_login(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    username = str(data.get("username"))

    result = await db.execute(select(User).where(User.username == username))
    user = result.scalars().first()

    if not user:
        user = User(username=username)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    await generate_eval_index(user.username, db)
    response.set_cookie(key="user_name", value=str(user.username), max_age=3600)
    response.set_cookie(key="user_id", value=str(user.id), max_age=3600)
    return {"status": "ok", "user_id": user.id}

@app.get("/logout")
async def logout(response: Response):
    response.delete_cookie("user_id")
    return {"status": "ok", "message": "Logged out"}

@app.post("/api/selection")
async def receive_selection(request: Request):
    data = await request.json()
    print("Received selection:", data)
    return {"status": "received"}

@app.get("/api/user-data")
async def get_user_data(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not logged in")

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"status": "ok", "data": {"id": user.id, "username": user.username}}

@app.get("/api/get-synthetic-reports")
async def get_synthetic_reports():
    reports = [
        {"id": 1, "title": "Synthetic Case A", "content": "Case A description..."},
        {"id": 2, "title": "Synthetic Case B", "content": "Case B description..."},
        {"id": 3, "title": "Synthetic Case C", "content": "Case C description..."}
    ]
    return {"reports": reports}

@app.post("/api/submit-synthetic-evals")
async def submit_synthetic_evals(request: Request):
    data = await request.json()
    print("Received synthetic evaluations:", data)
    return {"status": "ok"}

# @app.post("/api/submit-batch-eval")
# async def submit_batch_eval(request: Request):
#     # updates the front end file, then move on from this... 
#     # how are you going to update the front end? you neeed to update the front end in the following ways, 

#     data = await request.json()
#     print(data)
#     output_file = os.path.join(BASE_DIR, "evaluations.json")
#     existing = []
#     if os.path.exists(output_file):
#         with open(output_file, "r", encoding="utf-8") as f:
#             existing = json.load(f)
#     existing.extend(data)
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(existing, f, indent=2)
#     return {"status": "ok", "count": len(data)}

@app.post("/api/submit-batch-eval")
async def submit_batch_eval(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    new_evals = [Evaluation(
        username=record['username'],
        graph_id=record.get('graph_id', ''),
        element_id=record['element_id'],
        order=record['order'],
        accuracy=record['accuracy']
    ) for record in data]
    db.add_all(new_evals)
    await db.commit()
    return {"status": "ok", "count": len(new_evals)}

@app.get("/api/get-graph-data")
async def get_graph_data():
    nodes = [
        {"id": 1, "label": "Start"},
        {"id": 2, "label": "Checkup"},
        {"id": 3, "label": "Diagnosis"}
    ]
    edges = [
        {"from": 1, "to": 2},
        {"from": 2, "to": 3}
    ]
    return {"nodes": nodes, "edges": edges}
async def generate_eval_index(username: str, db: AsyncSession) -> str:
    """
    Generate a per-user evaluation index indicating completion status for each graph.
    
    Args:
        username (str): The username for whom to generate the index.
        db (AsyncSession): SQLAlchemy async database session.
    
    Returns:
        str: Path to the generated evaluation index JSON file.
    """
    graph_dir = os.path.join(STATIC_DIR, "graphs")
    output_dir = os.path.join(STATIC_DIR, "user_data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{username}_eval_index.json")

    index_data = []

    graph_files = [
        os.path.join(graph_dir, fname)
        for fname in os.listdir(graph_dir)
        if re.match(r"^graph_\d+\.json$", fname)

    ]

    for graph_path in graph_files:
        graph_id = os.path.splitext(os.path.basename(graph_path))[0]
        graph_data = _load_graph_data(graph_path)
        total_elements = _count_graph_elements(graph_data)

        evaluated_count = await _count_user_evaluations(db, username, graph_id)

        status = "completed" if evaluated_count >= total_elements else "incomplete"
        index_data.append({"graph_id": graph_id, "status": status})

    _write_json_file(output_file, index_data)

    return output_file


def _load_graph_data(graph_path: str) -> dict:
    """
    Load graph data JSON from file, ensure it’s a dict with 'nodes' and 'edges'.
    
    Returns an empty dict if invalid.
    """
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and 'nodes' in data and 'edges' in data:
            return data
        else:
            print(f"⚠️ Invalid graph structure in {graph_path}, skipping.")
            return {"nodes": [], "edges": []}
    except Exception as e:
        print(f"⚠️ Failed to load {graph_path}: {e}")
        return {"nodes": [], "edges": []}



def _count_graph_elements(graph_data: dict) -> int:
    return len(graph_data.get("nodes", [])) + len(graph_data.get("edges", []))


async def _count_user_evaluations(db: AsyncSession, username: str, graph_id: str) -> int:
    """Query number of evaluated elements for a user on a graph."""
    result = await db.execute(
        select(Evaluation).where(
            Evaluation.username == username,
            Evaluation.graph_id == graph_id
        )
    )
    return len(result.scalars().all())


def _write_json_file(filepath: str, data: list) -> None:
    """Write JSON data to a file with pretty formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

@app.on_event("startup")
async def on_startup():
    await init_db()
