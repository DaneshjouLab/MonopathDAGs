from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, select
from .shared_graph_state import get_graph,set_graph
import os
import asyncio


#try to minimize backend computations and move ot the front end to reduce costs and such.....


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


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ----------------- Routes -----------------

@app.get("/", response_class=HTMLResponse)
async def serve_root(request: Request):
    user_id = request.cookies.get("user_id")
    if user_id:
        index_path = os.path.join(STATIC_DIR, "index.html")
    else:
        index_path = os.path.join(STATIC_DIR, "login.html")

    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/node-evaluation", response_class=HTMLResponse)
async def serve_node_evaluation():
    file_path = os.path.join(STATIC_DIR, "node-evaluation.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/synthetic-evaluation", response_class=HTMLResponse)
async def serve_synthetic_evaluation():
    file_path = os.path.join(STATIC_DIR, "synthetic-evaluation.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
##update this 
@app.get("/graph-visualization", response_class=HTMLResponse)
async def serve_graph_visualization():
    file_path = os.path.join(STATIC_DIR, "graph-visualization.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


#this needs to be fixed to graph visualization... cause this is wrong, 
@app.get("/graph-data", response_class=JSONResponse)
async def serve_graph_data():
    return get_graph()

@app.post("/graph-data", response_class=JSONResponse)
async def update_graph_data(request: Request):
    payload = await request.json()
    set_graph(payload)
    return {"status": "ok"}


#this is fine, 
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

    response.set_cookie(key="user_id", value=str(user.id), httponly=True, max_age=3600)
    return {"status": "ok", "user_id": user.id}

#fine
@app.get("/logout")
async def logout(response: Response):
    response.delete_cookie("user_id")
    return {"status": "ok", "message": "Logged out"}


#ehh naming schema fro routes is off. 
@app.post("/api/selection")
async def receive_selection(request: Request):
    data = await request.json()
    print("Received selection:", data)
    return {"status": "received"}

# the user-data login thing, imagine how user db should work... nodes and edge stored as objects for the thing, 
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

# generate synthetic repors. 
@app.get("/api/get-synthetic-reports")
async def get_synthetic_reports():
    # Example static list; replace with DB or dynamic source
    reports = [
        {"id": 1, "title": "Synthetic Case A", "content": "Case A description..."},
        {"id": 2, "title": "Synthetic Case B", "content": "Case B description..."},
        {"id": 3, "title": "Synthetic Case C", "content": "Case C description..."}
    ]
    return {"reports": reports}

#post should recieve and update the db... the writes should be easy.... loading should 
@app.post("/api/submit-synthetic-evals")
async def submit_synthetic_evals(request: Request):
    data = await request.json()
    print("Received synthetic evaluations:", data)
    # TODO: save to DB or process
    return {"status": "ok"}





@app.get("/api/get-graph-data")
async def get_graph_data():
    # Replace with real graph source or DB
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


# @app.get("/api/get-node/{node_id}")
# async def get_node(node_id: int):
#     # Replace with real node details + HTML content
#     return {
#         "description": f"Details about node {node_id}",
#         "html_content": f"<p>Case content for node {node_id}</p>"
#     }
import re

def clean_html(html):
    html = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<link.*?>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'<img.*?>', '', html, flags=re.IGNORECASE)
    return html

@app.get("/api/get-node/{node_id}")
async def get_node(node_id: int):
    html_path = "./samples/html/Small Cell Lung Cancer in the Course of Idiopathic Pulmonary Fibrosis—Case Report and Literature Review - PMC.html"
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            
            html_content = f.read()
            html_stuff=clean_html(html_content)
    except FileNotFoundError:
        html_content = f"<p>File not found for node {node_id}</p>"

    return {
        "description": f"Details about node {node_id}",
        "html_content": html_stuff
    }

@app.post("/api/submit-node-eval")
async def submit_node_eval(request: Request):
    data = await request.json()
    print("Received node evaluation:", data)
    # TODO: Save to DB or process
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup():
    await init_db()
