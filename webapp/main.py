from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, select
from .shared_graph_state import get_graph,set_graph
import os
import asyncio

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

@app.get("/graph-visualization", response_class=HTMLResponse)
async def serve_graph_visualization():
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

    response.set_cookie(key="user_id", value=str(user.id), httponly=True, max_age=3600)
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


@app.on_event("startup")
async def on_startup():
    await init_db()
