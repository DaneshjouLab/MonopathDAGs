import asyncio
import csv
import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import select, Column, Integer, String

# ------------------ Setup ------------------

DATABASE_URL = "sqlite+aiosqlite:///./webapp/.data/localdata.db"
EXPORT_PATH = "./webapp/.data/evaluation_dump_nodes.csv"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

# ------------------ Model ------------------

class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    graph_id = Column(String)
    element_id = Column(String)
    order = Column(String)
    accuracy = Column(String)

# ------------------ Export Function ------------------

async def export_evaluation_to_csv():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Evaluation))
        rows = result.scalars().all()

        os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
        with open(EXPORT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "username", "graph_id", "element_id", "order", "accuracy"])
            for row in rows:
                writer.writerow([
                    row.id,
                    row.username,
                    row.graph_id,
                    row.element_id,
                    row.order,
                    row.accuracy
                ])

        print(f"✅ Exported {len(rows)} rows to {EXPORT_PATH}")

# ------------------ Entry ------------------

if __name__ == "__main__":
    asyncio.run(export_evaluation_to_csv())
