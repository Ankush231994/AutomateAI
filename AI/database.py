from dotenv import load_dotenv
import os
from sqlmodel import SQLModel, create_engine

load_dotenv()
# SQLite database URL (file-based)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite")
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine) 