import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Create SQLAlchemy engine
# Note: For production-grade resilience, we could add connection pooling options here
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class for models
Base = declarative_base()

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_db_connection() -> bool:
    """
    Attempts to connect to the database and execute a simple query to verify connectivity.
    Returns True if connection succeeds, False otherwise.
    """
    try:
        # We use a short connection timeout if possible, or just attempt a simple execute
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as error:
        # Log the error locally for developer diagnosis
        print(f"Database connection check failed: {error}")
        return False
