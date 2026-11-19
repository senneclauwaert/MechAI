"""Database connection, the base class for all tables, and a session helper."""

import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated

from fastapi import Depends
from sqlalchemy import DateTime, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings

engine = create_engine(get_settings().db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Every table gets an id, created_at and updated_at (see the vault: Data model)."""

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    with SessionLocal() as session:
        yield session


# Use this in an endpoint to get a session:  def my_endpoint(db: DB): ...
DB = Annotated[Session, Depends(get_db)]
