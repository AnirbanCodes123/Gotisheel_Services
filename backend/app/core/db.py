"""SQLAlchemy models and session helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Generator

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from .config import get_config

Base = declarative_base()


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, unique=True)
    camera_id = Column(String(256), nullable=False, default="")
    rtsp_url = Column(Text, nullable=False)
    enabled = Column(Boolean, default=True)
    device = Column(String(32), default="")  # override: cuda:0 | cpu | empty=global
    detect_fps = Column(Float, default=0.0)  # 0 = use global
    stream_role = Column(String(32), default="both")  # detect | live | both
    modules = Column(JSON, default=list)  # ["ppe", "crowd", ...]
    extra = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    events = relationship("Event", back_populates="camera", cascade="all, delete-orphan")


class ModelAsset(Base):
    __tablename__ = "model_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False, unique=True)
    filename = Column(String(512), nullable=False)
    module_id = Column(String(64), nullable=True)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_pk = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    camera_name = Column(String(128), default="")
    camera_id = Column(String(256), default="")
    label = Column(String(64), nullable=False)
    module_id = Column(String(64), default="")
    detail = Column(JSON, default=dict)
    bbox = Column(JSON, default=list)
    image_path = Column(String(512), default="")
    thumbnail_path = Column(String(512), default="")
    uploaded = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="events")


_engine = None
_SessionLocal = None


def init_db() -> None:
    global _engine, _SessionLocal
    config = get_config()
    db_url = config["db"]["url"]
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    _engine = create_engine(db_url, connect_args=connect_args, future=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=_engine)


def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        init_db()
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def session_factory() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()
