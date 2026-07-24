from .cameras import router as cameras_router
from .streams import router as streams_router
from .system import router as system_router

__all__ = ["cameras_router", "streams_router", "system_router"]
