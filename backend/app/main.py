"""Entry point of the api container: builds the FastAPI app.

Sets up logging, the request id middleware and the error handlers, then mounts the
router of every module under /api/v1. Each module adds its router here with one line.
"""

from fastapi import FastAPI

from app.core.errors import register_error_handlers
from app.core.logging import add_request_id_middleware, setup_logging
from app.core.router import health
from app.core.router import router as core_router
from app.modules.knowledge.router import router as knowledge_router

# Logging and error handling must be in place before any request comes in.
setup_logging()

app = FastAPI(
    title="MechAi API", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json"
)
add_request_id_middleware(app)
register_error_handlers(app)

# Health check at the root as well, for Docker and nginx (no /api/v1 prefix).
app.add_api_route("/health", health, methods=["GET"])

# Module routers, all under the versioned API prefix.
API = "/api/v1"
app.include_router(core_router, prefix=API)
app.include_router(knowledge_router, prefix=f"{API}/knowledge")
