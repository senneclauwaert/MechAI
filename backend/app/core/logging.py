"""JSON log lines with a request id, so an error reference leads to the right line."""

import json
import logging
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request

# The id of the request being handled, so every log line of that request carries it.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


# One JSON object per log line: easy to search and to read by log tools.
class JsonFormatter(logging.Formatter):
    # Turn one log record into a JSON line, with the stack trace when there is one.
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line)


# Send all logs (ours and the libraries') to stdout as JSON, level INFO.
def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


# Give every request a short id, put it in the logs and in the X-Request-ID header.
def add_request_id_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = "req_" + uuid.uuid4().hex[:8]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response
