"""One error format for the whole API (vault: Error handling, ADR-32).

Every error returns: {"error": {"code": ..., "message": ..., "reference": ...}}
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("mechai")


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def not_found(what: str = "Item") -> AppError:
    return AppError("CORE_NOT_FOUND", f"{what} not found.", 404)


def internal() -> AppError:
    return AppError("CORE_INTERNAL", "Something went wrong on the server.", 500)


def _body(code: str, message: str, request: Request, **extra: object) -> dict:
    reference = getattr(request.state, "request_id", None)
    return {"error": {"code": code, "message": message, "reference": reference, **extra}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(_body(exc.code, exc.message, request), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(p) for p in e["loc"]) for e in exc.errors()]
        body = _body("CORE_VALIDATION_FAILED", "The request is not valid.", request, fields=fields)
        return JSONResponse(body, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # The full stack trace goes to the log, never to the screen.
        log.exception("Unhandled error (reference %s)", getattr(request.state, "request_id", None))
        err = internal()
        return JSONResponse(_body(err.code, err.message, request), status_code=err.status)
