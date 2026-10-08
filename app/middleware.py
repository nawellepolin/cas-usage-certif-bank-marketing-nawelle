"""Middleware de logging — request_id, latence, pas de PII (cf. M1-B2).

Logs d'accès uniquement (qui a appelé quoi, en combien de temps) — pas de
logs métier/PII, conformément à la règle d'or de la fiche M1-B2.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


def _is_valid_uuid(value: str | None) -> bool:
    if value is None:
        return False
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log chaque requête : method, path, status, latence, request_id."""

    async def dispatch(self, request: Request, call_next):
        client_request_id = request.headers.get("X-Request-ID")
        # On ne réutilise l'id client que s'il ressemble à un UUID — sinon un client
        # pourrait injecter une valeur arbitrairement longue/malformée dans les logs.
        request_id = client_request_id if _is_valid_uuid(client_request_id) else str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            # Même une exception non convertie en HTTPException doit renvoyer une
            # réponse avec X-Request-ID — sinon l'invariant "toute réponse porte ce
            # header" se brise silencieusement sur les erreurs les plus graves.
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.bind(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=500,
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).exception("Unhandled exception in request")
            error_response = JSONResponse(
                status_code=500,
                content={"detail": f"Erreur interne (request_id={request_id})"},
            )
            error_response.headers["X-Request-ID"] = request_id
            return error_response

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        log_level = "INFO" if status_code < 400 else "WARNING" if status_code < 500 else "ERROR"

        logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).log(
            log_level,
            "{method} {path} {status} {latency_ms}ms",
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=latency_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response
