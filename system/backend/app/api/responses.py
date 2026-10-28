"""Consistent API response helpers."""

from __future__ import annotations

from typing import Any

from fastapi import Request


def response(data: Any, request: Request, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "data": data,
        "meta": meta or {},
        "request_id": getattr(request.state, "request_id", None),
    }
