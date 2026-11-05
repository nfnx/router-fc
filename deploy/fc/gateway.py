"""FC HTTP adapter; retains the original one-review-call routing contract.

Authentication is enforced by the FC Bearer HTTP trigger, not by this process.
Never expose the container port directly to an unauthenticated public network.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rethos_router.common import RequestError, ReviewError, strict_json
from rethos_router.router import Router

LOG = logging.getLogger("router.fc")


class RouteInput(BaseModel):
    request: str = Field(min_length=1, max_length=32000, description="Original user task text; no prior LLM rewrite.")


class RouteOutput(BaseModel):
    skills: list[str] = Field(description="Stable IDs from the configured skill registry; may be empty.")
    harness: Literal["codex", "claude", "dsh"]
    model: Literal["deepseek", "glm"]


def load_router():
    # The FC-specific config points to NAS model/data and a shared NAS index.
    path = os.environ.get("ROUTER_CONFIG", "/app/config/request_router.fc.json")
    return Router.from_config(path, root=Path("/app"))


class Runtime:
    def __init__(self, factory=load_router):
        self.factory = factory
        self.router = None
        self.failed = False
        self.init_lock = asyncio.Lock()
        self.route_lock = asyncio.Lock()
        self.initialization = None

    async def initialize(self):
        async with self.init_lock:
            if self.failed:
                raise RuntimeError("Initialization failed; replace this instance")
            if self.router is not None:
                return
            self.initialization = asyncio.create_task(asyncio.to_thread(self.factory))
            try:
                self.router = await asyncio.wait_for(asyncio.shield(self.initialization),
                                                    float(os.environ.get("ROUTER_INIT_TIMEOUT", "600")))
            except BaseException:
                self.failed = True
                raise


def create_app(factory=Runtime):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = factory()
        yield

    app = FastAPI(title="Rethos Router on Alibaba FC", version="1.0.0", lifespan=lifespan,
                  description="FC trigger requires Bearer authentication. Successful routing returns exactly three fields.")

    @app.get("/")
    async def root():
        return {"service": "rethos-router-fc", "version": "1.0.0", "route_path": "/route"}

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request):
        runtime = request.app.state.runtime
        if runtime.router is None or runtime.failed:
            raise HTTPException(503, "router_not_ready")
        return {"status": "ready"}

    @app.post("/initialize", include_in_schema=False)
    async def initialize(request: Request):
        try:
            await request.app.state.runtime.initialize()
        except Exception:
            LOG.exception("Router initialization failed")
            raise HTTPException(500, "initialization_failed; inspect FC logs")
        return {"status": "ready"}

    @app.post("/route", response_model=RouteOutput,
              openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": {
                  "type": "object", "required": ["request"], "additionalProperties": False,
                  "properties": {"request": {"type": "string", "minLength": 1, "maxLength": 32000}},
                  "example": {"request": "Compare recent evidence and write a literature review."}}}}}})
    async def route(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(415, "application/json required")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 1_048_576:
                raise HTTPException(413, "request_too_large")
            raw.extend(chunk)
        try:
            body = strict_json(raw.decode("utf-8"))
            if not isinstance(body, dict) or set(body) != {"request"}:
                raise ValueError("Only request is permitted")
            text = body["request"]
            if not isinstance(text, str) or not text.strip() or len(text) > 32000:
                raise ValueError("Invalid task text")
        except (UnicodeError, ValueError):
            raise HTTPException(400, "Body must contain only nonempty request text, at most 32000 characters")
        runtime = request.app.state.runtime
        if runtime.router is None or runtime.failed:
            raise HTTPException(503, "router_not_ready")
        if runtime.route_lock.locked():
            raise HTTPException(429, "instance_busy")
        async with runtime.route_lock:
            task = asyncio.create_task(asyncio.to_thread(runtime.router.route, text))
            try:
                return await asyncio.shield(task)
            except RequestError:
                raise HTTPException(400, "Invalid routing request")
            except ReviewError:
                raise HTTPException(502, "Skill review failed; no retry was made")
            except asyncio.CancelledError:
                # Do not free the GPU/model slot while the thread still runs.
                await asyncio.gather(task, return_exceptions=True)
                raise
            except Exception:
                LOG.exception("Routing failed")
                raise HTTPException(500, "Routing failed")

    schema = app.openapi
    def authenticated_schema():
        result = schema()
        result.setdefault("components", {}).setdefault("securitySchemes", {})["FCBearer"] = {"type": "http", "scheme": "bearer"}
        result["security"] = [{"FCBearer": []}]
        return result
    app.openapi = authenticated_schema
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, workers=1)
