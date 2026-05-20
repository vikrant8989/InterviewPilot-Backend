from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.sessions import router as sessions_router
from app.api.routes.history import router as history_router
from app.core.config import settings
from app.ws.interview_socket import router as interview_ws_router
from app.db.session import engine
from app.db.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly validate env/config at startup.
    _ = settings.model_dump()

    # Dev-only: create tables automatically so the text-interview loop works without migrations.
    if settings.auto_create_tables:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    yield


app = FastAPI(
    title="AI Interview Simulator API",
    version="0.1.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(sessions_router, prefix="/api/sessions", tags=["sessions"])
app.include_router(history_router, prefix="/api/history", tags=["history"])
app.include_router(interview_ws_router, prefix="/ws", tags=["websocket"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

