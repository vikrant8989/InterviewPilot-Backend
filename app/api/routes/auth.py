from datetime import datetime

import secrets
import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, oauth2_scheme, verify_jwt, verify_password
from app.db.session import get_db
from app.db.models import OAuthIdentity, User

import redis.asyncio as redis_asyncio
from authlib.integrations.starlette_client import OAuth

from app.core.config import settings

router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    accessToken: str
    user: dict


@router.post("/register", response_model=AuthResponse)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    now = datetime.utcnow()
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        name=payload.name,
        auth_provider="local",
        created_at=now,
        updated_at=now,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(subject=user.id)
    return AuthResponse(accessToken=token, user={"id": user.id, "email": user.email, "name": user.name})


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == payload.email))
    user = existing.scalar_one_or_none()
    if not user or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(subject=user.id)
    return AuthResponse(accessToken=token, user={"id": user.id, "email": user.email, "name": user.name})


# OAuth endpoints are intentionally stubbed for now.
oauth = OAuth()


def _require_google_config():
    if not settings.google_client_id or not settings.google_client_secret or not settings.google_redirect_uri:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google OAuth not configured")


def _get_google_client() -> None:
    # Lazy registration so missing env vars don’t crash import time.
    _require_google_config()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


@router.get("/google/start")
async def google_start(request: Request):
    _get_google_client()
    state = secrets.token_urlsafe(32)

    # CSRF prevention (free-first): store state in Redis for short TTL.
    if not settings.redis_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis not configured for OAuth state")
    r = redis_asyncio.from_url(settings.redis_url)
    await r.setex(f"oauth_state:{state}", 600, "1")

    redirect_uri = settings.google_redirect_uri
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)


@router.get("/google/callback")
async def google_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    _get_google_client()

    if not state:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing state")
    r = redis_asyncio.from_url(settings.redis_url)
    stored = await r.get(f"oauth_state:{state}")
    if not stored:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")

    token = await oauth.google.authorize_access_token(request)
    # Prefer OpenID token parsing for stable subject.
    try:
        userinfo = await oauth.google.parse_id_token(request, token)
    except Exception:
        userinfo_resp = await oauth.google.get("userinfo", token=token)
        userinfo = userinfo_resp.json()

    provider_subject = userinfo.get("sub") or userinfo.get("id")
    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("given_name")
    if not provider_subject or not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to fetch Google identity")

    # Persist/link identity.
    async def get_or_create_user(db: AsyncSession) -> User:
        identity = (
            await db.execute(select(OAuthIdentity).where(OAuthIdentity.provider == "google", OAuthIdentity.provider_subject == provider_subject))
        ).scalar_one_or_none()
        if identity:
            return identity.user

        existing_user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if existing_user:
            user = existing_user
        else:
            now = datetime.utcnow()
            user = User(
                email=email,
                password_hash=None,
                name=name,
                auth_provider="google",
                created_at=now,
                updated_at=now,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)

        db.add(
            OAuthIdentity(
                user_id=user.id,
                provider="google",
                provider_subject=str(provider_subject),
                created_at=datetime.utcnow(),
            )
        )
        await db.commit()
        return user

    # Use the normal DB dependency for request scope.
    async def db_flow(db: AsyncSession):
        user = await get_or_create_user(db)
        access_token = create_access_token(subject=user.id)
        return {"accessToken": access_token, "user": {"id": user.id, "email": user.email, "name": user.name}}

    # FastAPI doesn’t allow calling Depends inside; we do manual DB via get_db.
    async for db in get_db():
        result = await db_flow(db)
        access_token = result["accessToken"]
        user = result["user"]
        user_json = json.dumps(user)
        html = f"""
<!doctype html>
<html>
  <head><meta charset="utf-8"/></head>
  <body>
    <script>
      localStorage.setItem("accessToken", "{access_token}");
      localStorage.setItem("user", {user_json});
      window.location.href = "/dashboard";
    </script>
    Logging you in...
  </body>
</html>
"""
        return HTMLResponse(html)


