from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import hashlib
import secrets

from app.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(password: str) -> str:
    # Use SHA-256 + salt for better compatibility
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"sha256${salt}${password_hash}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt, stored_hash = password_hash.split('$')
        if algorithm != 'sha256':
            return False
        
        computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return computed_hash == stored_hash
    except:
        return False


def create_access_token(*, subject: str, expires_minutes: int | None = None) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": exp}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_jwt(token: str) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        subject: str | None = payload.get("sub")
        if not subject:
            raise credentials_exception
        return {"user_id": subject}
    except JWTError as e:
        raise credentials_exception from e


def extract_bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None
    parts = authorization_header.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]

