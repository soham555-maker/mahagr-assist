"""
Auth, RBAC, and rate limiting for Phase 4.
"""
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import bcrypt
import jwt

# Imported for its SIDE EFFECT: engine/config.py is the one place backend/.env is
# loaded, and it must run before anything reads os.environ (HANDOFF §5 — a .env
# loaded too late is silently ignored). app/api.py happens to import config
# first, but scripts/seed_users.py imports this module directly, so the
# guarantee is made here rather than left to import order.
from engine import config as _config   # noqa: F401

# NO DEFAULT ON PURPOSE. A shipped fallback secret is worse than a crash: every
# deployment that forgets to set JWT_SECRET would sign tokens with a value that
# is public in the repo, so anyone could mint an "IT Admin" token. Failing loudly
# at import is the only version of this that cannot be missed.
#
# Development still works without ceremony: DEV_NO_AUTH=1 bypasses token
# verification entirely, so a dev box that has not set a secret gets a clear
# instruction instead of a silent insecure default.
SECRET_KEY = os.environ.get("JWT_SECRET")
if not SECRET_KEY and os.environ.get("DEV_NO_AUTH") != "1":
    raise RuntimeError(
        "JWT_SECRET is not set. Generate one into backend/.env with:\n"
        "    python -c \"import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(48))\"\n"
        "(or set DEV_NO_AUTH=1 to run the demo without authentication)")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if os.environ.get("DEV_NO_AUTH") == "1":
        # Escape hatch for development
        from app.db import get_user_by_username
        user = get_user_by_username("admin1") # mock admin
        if not user:
            return {"id": "dev", "username": "dev", "role": "IT Admin"}
        return user

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    from app.db import get_user_by_username
    user = get_user_by_username(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def require_role(allowed_roles: list[str]):
    async def role_checker(user: dict = Depends(get_current_user)):
        if os.environ.get("DEV_NO_AUTH") == "1":
            return user
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        return user
    return role_checker

# Simple token bucket rate limiting (in-process)
RATE_LIMIT = 20
class TokenBucket:
    def __init__(self, capacity, fill_rate):
        self.capacity = capacity
        self.fill_rate = fill_rate  # tokens per second
        self.tokens = capacity
        self.last_update = time.time()
        
    def consume(self, amount=1):
        now = time.time()
        # refill
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = now
        
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

# user_id -> TokenBucket. 20 per minute.
buckets = defaultdict(lambda: TokenBucket(RATE_LIMIT, RATE_LIMIT / 60.0))

def check_rate_limit(user_id: str):
    if os.environ.get("DEV_NO_AUTH") == "1":
        return
    if not buckets[user_id].consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment.")
