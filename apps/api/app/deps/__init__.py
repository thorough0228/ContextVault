"""FastAPI dependencies shared across routers."""

from app.deps.auth import (
    get_current_user,
    get_current_user_id,
)

__all__ = ["get_current_user", "get_current_user_id"]