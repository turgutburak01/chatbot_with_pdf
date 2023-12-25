
from fastapi import APIRouter

from .routers import health, chat

api_router = APIRouter(prefix="")



api_router.include_router(health.router, prefix="", tags=["health"])


api_router.include_router(chat.router, prefix="", tags=["GPT"])

