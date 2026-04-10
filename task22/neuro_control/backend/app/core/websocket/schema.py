# app/core/websocket/schema.py

from typing import Optional, Literal
from pydantic import BaseModel, Field


class WSMessage(BaseModel):
    """
    Универсальное сообщение для WebSocket broadcast.
    Используется всеми модулями (start, progress, end).
    """

    type: Literal["start", "progress", "end"] = Field(..., example="progress")
    code: str = Field(..., example="transcribe")
    title: str = Field(..., example="Транскрибация")
    status: str = Field(..., example="processing")
    result: Optional[str] = Field(None, example="Выполняется транскрибация...")
    progress: Optional[int] = Field(None, example=50)
    is_complete: Optional[bool] = Field(None, example=False)
