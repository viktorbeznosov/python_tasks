# app/core/auth/schema.py

"""
Pydantic-схемы для модуля auth.

Назначение:
- Структура ответа при запросе JWT по API_TOKEN
- Используется в route.py для response_model

Документация статусов:
- 200 OK: токен успешно выдан
- 401 Unauthorized: неверный API_TOKEN
"""

from pydantic import BaseModel


class TokenResponse(BaseModel):
    """
    Ответ при успешной выдаче JWT.

    Поля:
    - access_token: str — сам JWT токен
    - token_type: str — тип токена, всегда "bearer"
    - expires_in: int — время жизни токена в секундах
    """
    access_token: str
    token_type: str = "bearer"
    expires_in: int
