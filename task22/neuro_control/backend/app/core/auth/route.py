# app/core/auth/route.py

"""
Роут для выдачи JWT токена.

Назначение:
- Принимает API_TOKEN в body
- Выдаёт JWT и expires_in
- Использует сервис generate_jwt из service.py
- Не требует логина и пароля

Документация статусов:
- 200 OK: токен успешно выдан
- 401 Unauthorized: API_TOKEN неверный
"""

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from .service import generate_jwt
from .schema import TokenResponse

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Получить JWT по API_TOKEN",
    description="Принимает API_TOKEN и возвращает JWT с expires_in"
)
async def get_jwt_token(
    request: Request,
    api_token: str = Body(..., embed=True)
):
    """
    Endpoint для frontend:
    1. Получаем api_token из body
    2. Передаём в сервис generate_jwt
    3. Логируем результат через app.state.log
    4. Возвращаем JSON с токеном и expires_in
    """

    token_data = await generate_jwt(api_token, request=request)
    return JSONResponse(content=token_data)
