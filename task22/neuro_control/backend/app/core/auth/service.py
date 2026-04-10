# app/core/auth/service.py

"""
Сервис для выдачи JWT токена.

Назначение:
- Проверка API_TOKEN из .env
- Генерация JWT с указанием времени жизни
- Логирование успешных и неуспешных попыток через app.state.log

Используется в route.py
"""

import time
from fastapi import HTTPException, status, Request
from app.config import settings
import jwt  
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any

async def generate_jwt(api_token: str, request: Request | None = None) -> dict:
    """
    Генерирует JWT для frontend по API_TOKEN.

    Алгоритм:
    1. Сравниваем api_token с TOKEN_API из .env
    2. Если неверно — вызываем HTTPException 401 и логируем через log
    3. Если верно — создаём JWT с payload:
       - iat: issued at
       - exp: expiry timestamp
    4. Логируем успешную выдачу через log
    5. Возвращаем словарь с access_token и expires_in
    """

    client_info = ""
    log_obj = None
    if request:
        log_obj = request.app.state.log
        client_info = f" from {request.client.host}"

    # ----------------------
    # Проверка API_TOKEN
    # ----------------------
    if api_token != settings.API_TOKEN:
        if log_obj:
            await log_obj.log_warning(target="auth", message=f"Unauthorized JWT attempt{client_info}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API_TOKEN"
        )

    # ----------------------
    # Время жизни токена (секунды)
    # ----------------------
    ttl = settings.JWT_EXPIRATION  # например 900 сек

    now = int(time.time())
    payload = {
        "iat": now,
        "exp": now + ttl
    }

    # ----------------------
    # Генерация JWT
    # ----------------------
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

    if log_obj:
        await log_obj.log_info(target="auth", message=f"JWT issued successfully{client_info}, expires in {ttl} sec")

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ttl
    }


async def verify_jwt(token: str, request: Request | None = None) -> dict:
    """
    Проверяет JWT токен.

    1. Декодирует токен
    2. Проверяет подпись и срок действия
    3. Логирует результат
    4. Возвращает payload
    """

    log_obj = None
    client_info = ""

    if request:
        log_obj = request.app.state.log
        client_info = f" from {request.client.host}"

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"]
        )

        if log_obj:
            await log_obj.log_info(
                target="auth",
                message=f"JWT verified successfully{client_info}"
            )

        return payload

    except jwt.ExpiredSignatureError:
        if log_obj:
            await log_obj.log_warning(
                target="auth",
                message=f"Expired JWT{client_info}"
            )
        raise HTTPException(status_code=401, detail="Token expired")

    except jwt.InvalidTokenError:
        if log_obj:
            await log_obj.log_warning(
                target="auth",
                message=f"Invalid JWT{client_info}"
            )
        raise HTTPException(status_code=401, detail="Invalid token")

security = HTTPBearer()

async def get_current_jwt_payload(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    """
    Dependency для защиты маршрутов.

    1. Получает Bearer токен
    2. Проверяет JWT
    3. Возвращает payload
    """

    token = credentials.credentials
    return await verify_jwt(token, request)