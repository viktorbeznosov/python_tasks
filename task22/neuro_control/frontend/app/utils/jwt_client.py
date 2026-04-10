# app/utils/jwt_client.py

"""
JWT-клиент для frontend runtime (server-side).

Назначение:
- Получает JWT у backend по API_TOKEN
- Хранит JWT в памяти процесса (не в браузере)
- Автоматически обновляет токен ДО фактического истечения

Важно:
- API_TOKEN хранится в .env (через settings)
- Backend возвращает expires_in (в секундах)
- TTL уменьшается на 10%, чтобы обновлять токен заранее

Backend endpoint:
    POST {API_URL}/auth/token

Тело запроса (JSON):
    {
        "api_token": "<API_TOKEN>"
    }

Response:
    {
        "access_token": "...",
        "expires_in": 900
    }

Логирование:
- События получения токена и ошибок могут логироваться через app.state.log
"""

import time
import httpx
from ..config import settings


class JWTClient:
    """
    Управление JWT-токеном.

    Хранит:
    - self.token — текущий JWT
    - self.expire_at — UNIX timestamp, когда токен нужно обновить

    expire_at рассчитывается с -10% буфером, чтобы избежать race-condition
    при WebSocket или долгих запросах.
    """

    def __init__(self) -> None:
        # Текущий JWT (None, если ещё не получен)
        self.token: str | None = None

        # Timestamp, когда нужно запрашивать новый токен
        self.expire_at: float = 0

    async def get_token(self) -> str:
        """
        Возвращает актуальный JWT.

        Если токен отсутствует или истёк — автоматически вызывает refresh_token().
        """
        now = time.time()
        if self.token is None or now >= self.expire_at:
            await self.refresh_token()
        return self.token

    async def refresh_token(self) -> None:
        """
        Получает новый JWT от backend.

        Алгоритм:
        1. POST /auth/token
        2. Передаём API_TOKEN в теле JSON: {"api_token": "..."}
        3. Получаем access_token и expires_in
        4. Уменьшаем TTL на 10%
        5. Вычисляем expire_at
        """

        # Используем асинхронный HTTP клиент с таймаутом 30 сек
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{settings.API_URL}/auth/token",  # Backend endpoint
                json={"api_token": settings.API_TOKEN}  # Передаем API_TOKEN
            )

            # Проверяем статус ответа
            response.raise_for_status()
            data = response.json()

        # Новый JWT
        self.token = data["access_token"]

        # Backend должен вернуть expires_in (TTL в секундах)
        expires_in = data["expires_in"]

        # ======================================
        # Буфер безопасности (-10% от TTL)
        # ======================================
        # Чтобы токен обновлялся заранее и не успел истечь во время запроса
        effective_ttl = int(expires_in * 0.9)

        # Абсолютное время обновления токена
        self.expire_at = time.time() + effective_ttl
