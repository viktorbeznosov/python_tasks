# app/core/websocket/route.py

"""
WebSocket роуты
- Подключение через /ws
- Проверка JWT после handshake
- Логирование подключений, сообщений и отключений
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..auth.service import verify_jwt

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket подключение:

    Алгоритм:
    1. Сразу вызываем websocket.accept() для успешного handshake
    2. Ждём первичное сообщение от клиента с JWT
    3. Проверяем токен через verify_jwt
    4. Если токен верный — подключаем клиента в WSManager
    5. Сохраняем соединение и ждём сообщения от клиента
    6. При отключении клиента — удаляем из WSManager
    """
    app = websocket.app
    log = app.state.log
    ws_manager = app.state.ws

    # ----------------------
    # 0. Принимаем handshake
    # ----------------------
    try:
        await websocket.accept()
        await log.log_info(target="ws", message="WS handshake: клиент подключился, ждём JWT")
    except Exception as e:
        await log.log_error(target="ws", message=f"Ошибка при accept WebSocket: {e}")
        return

    # ----------------------
    # 1. Ожидаем auth сообщение
    # ----------------------
    try:
        data = await websocket.receive_json()
        await log.log_info(target="ws", message=f"Получено auth сообщение: {data}")

        # Проверяем структуру сообщения
        if data.get("type") != "auth" or "token" not in data:
            await log.log_info(target="ws", message="Auth сообщение некорректно, закрываем соединение")
            await websocket.close(code=1008)
            return

        token = data["token"]

        # ----------------------
        # 2. Проверка JWT
        # ----------------------
        await verify_jwt(token)
        await log.log_info(target="ws", message="JWT проверен успешно")

    except Exception as e:
        await log.log_error(target="ws", message=f"Ошибка при аутентификации WS: {e}")
        await websocket.close(code=1008)
        return

    # ----------------------
    # 3. Подключаем в WSManager
    # ----------------------
    await ws_manager.connect(websocket)
    await log.log_info(target="ws", message="WS клиент добавлен в менеджер")

    # ----------------------
    # 4. Основной цикл приёма сообщений от клиента
    # ----------------------
    try:
        while True:
            msg = await websocket.receive_text()
            await log.log_info(target="ws", message=f"Получено сообщение от клиента: {msg}")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        await log.log_info(target="ws", message="WS клиент отключился")
