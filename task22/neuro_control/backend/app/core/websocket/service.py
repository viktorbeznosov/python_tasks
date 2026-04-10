# app/core/websocket/service.py

"""
WebSocket менеджер
Управляет активными соединениями и рассылкой сообщений.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List


class WSManager:
    def __init__(self):
        # Список активных подключений
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """
        Добавляет новое подключение.
        Важно: websocket уже должен быть принят (await websocket.accept() вызван в роуте)
        """
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        Удаляет подключение из списка активных.
        """
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: dict):
        """
        Отправляет сообщение всем подключённым клиентам.
        Автоматически отсеивает отключенные WebSocket.
        """
        disconnected = []

        for conn in self.connections:
            try:
                await conn.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(conn)

        for conn in disconnected:
            self.disconnect(conn)

    async def shutdown(self):
        # Закрываем все активные соединения
        for conn in self.connections:
            try:
                await conn.close()
            except Exception:
                pass
        self.connections.clear()            
