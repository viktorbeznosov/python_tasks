# app/main.py

from fastapi import FastAPI             # Импортируем класс FastAPI из библиотеки FastAPI — он нужен для создания веб‑приложения
import uvicorn                          # Импортируем библиотеку uvicorn — это ASGI‑сервер, который будет запускать наше приложение
from .utils.log import Log              # Импорт логгера
from .utils.pickle import PickleStorage # Класс для работы с дампом памяти

# WebSocket
from .core.websocket.service import WSManager
from .core.websocket.route import router as router_websocket

# Роуты
from .core.auth.route import router as router_auth
from .core.load.route import router as router_load
from .core.neuro.route import router as router_neuro
from .core.whisper.route import router as router_whisper
from .core.faiss.route import router as router_faiss
from .core.gpt.route import router as router_gpt
import os

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()

# Контекст управления жизненным циклом приложения
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Создание объекта логгера
    app.state.log = Log() 

    # Лог запуска приложения
    await app.state.log.log_info(target="startup", message="Запуск приложения")
    
    # Переменные состояния
    app.state.pickle = PickleStorage()  # объект дампа состояния
    app.state.out = []                  # словарь результатов
    
    # WebSocket менеджер
    app.state.ws = WSManager()    

    # Контекст приложения
    yield  

    # Лог остановки
    await app.state.log.log_info(target="startup", message="Остановка приложения")

    # Shutdown для WebSocket
    await app.state.ws.shutdown()

    # Shutdown для корректного завершения работы асинхронных логгеров
    await app.state.log.shutdown()

# Создаём FastAPI приложение
app = FastAPI(lifespan=lifespan)

# Обработка CORS-запросов
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Эндпоинт "Обработчик GET-метода /"
@app.get("/")               # Декоратор, который регистрирует функцию как обработчик GET‑запросов к корневому пути `/`;
def root():                 # Определяем функцию‑обработчик для маршрута `/`;
                            # имя функции может быть любым, но должно быть уникальным в рамках модуля
    return {"message": "Hello, World!"}  # Возвращаем словарь в формате JSON — FastAPI автоматически преобразует его в HTTP‑ответ;

# Подключение маршрутов
app.include_router(router_auth)
app.include_router(router_load)
app.include_router(router_neuro)
app.include_router(router_websocket)
app.include_router(router_whisper)
app.include_router(router_faiss)
app.include_router(router_gpt)

 
# Запуск
if __name__ == "__main__":  # Проверяем, запускается ли скрипт напрямую (а не импортируется как модуль)
    uvicorn.run(            # Вызываем функцию `run` из библиотеки Uvicorn — она запускает ASGI‑сервер
        "app.main:app",     # Строка формата 'модуль:объект': 
                            # - 'app' — имя пакета/папки, где лежит код;
                            # - 'main' — имя Python‑файла (без расширения .py), где определён объект приложения;
                            # - второй 'app' — имя объекта FastAPI (экземпляра класса `FastAPI`), объявленного в файле `main.py`
        host="127.0.0.1",   # Указываем IP‑адрес, на котором будет слушать сервер;
                            # '127.0.0.1' — локальный хост (только для текущего компьютера)
        port=5000,          # Задаём порт, на котором будет работать сервер; 
        log_level="info",   # Устанавливаем уровень логирования: 'info' — будут выводиться информационные сообщения
                            # (например, о принятых запросах, ошибках и т. д.)
        reload=True         # Включаем режим автоперезагрузки: при изменении кода сервер автоматически перезапустится;
                            # удобно при разработке, но не рекомендуется для продакшена
    )