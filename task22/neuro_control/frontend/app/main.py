# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ----------------------
# Логгер и JWT клиент
# ----------------------
from .utils.log import Log
from .utils.jwt_client import JWTClient  # Клиент для получения/обновления JWT

# ----------------------
# Роут главной страницы
# ----------------------
from .core.index.route import router as router_index

# ----------------------
# Загружаем переменные окружения из .env
# ----------------------
load_dotenv()


# ----------------------
# Контекст жизненного цикла приложения (lifespan)
# ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan приложения:
    - создаём логгер
    - создаём JWT-клиент (автообновление токена)
    - инициализируем Jinja2 templates
    - yield передаёт управление FastAPI
    - корректный shutdown логгера
    """
    # ----------------------
    # Логгер приложения
    # ----------------------
    app.state.log = Log()
    await app.state.log.log_info("startup", "Запуск frontend-приложения")

    # ----------------------
    # JWT клиент
    # ----------------------
    # Используется для автоматического получения нового токена
    # при истечении времени жизни текущего
    app.state.jwt_client = JWTClient()

    # ----------------------
    # Templates (Jinja2)
    # ----------------------
    # Сохраняем в app.state, чтобы роуты могли обращаться через request.app.state.jinja_env
    TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
    print(f"Templates directory: {TEMPLATES_DIR}")
    print(f"Directory exists: {TEMPLATES_DIR.exists()}")
    
    TEMPLATES_DIR.mkdir(exist_ok=True)
    
    # Создаем окружение Jinja2 напрямую (вместо Jinja2Templates)
    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(['html', 'xml']),
        cache_size=0,  # Полностью отключаем кэш
        auto_reload=True  # Автоматическая перезагрузка шаблонов
    )
    
    # Сохраняем окружение в app.state
    app.state.jinja_env = jinja_env

    # ----------------------
    # Передаём управление FastAPI
    # ----------------------
    yield

    # ----------------------
    # Корректный shutdown
    # ----------------------
    await app.state.log.log_info("shutdown", "Остановка frontend-приложения")
    await app.state.log.shutdown()


# ----------------------
# Создаём FastAPI приложение
# ----------------------
app = FastAPI(lifespan=lifespan)

# ----------------------
# Подключаем статические файлы
# ----------------------
# Все файлы в app/frontend/static будут доступны по пути /static
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ----------------------
# CORS middleware
# ----------------------
# Разрешаем любые запросы для разработки
# (для продакшена лучше ограничить конкретные домены)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Разрешаем любые домены
    allow_credentials=True,
    allow_methods=["*"],        # Разрешаем все HTTP методы
    allow_headers=["*"],        # Разрешаем все заголовки
)

# ----------------------
# Подключаем маршруты
# ----------------------
# Только фронтенд-роуты, без WebSocket proxy
app.include_router(router_index)  # Главная страница с шаблонами

# ----------------------
# Запуск сервера
# ----------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",   # формат "модуль:объект"
        host="127.0.0.1", # локальный хост
        port=8000,        # порт сервера
        log_level="info", # уровень логов
        reload=True       # авто-перезагрузка при изменении кода (для разработки)
    )