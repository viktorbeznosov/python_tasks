# app/core/load/route.py

"""
Маршруты модуля load.

Модуль реализует полный цикл обработки аудио формата m4a:

1. POST   /load/upload     — загрузка аудио
2. GET    /load/transcribe — транскрибация
3. GET    /load/detect     — определение ролей
4. GET    /load/index      — создание индексных баз

Все методы возвращают унифицированную структуру ответа:

{
    "title": str,      # Название операции
    "status": str,     # success | error
    "result": str,     # Человекочитаемый результат
    "progress": int    # Прогресс выполнения (0–100)
}

Описание статусов:

success — операция успешно выполнена  
error   — во время выполнения произошла ошибка
"""

from fastapi import APIRouter, UploadFile, File, Request, Depends
from ..auth.service import get_current_jwt_payload
from .service import LoadService
from .schema import (
    UploadResponse,
    TranscribeResponse,
    DetectRolesResponse,
    IndexResponse
)

# -------------------------------------------------------------------------
# Инициализация роутера
# -------------------------------------------------------------------------
router = APIRouter(prefix="/load", tags=["load"], dependencies=[Depends(get_current_jwt_payload)])

# =============================================================================
# ЗАГРУЗКА АУДИО (m4a)
# =============================================================================
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Загрузка аудио m4a",
    description="""
Загружает аудиофайл формата m4a на сервер.

После успешной загрузки файл сохраняется
и становится доступным для:

- транскрибации
- определения ролей
- создания индексных баз

Ответ всегда имеет унифицированную структуру.
""",
    responses={
        200: {"description": "Аудио успешно загружено (status=success)"},
        400: {"description": "Передан неверный формат файла (status=error)"},
        500: {"description": "Ошибка сохранения файла (status=error)"},
    },
)
async def upload_audio(request: Request, file: UploadFile = File(...)):
    """
    Endpoint загрузки аудио.

    Параметры:
        file (UploadFile): аудиофайл формата m4a

    Возвращает:
        UploadResponse
    """

    log = request.app.state.log
    await log.log_info(target="route.upload", message="POST /load/upload — получен запрос на загрузку аудио")

    try:
        response = await LoadService.upload(request, file)
        await log.log_info(target="route.upload", message="Загрузка аудио завершена успешно")
        return response
    except Exception as e:
        await log.log_error(target="route.upload", message=f"Ошибка при загрузке аудио: {str(e)}")
        raise


# =============================================================================
# ТРАНСКРИБАЦИЯ
# =============================================================================
@router.get(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Транскрибация аудио",
    description="""
Выполняет транскрибацию ранее загруженного аудиофайла.

Процесс:

1. Проверяется наличие загруженного файла
2. Выполняется распознавание речи
3. Формируется текст транскрипции

title ответа: "Транскрибация"
""",
    responses={
        200: {"description": "Транскрибация успешно выполнена (status=success)"},
        400: {"description": "Аудио файл не найден (status=error)"},
        500: {"description": "Ошибка транскрибации (status=error)"},
    },
)
async def transcribe_audio(request: Request):
    """
    Endpoint транскрибации.

    Возвращает:
        TranscribeResponse
    """

    log = request.app.state.log
    await log.log_info(target="route.transcribe", message="GET /load/transcribe — запуск транскрибации")

    try:
        response = await LoadService.transcribe(request)
        await log.log_info(target="route.transcribe", message="Транскрибация успешно завершена")
        return response
    except Exception as e:
        await log.log_error(target="route.transcribe", message=f"Ошибка во время транскрибации: {str(e)}")
        raise


# =============================================================================
# ОПРЕДЕЛЕНИЕ РОЛЕЙ
# =============================================================================
@router.get(
    "/detect",
    response_model=DetectRolesResponse,
    summary="Определение ролей",
    description="""
Определяет роли участников разговора
на основе полученной транскрипции.

Процесс:

1. Проверяется наличие транскрипции
2. Выполняется анализ текста
3. Определяются роли участников

title ответа: "Определение ролей"
""",
    responses={
        200: {"description": "Роли успешно определены (status=success)"},
        400: {"description": "Транскрипция или аудио отсутствуют (status=error)"},
        500: {"description": "Ошибка определения ролей (status=error)"},
    },
)
async def detect(request: Request):
    """
    Endpoint определения ролей.

    Возвращает:
        DetectRolesResponse
    """

    log = request.app.state.log
    await log.log_info(target="route.detect", message="GET /load/detect — запуск определения ролей")

    try:
        response = await LoadService.detect(request)
        await log.log_info(target="route.detect", message="Определение ролей завершено успешно")
        return response
    except Exception as e:
        await log.log_error(target="route.detect", message=f"Ошибка во время определения ролей: {str(e)}")
        raise


# =============================================================================
# СОЗДАНИЕ ИНДЕКСНЫХ БАЗ
# =============================================================================
@router.get(
    "/index",
    response_model=IndexResponse,
    summary="Создание индексных баз",
    description="""
Создаёт индексные базы на основе транскрипции
и определённых ролей.

Процесс:

1. Проверяется наличие необходимых данных
2. Формируются структуры индексирования
3. Сохраняются индексные базы

title ответа: "Создание индексных баз"
""",
    responses={
        200: {"description": "Индексные базы успешно созданы (status=success)"},
        400: {"description": "Недостаточно данных для индексирования (status=error)"},
        500: {"description": "Ошибка создания индексов (status=error)"},
    },
)
async def index_audio(request: Request):
    """
    Endpoint создания индексных баз.

    Возвращает:
        IndexResponse
    """

    log = request.app.state.log
    await log.log_info(target="route.index", message="GET /load/index — запуск создания индексных баз")

    try:
        response = await LoadService.index(request)
        await log.log_info(target="route.index", message="Создание индексных баз завершено успешно")
        return response
    except Exception as e:
        await log.log_error(target="route.index", message=f"Ошибка во время создания индексных баз: {str(e)}")
        raise
