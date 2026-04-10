# app/core/faiss/route.py

"""
Маршруты модуля FAISS.

Эндпоинты для работы с FAISS-индексами на основе Markdown баз знаний:

1. POST /faiss/build   — построение нового индекса (если нет)
2. POST /faiss/rebuild — полное пересоздание индекса
3. POST /faiss/search  — поиск по существующему индексу

Все маршруты защищены JWT-токеном.
"""

from fastapi import APIRouter, HTTPException, Depends
from .service import FaissService
from .schema import FaissBuildRequest, FaissSearchRequest, FaissSearchResponse
from ..auth.service import get_current_jwt_payload  # проверка JWT

# -------------------------------------------------------------------------
# Инициализация роутера с обязательной проверкой JWT
# -------------------------------------------------------------------------
router = APIRouter(
    prefix="/faiss",
    tags=["FAISS"],
    dependencies=[Depends(get_current_jwt_payload)]  # все эндпоинты проверяют токен
)

# =============================================================================
# СОЗДАНИЕ ИНДЕКСА
# =============================================================================
@router.post("/build")
async def build_index(req: FaissBuildRequest):
    """
    Создание нового индекса для Markdown базы знаний.
    Если индекс уже существует — возвращается существующий.
    """
    try:
        service = FaissService(text_file=req.text_file, chunk_size=req.chunk_size)
        service.build_index()
        return {"status": "success", "index_path": str(service.index_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ПЕРЕСОЗДАНИЕ ИНДЕКСА
# =============================================================================
@router.post("/rebuild")
async def rebuild_index(req: FaissBuildRequest):
    """
    Полное пересоздание FAISS индекса:
    1. Удаление старой индексной базы
    2. Построение нового индекса
    """
    try:
        service = FaissService(text_file=req.text_file, chunk_size=req.chunk_size)
        service.rebuild_index()
        return {"status": "success", "index_path": str(service.index_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ПОИСК ПО ИНДЕКСУ
# =============================================================================
@router.post("/search", response_model=FaissSearchResponse)
async def search_index(req: FaissSearchRequest):
    """
    Поиск по существующему индексу Markdown базы знаний.
    Если индекс не найден, он создается автоматически.

    Входные параметры:
    - md_file: имя файла базы знаний (.md)
    - query: поисковый запрос
    - k: количество релевантных результатов (по умолчанию 5)
    """
    try:
        service = FaissService(text_file=req.text_file)
        results = service.search(query=req.query, k=req.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))