# app/core/neuro/route.py

"""
Маршруты для модуля neuro.

Методы:
- POST /neuro/start — старт обработки файла
- GET /neuro/next — следующий шаг обработки
- POST /neuro/step — выполнить конкретный шаг по коду независимо от состояния

Документация статусов:
- 200 OK — шаг успешно выполнен
- 400 Bad Request — неверные параметры
- 500 Internal Server Error — ошибка выполнения шага
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from ..auth.service import get_current_jwt_payload
from .schema import NeuroStartResponse, NeuroNextResponse, AutoResponse, AutoRequest, StepRequest
from .service import NeuroService

router = APIRouter(prefix="/neuro", tags=["neuro"], dependencies=[Depends(get_current_jwt_payload)])


@router.post("/start", response_model=NeuroStartResponse, summary="Старт обработки файла")
async def neuro_start(request: Request, file: UploadFile = File(...)):
    """
    Начало обработки файла через NeuroService.start

    - Загружает файл через load/upload
    - Очищает состояние out
    - Сохраняет out через pickle
    - Возвращает структуру ответа
    """
    try:
        return await NeuroService.start(request, file)
    except Exception as e:
        log = request.app.state.log
        await log.log_error(target="neuro_start", message=f"Ошибка старта нейроанализа: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка старта нейроанализа: {str(e)}")


@router.get("/next", summary="Следующий шаг обработки")
async def neuro_next(request: Request):    
    """
    Выполняет следующий шаг из steps.txt через NeuroService.process_next

    - Возвращает результат с is_complete=True только если это последний шаг
    """
    try:
        return await NeuroService.process_next(request)
    except Exception as e:
        log = request.app.state.log
        await log.log_error(target="neuro_next", message=f"Ошибка в next: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка в next: {str(e)}")


@router.get("/step", summary="Выполнить конкретный шаг по коду")
async def neuro_step(request: Request, code: str):
    """
    Выполняет конкретный шаг независимо от того, выполнялся он ранее или нет.

    Параметры:
    - code: str — код шага из steps.txt или пользовательский

    Возвращает:
    - code, title, status, result, progress, is_complete
    """
    try:
        return await NeuroService.step(request, code)
    except Exception as e:
        log = request.app.state.log
        await log.log_error(target="neuro_step", message=f"Ошибка в step {code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка в step {code}: {str(e)}")
    
@router.post("/step", response_model=NeuroNextResponse, summary="Выполнить конкретный шаг по коду (POST)")
async def neuro_step_post(request: Request, payload: StepRequest):
    try:
        return await NeuroService.step(request, payload.code)
    except Exception as e:
        log = request.app.state.log
        await log.log_error(target="neuro_step", message=f"Ошибка в step {payload.code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка в step {payload.code}: {str(e)}")    

@router.get("/state", summary="Получить текущее состояние обработки")
async def neuro_state(request: Request):
    """
    Возвращает текущее состояние out из pickle.

    Используется для:
    - восстановления состояния фронтенда
    - отладки
    - проверки выполненных шагов
    """
    try:
        return await NeuroService.get_state(request)
    except Exception as e:
        log = request.app.state.log
        await log.log_error(target="neuro_state", message=f"Ошибка получения состояния: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения состояния: {str(e)}")

@router.post("/auto", summary="Автоматическое выполнение шагов", response_model=AutoResponse)
async def neuro_auto(request: Request, payload: AutoRequest):
    """
    Выполняет все шаги начиная с указанного кода до конца.
    """
    try:
        return await NeuroService.auto(request, payload.code)
    except Exception as e:
        log = request.app.state.log
        await log.log_error(target="neuro_auto", message=str(e))
        raise HTTPException(status_code=500, detail=str(e))