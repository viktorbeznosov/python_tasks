# app/core/gpt/route.py

"""
Маршруты для работы с LLM (GPT-модели).

Назначение:
1. Приём POST-запросов с текстом prompt
2. Проверка авторизации пользователя
3. Асинхронный вызов GPT через service.py
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from .schema import GPTRequest, GPTResponse
from .service import call_gpt
from ..auth.service import get_current_jwt_payload  # JWT проверка

router = APIRouter( 
    prefix="/gpt",
    tags=["gpt"],
    dependencies=[Depends(get_current_jwt_payload)]  # все эндпоинты требуют авторизацию
)

@router.post("/", response_model=GPTResponse, status_code=status.HTTP_200_OK)
async def gpt_endpoint(data: GPTRequest, request: Request):
    """
    POST /gpt
    Асинхронно вызывает GPT-модель и возвращает ответ.
    Требует авторизации пользователя.
    """
    try:
        answer = await call_gpt(
            request=request,            
            prompt=data.prompt,
            temperature=data.temperature
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при вызове LLM: {str(e)}"
        )