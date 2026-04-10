# app/core/gpt/schema.py

"""
Pydantic-схемы для модуля GPT.

Назначение:
1. Валидация данных входящих запросов к LLM
2. Описание структуры ответа от LLM
3. Генерация документации OpenAPI (Swagger)
"""

from pydantic import BaseModel, Field

class GPTRequest(BaseModel):
    """
    Схема запроса к GPT-модели.
    """
    prompt: str = Field(..., description="Текстовый запрос для модели GPT")
    temperature: float = Field(0.7, ge=0, le=1, description="Степень креативности")


class GPTResponse(BaseModel):
    """
    Схема ответа от GPT-модели.
    """
    answer: str = Field(..., description="Ответ модели GPT")