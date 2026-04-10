# app/core/neuro/schema.py

"""
Схемы для модуля neuro

Используется в route.py для описания входных и выходных данных.
"""

from pydantic import BaseModel, Field


class NeuroStartRequest(BaseModel):
    """
    Входные параметры для start.
    
    Примечание:
    - Файл через UploadFile передается напрямую в route, поэтому здесь нет полей.
    """
    pass


class NeuroStartResponse(BaseModel):
    """
    Выходные параметры для start.
    
    Дублируют UploadResponse и добавляют поле is_complete.
    """
    code: str
    title: str
    status: str
    result: str
    progress: int
    is_complete: bool  # True, если обработка выполнена


class NeuroNextResponse(BaseModel):
    """
    Выходные параметры для next.
    
    Такая же структура, что и start, с полем is_complete.
    """
    code: str
    title: str
    status: str
    result: str
    progress: int
    is_complete: bool
    
class AutoRequest(BaseModel):
    """
    Запуск автоматического выполнения с указанного шага.
    """
    code: str = Field(..., example="transcribe")

class AutoResponse(BaseModel):
    """
    Ответ метода автоматического выполнения шагов.
    """
    status: str = Field(..., example="success")
    started_from: str = Field(..., example="transcribe")
    is_complete: bool = Field(..., example=True)    
    
class StepRequest(BaseModel):
    """
    # POST /step — код передаётся в data (JSON)
    """
    code: str    
