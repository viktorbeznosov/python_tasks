# app/core/load/schema.py

"""
Pydantic-схемы для модуля load

Стандарт ответа API:
- title: заголовок операции
- status: статус выполнения (success / error)
- result: текстовый результат операции
- progress: прогресс выполнения (0–100)

Такая структура позволяет унифицировать ответы всех методов модуля.
"""

from pydantic import BaseModel, Field


class BaseLoadResponse(BaseModel):
    """
    Базовая структура ответа для методов модуля load.

    Поля:
    - code: код шага
    - title: человеко-читаемый заголовок шага
    - status: статус выполнения (success / error)
    - result: текстовый результат
    - progress: прогресс выполнения от 0 до 100
    """

    code: str = Field(..., example="")
    title: str = Field(..., example="Загрузка файла")
    status: str = Field(..., example="success")
    result: str = Field(..., example="Файл успешно загружен")
    progress: int = Field(..., ge=0, le=100, example=0)


class UploadResponse(BaseLoadResponse):
    """Ответ метода загрузки файла"""
    pass


class TranscribeResponse(BaseLoadResponse):
    """Ответ метода транскрипции"""
    pass


class DetectRolesResponse(BaseLoadResponse):
    """Ответ метода определения ролей"""
    pass


class IndexResponse(BaseLoadResponse):
    """Ответ метода создания индексных баз"""
    pass
