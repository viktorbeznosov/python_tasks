# app/core/whisper/schema.py

from pydantic import BaseModel

# -------------------------------
# Запрос на скачивание модели
# -------------------------------
class ModelDownloadRequest(BaseModel):
    # Название модели, которую нужно скачать (по умолчанию "small")
    model_name: str = "small"


# -------------------------------
# Ответ о статусе модели
# -------------------------------
class ModelStatusResponse(BaseModel):
    # Статус операции, например "success"
    status: str
    # Название модели
    model_name: str
    # Загружена ли модель на диск
    is_loaded: bool



