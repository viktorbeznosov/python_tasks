# app/core/whisper/route.py

from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from .service import WhisperService
from .schema import ModelDownloadRequest, ModelStatusResponse

# Роутер для работы с Whisper
router = APIRouter(prefix="/whisper", tags=["Whisper"])


@router.post("/download", response_model=ModelStatusResponse)
async def download_model(data: ModelDownloadRequest, request: Request):
    """
    Скачивает и загружает модель в локальную папку base/whisper.

    Если модель уже скачана, повторно не загружается,
    а просто инициализируется.
    """
    return await WhisperService.download_model(
        model_name=data.model_name,
        request=request
    )


@router.get("/status", response_model=ModelStatusResponse)
async def model_status(request: Request):
    """
    Проверка текущего состояния модели:
    - загружена ли в память
    - какое имя модели используется
    """
    return await WhisperService.get_status(request=request) 


@router.post("/transcribe")
async def transcribe_audio(request: Request):
    """
    Роут для транскрибации аудио.
    """
    audio_path = Path("base") / "audio" / "audio.m4a"

    if not audio_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Файл base/audio/audio.m4a не найден"
        )

    # Вызываем WhisperService и получаем асинхронный генератор
    # и общее количество шагов для прогресса
    segment_gen, total_steps = await WhisperService.transcribe(
        audio_path=str(audio_path),
        request=request
    )

    segments_list = []

    # Превращаем асинхронный генератор в список словарей
    async for s in segment_gen:
        segments_list.append({
            "start": s.start,
            "end": s.end,
            "text": s.text
        })

    # Возвращаем результат
    return segments_list
