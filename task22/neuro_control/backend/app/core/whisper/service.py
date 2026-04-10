# app/core/whisper/service.py

"""
Сервис для работы с официальными моделями Whisper.

Функции:
- Скачивание модели в base/whisper
- Автозагрузка после рестарта
- Проверка статуса
- Транскрибация аудио
"""

from pathlib import Path
from faster_whisper import WhisperModel
from fastapi import HTTPException, Request  
from pathlib import Path
import os
from ...config import settings
import tempfile
from pydub import AudioSegment
import shutil
import asyncio

# Папка для хранения моделей
MODEL_DIR = Path.home() / "projects_python" / "models" / "whisper"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Установим токен для Hugging Face
os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HF_TOKEN

class WhisperService:
    """
    Сервис для работы с WhisperModel.
    Хранит:
    - _model: объект модели
    - _model_name: имя текущей загруженной модели
    """

    _model = None       # Объект модели Whisper

    # -------------------------------------------------
    # Автозагрузка модели по имени
    # -------------------------------------------------
    @classmethod
    async def _ensure_model_loaded(cls, request: Request):
        """
        Гарантирует, что модель Whisper загружена в память.

        Источник имени модели — settings.WHISPER_MODEL.
        После рестарта backend модель автоматически загрузится
        из base/whisper или будет скачана при необходимости.
        """

        log = request.app.state.log

        # Если модель уже загружена — ничего не делаем
        if cls._model is not None:
            return

        model_name = settings.WHISPER_MODEL

        await log.log_info(
            target="whisper.service",
            message=f"Loading Whisper model: {model_name}"
        )

        try:
            cls._model = WhisperModel(
                model_size_or_path=model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR)
            )

            await log.log_info(
                target="whisper.service",
                message=f"Whisper model loaded successfully: {model_name}"
            )

        except Exception as e:
            await log.log_error(
                target="whisper.service",
                message=f"Failed to load Whisper model: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------
    # Скачивание официальной модели
    # -------------------------------------------------
    @classmethod
    async def download_model(cls, model_name: str, request: Request):
        """
        Скачивает официальную модель Whisper в base/whisper.
        После скачивания загружает её в память.
        """
        log = request.app.state.log

        cls._model_name = model_name  # сохраняем имя модели

        await log.log_info(
            target="whisper.service",
            message=f"Downloading Whisper model: {model_name}"
        )

        try:
            # При инициализации WhisperModel автоматически скачивает модель в download_root
            cls._model = WhisperModel(
                model_size_or_path=model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR)
            )

            await log.log_info(
                target="whisper.service",
                message=f"Whisper model downloaded and loaded: {model_name}"
            )

            return {
                "status": "success",
                "model_name": model_name,
                "is_loaded": True
            }

        except Exception as e:
            await log.log_error(
                target="whisper.service",
                message=f"Error downloading Whisper model {model_name}: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------
    # Проверка статуса модели
    # -------------------------------------------------
    @classmethod
    async def get_status(cls, request: Request):

        log = request.app.state.log

        # Если модели нет в памяти — пробуем загрузить
        if cls._model is None:
            await log.log_info(
                target="whisper.service",
                message="Model not loaded. Lazy loading triggered from /status"
            )

            try:
                await cls._ensure_model_loaded(request)
            except Exception:
                return {
                    "status": "success",
                    "model_name": settings.WHISPER_MODEL,
                    "is_loaded": False
                }

        return {
            "status": "success",
            "model_name": settings.WHISPER_MODEL,
            "is_loaded": cls._model is not None
        }

    # -------------------------------------------------
    # Транскрибация аудио
    # -------------------------------------------------
    @classmethod
    async def transcribe(cls, audio_path: str, request: Request):
        """
        Транскрибация аудио через Whisper с разбиением на чанки через AudioSegment.

        Логика:
        1. Загружаем модель, если не загружена
        2. Разбиваем аудио на сегменты (1% или 5% в зависимости от длины)
        3. Генератор по сегментам WhisperModel
        4. Тайминги сегментов корректируются относительно оригинального аудио
        5. Возвращаем асинхронный генератор сегментов
        """

        log = request.app.state.log
        await cls._ensure_model_loaded(request)
        await log.log_info(target="whisper.service", message=f"Starting chunked transcription: {audio_path}")

        # -------------------------------------------------
        # 1. Загружаем аудио через AudioSegment
        # -------------------------------------------------
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            await log.log_error(target="whisper.service", message=f"Failed to open audio file: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        # -------------------------------------------------
        # 2. Определяем длительность шагов
        # -------------------------------------------------
        audio_length_ms = len(audio)
        chunk_ratio = 0.05 if audio_length_ms < 10*60*1000 else 0.01  # <10мин = 5%, иначе 1%
        chunk_duration_ms = int(audio_length_ms * chunk_ratio)
        await log.log_info(target="whisper.service", message=f"Audio length: {audio_length_ms}ms, chunk duration: {chunk_duration_ms}ms")

        # -------------------------------------------------
        # 3. Создаем временную папку для чанков
        # -------------------------------------------------
        temp_dir = Path(tempfile.mkdtemp(prefix="tmp_whisper_"))
        await log.log_info(target="whisper.service", message=f"Temp dir created: {temp_dir}")

        # -------------------------------------------------
        # 4. Разбиваем аудио на сегменты и сохраняем в temp_dir
        # -------------------------------------------------
        chunk_files = []
        try:
            for i in range(0, audio_length_ms, chunk_duration_ms):
                chunk = audio[i:i+chunk_duration_ms]
                chunk_file = temp_dir / f"chunk_{i//chunk_duration_ms:03d}.wav"
                chunk.export(chunk_file, format="wav")
                chunk_files.append(chunk_file)
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            await log.log_error(target="whisper.service", message=f"Audio splitting failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Audio splitting failed: {str(e)}")

        total_steps = len(chunk_files)
        total_steps = max(1, int(total_steps * 2))
        await log.log_info(target="whisper.service", message=f"Chunked transcription ready, total steps: {total_steps}")

        # -------------------------------------------------
        # 5. Генератор сегментов
        # -------------------------------------------------
        async def segment_generator():
            time_offset_ms = 0
            for chunk_file in chunk_files:
                try:
                    segments, _info = cls._model.transcribe(
                        str(chunk_file),
                        beam_size=5,
                        vad_filter=False
                    )
                    for seg in segments:
                        # корректируем тайминги относительно полного аудио
                        seg.start += time_offset_ms / 1000.0
                        seg.end += time_offset_ms / 1000.0
                        yield seg
                except Exception as e:
                    await log.log_error(target="whisper.service", message=f"Transcription failed for {chunk_file}: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

                # сдвигаем offset
                time_offset_ms += chunk_duration_ms
                await asyncio.sleep(0)  # отдаём управление loop

            # -------------------------------------------------
            # 6. Очистка временной папки после всех сегментов
            # -------------------------------------------------
            shutil.rmtree(temp_dir, ignore_errors=True)
            await log.log_info(target="whisper.service", message=f"Temp dir {temp_dir} removed")

        return segment_generator(), total_steps