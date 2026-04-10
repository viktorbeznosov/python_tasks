# app/core/load/service.py

"""
Сервисный модуль для работы с загрузкой и обработкой аудио файлов формата m4a.
"""

import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException, Request
import asyncio
from .schema import (
    UploadResponse,
    TranscribeResponse,
    DetectRolesResponse,
    IndexResponse
)
from ..websocket.schema import WSMessage
from ..whisper.service import WhisperService
from ..gpt.service import call_gpt
from ..faiss.service import FaissService

# ---------------------------------------------------------
# Пути проекта
# ---------------------------------------------------------

# Корень backend (уровень выше app)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Директория хранения аудио
BASE_AUDIO_DIR = BASE_DIR / "base" / "audio"
BASE_AUDIO_DIR.mkdir(exist_ok=True)

# Имена файлов
AUDIO_FILE = BASE_AUDIO_DIR / "audio.m4a"
TRANSCRIBE_FILE = BASE_AUDIO_DIR / "audio.txt"


class LoadService:
    """
    Сервисный класс для работы с аудио формата m4a.
    """

    # =====================================================
    # ЗАГРУЗКА АУДИО
    # =====================================================

    @staticmethod
    async def upload(request: Request, file: UploadFile) -> UploadResponse:
        """
        Загружает аудио файл формата m4a.

        Логика:
        1. Проверка MIME-типа (audio/m4a, audio/x-m4a)
        2. Удаление предыдущего файла при наличии
        3. Сохранение нового файла
        4. Логирование результата

        Возвращает:
            UploadResponse

        Возможные ошибки:
        - 400: неверный формат файла
        - 500: ошибка файловой системы
        """

        log = request.app.state.log
        client = request.client.host

        # ------------------------------
        # Проверка MIME-типа
        # ------------------------------
        if file.content_type not in ("audio/m4a", "audio/x-m4a"):
            await log.log_warning(
                target="load.upload",
                message=f"Invalid MIME type from {client}"
            )
            raise HTTPException(
                status_code=400,
                detail="Разрешены только m4a файлы"
            )

        try:
            # Удаляем старый файл
            if AUDIO_FILE.exists():
                await log.log_info(target="load.upload",  message=f"Удаление {AUDIO_FILE}")                                
                AUDIO_FILE.unlink()

            # Сохраняем новый
            with AUDIO_FILE.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            await log.log_info(
                target="load.upload",
                message=f"Audio uploaded successfully from {client}"
            )

            return UploadResponse(
                code="upload",
                title="Загрузка файла",
                status="success",
                result="Аудио успешно загружено",
                progress=0
            )

        except Exception as e:
            await log.log_error(
                target="load.upload",
                message=f"Upload error from {client}: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка сохранения файла: {str(e)}"
            )

    # =====================================================
    # ТРАНСКРИБАЦИЯ
    # =====================================================
    @staticmethod
    async def transcribe(request: Request) -> TranscribeResponse:
        """
        Транскрибация аудио файла base/audio/audio.m4a
        через WhisperService.

        Логика:
        1. Проверка наличия аудио
        2. Удаление старого audio.txt
        3. Получение генератора сегментов + total_steps
        4. Стриминг сегментов + отправка прогресса через WS
        5. Сохранение результата
        """

        log = request.app.state.log
        ws = request.app.state.ws
        client = request.client.host

        # --------------------------------------------------
        # 1. Проверяем наличие загруженного аудио
        # --------------------------------------------------
        if not AUDIO_FILE.exists():
            await log.log_warning(
                target="load.transcribe",
                message=f"Audio file not found for {client}"
            )
            raise HTTPException(
                status_code=400,
                detail="Сначала загрузите аудио файл"
            )

        # --------------------------------------------------
        # 2. Удаляем старую транскрипцию (если есть)
        # --------------------------------------------------
        if TRANSCRIBE_FILE.exists():
            TRANSCRIBE_FILE.unlink()

        await log.log_info(
            target="load.transcribe",
            message=f"Transcription started for {client}"
        )

        # --------------------------------------------------
        # 3. Получаем асинхронный генератор сегментов и total_steps
        # --------------------------------------------------
        segment_gen, total_steps = await WhisperService.transcribe(str(AUDIO_FILE), request)

        lines: list[str] = []
        step_counter = 0

        # --------------------------------------------------
        # 4. Стриминг сегментов
        # --------------------------------------------------
        async for segment in segment_gen:
            lines.append(segment.text)
            step_counter += 1

            # Прогресс рассчитываем по количеству шагов
            progress = int(step_counter / total_steps * 100)
            if progress > 100:
                progress = 100  # никогда не больше 100%

            if ws:
                message = WSMessage(
                    type="progress",
                    code="transcribe",
                    title="Транскрибация",
                    status="processing",
                    result=f"Фрагмент {segment.start:.2f}-{segment.end:.2f}",
                    progress=progress,
                    is_complete=False
                )
                await ws.broadcast(message.model_dump())
                await log.log_info(target="load.transcribe", message=f"Transcription progress: {progress} %")

            await asyncio.sleep(0)
 
        # --------------------------------------------------
        # 5. Сохраняем итоговую транскрипцию
        # --------------------------------------------------
        final_text = "\n".join(lines)
        TRANSCRIBE_FILE.write_text(final_text, encoding="utf-8")

        await log.log_info(
            target="load.transcribe",
            message=f"Transcription completed for {client}"
        )

        # --------------------------------------------------
        # Возвращаем HTTP-ответ
        # --------------------------------------------------
        return TranscribeResponse(
            code="transcribe",
            title="Транскрибация",
            status="success",
            result=final_text,
            progress=0
        )

    # =====================================================
    # ОПРЕДЕЛЕНИЕ РОЛЕЙ
    # =====================================================
    @staticmethod
    async def detect(request: Request) -> DetectRolesResponse:
        """
        Определение ролей для больших аудио/текстов с прогрессом и разбиением на части.
        Логи детализированы: чтение исходника, каждый chunk, прогресс, GPT результат.
        """
        log = request.app.state.log
        ws = request.app.state.ws
        client = request.client.host

        source_file = BASE_AUDIO_DIR / "audio.txt"
        if not source_file.exists():
            await log.log_warning("load.detect", f"Source file not found for {client}")
            raise HTTPException(400, detail="Сначала нужно выполнить транскрибацию аудио")

        # Пути к файлам
        manager_path = BASE_AUDIO_DIR / "manager.txt"
        client_path = BASE_AUDIO_DIR / "client.txt"
        dialog_path = BASE_AUDIO_DIR / "dialog.txt"

        # Удаляем старые файлы
        for f in [manager_path, client_path, dialog_path]:
            if f.exists():
                f.unlink()
                await log.log_info("load.detect", f"Удалён старый файл: {f.name}")

        # Чтение system prompt
        prompt_file = BASE_DIR / "base" / "prompt" / "detect.md"
        if prompt_file.exists():
            system_text = prompt_file.read_text(encoding="utf-8").strip()
            await log.log_info("load.detect", f"Prompt загружен: {prompt_file}")
        else:
            system_text = ""
            await log.log_warning("load.detect", f"Prompt файл не найден: {prompt_file}")

        # Чтение исходного текста
        source_text = source_file.read_text(encoding="utf-8")
        lines = source_text.splitlines()
        await log.log_info("load.detect", f"Исходный файл: {source_file}, строк: {len(lines)}")

        # Разбиваем на чанки
        chunk_size = 50
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        await log.log_info("load.detect", f"Всего чанков: {len(chunks)}, размер чанка: {chunk_size} строк")

        final_result = []

        # Обработка каждого чанка
        for idx, chunk_lines in enumerate(chunks, start=1):
            chunk_text = "\n".join(chunk_lines)
            gpt_prompt = f"{system_text}\n\nТекст аудио:\n{chunk_text}"

            await log.log_info(
                "load.detect",
                f"Обработка chunk {idx}/{len(chunks)}, строк: {len(chunk_lines)}, символов: {len(chunk_text)}"
            )

            try:
                chunk_result = await call_gpt(prompt=gpt_prompt, temperature=0.7, request=request)
                await log.log_info("load.detect", f"GPT обработка chunk {idx} завершена, символов в ответе: {len(chunk_result)}")
            except Exception as e:
                await log.log_error("load.detect", f"GPT ошибка на chunk {idx}: {str(e)}")
                raise HTTPException(500, detail=f"Ошибка GPT сервиса: {str(e)}")

            final_result.append(chunk_result)

            # Промежуточный прогресс
            progress = int((idx / len(chunks)) * 100)
            if ws:
                await ws.broadcast({
                    "type": "progress",
                    "code": "detect",
                    "title": "Определение ролей",
                    "status": "processing",
                    "result": f"Обработан chunk {idx}/{len(chunks)}",
                    "progress": progress,
                    "is_complete": False
                })
            await log.log_info("load.detect", f"Прогресс: {progress}% после chunk {idx}")

        # Склеиваем все результаты GPT
        gpt_result = "\n".join(final_result)
        dialog_path.write_text(gpt_result, encoding="utf-8")
        await log.log_info("load.detect", f"Склеен итоговый диалог, записан в {dialog_path}")

        # Разделение на manager.txt и client.txt
        manager_lines, client_lines = [], []
        current_role = None
        for line in gpt_result.splitlines():
            line = line.strip()
            if line.startswith("Менеджер:"):
                current_role = "manager"
                manager_lines.append(line.replace("Менеджер:", "").strip())
            elif line.startswith("Клиент:"):
                current_role = "client"
                client_lines.append(line.replace("Клиент:", "").strip())
            else:
                if current_role == "manager":
                    manager_lines.append(line)
                elif current_role == "client":
                    client_lines.append(line)

        manager_path.write_text("\n".join(manager_lines).strip(), encoding="utf-8")
        client_path.write_text("\n".join(client_lines).strip(), encoding="utf-8")
        await log.log_info(
            "load.detect",
            f"Разделены роли: менеджер({len(manager_lines)} строк), клиент({len(client_lines)} строк)"
        )

        # Финальный результат
        result_text = (
            f"## dialog.txt{chr(10)}{gpt_result}{chr(10)}{chr(10)}"
            f"## manager.txt{chr(10)}{'{chr(10)}'.join(manager_lines)}{chr(10)}{chr(10)}"
            f"## client.txt{chr(10)}{'{chr(10)}'.join(client_lines)}"
        )
        await log.log_info("load.detect", f"Role detection завершён для {client}")

        return DetectRolesResponse(
            code="detect",
            title="Определение ролей",
            status="success",
            result=result_text,
            progress=100
        )

    # =====================================================
    # СОЗДАНИЕ ИНДЕКСНЫХ БАЗ
    # =====================================================
    from ..faiss.service import FaissService

    @staticmethod
    async def index(request: Request) -> IndexResponse:
        """
        Создаёт индексные базы для аудио (dialog, manager, client).

        Логика:
        1. Проверка наличия файлов dialog.txt, manager.txt, client.txt
        2. Создание FAISS индексов через FaissService
        3. Отправка прогресса через WS (25%, 50%, 75%)
        4. Логирование результата
        """
        log = request.app.state.log
        ws = request.app.state.ws
        client = request.client.host

        files = ["dialog.txt", "manager.txt", "client.txt"]
        progress_steps = [25, 50, 75]

        # Проверка файлов
        for f in files:
            path = BASE_AUDIO_DIR / f
            if not path.exists():
                await log.log_warning(
                    target="load.index",
                    message=f"Required file not found: {path} for {client}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Файл для индексации отсутствует: {f}"
                )

        try:
            # Создание индексов в цикле
            for i, f in enumerate(files):
                path = BASE_AUDIO_DIR / f
                faiss_service = FaissService(text_file=f)
                faiss_service.rebuild_index()  # полное пересоздание индекса
                await log.log_info(
                    target="load.index",
                    message=f"Создание индекса для {f}"
                )

                # WS-прогресс
                if ws:
                    message = WSMessage(
                        type="progress",
                        code="index",
                        title="Создание индексных баз",
                        status="processing",
                        result=f"Индекс {f} создан",
                        progress=progress_steps[i],
                        is_complete=False
                    )
                    await ws.broadcast(message.model_dump())

            await log.log_info(
                target="load.index",
                message=f"Индексы успешно созданы для {client}"
            )

            # Финальный результат
            return IndexResponse(
                code="index",
                title="Создание индексных баз",
                status="success",
                result="Индексные базы созданы для dialog.txt, manager.txt, client.txt",
                progress=100
            )

        except Exception as e:
            await log.log_error(
                target="load.index",
                message=f"Ошибка при создании индексных баз: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при создании индексных баз: {str(e)}"
            )