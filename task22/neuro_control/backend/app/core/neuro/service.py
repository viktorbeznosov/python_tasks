# app/core/neuro/service.py

"""
Сервисный модуль для модуля neuro.

Методы:
- start(request, file) — старт обработки файла (раньше было в route)
- step(code) — выполняет конкретный шаг (пользовательский или load)
- process_next(request) — выполняет следующий шаг из steps.txt
"""

from pathlib import Path
from typing import Dict, Optional, List
from fastapi import Request, UploadFile
from ..load.service import LoadService
from .schema import NeuroStartResponse, NeuroNextResponse
from ..websocket.schema import WSMessage
from ..gpt.service import call_gpt
from ..faiss.service import FaissService
import re

class NeuroService:

    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    STEPS_FILE = BASE_DIR / "base" / "steps.txt"
    PROMPT_DIR = BASE_DIR / "base" / "prompt"

    @staticmethod
    async def step(request: Request, code: str) -> Dict:
        """
        Выполняет конкретный шаг анализа

        Алгоритм:
        1. Загружает текущее состояние out из pickle.
        2. Читает список всех шагов из base/steps.txt.
        3. Находит индекс текущего шага и удаляет все последующие шаги из out.
        4. Сохраняет обновленный out через pickle.
        5. Логирование.
        6. Если шаг один из 'transcribe', 'detect', 'index' — выполняем через LoadService.
        7. Для всех остальных шагов выполняем имитацию.
        8. Возвращаем словарь с результатом шага:
        {
            code, title, status, result, progress, is_complete
        }
        """

        # ----------------------
        # 0. Получаем ссылки из app.state
        # ----------------------
        log = request.app.state.log
        pickle_client = request.app.state.pickle
        ws_manager = request.app.state.ws

        # ----------------------
        # 1. Загружаем текущее состояние out
        # ----------------------
        out = pickle_client.read() or {}
        await log.log_info(target="neuro_step", message=f"Step '{code}': состояние out загружено из pickle")

        # ----------------------
        # 2. Читаем все шаги из steps.txt
        # ----------------------
        steps_file = NeuroService.STEPS_FILE
        if not steps_file.exists():
            await log.log_error(target="neuro_step", message="Файл base/steps.txt не найден")
            raise FileNotFoundError("base/steps.txt не найден")

        steps = [line.strip() for line in steps_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        await log.log_info(target="neuro_step", message=f"Step '{code}': найдено {len(steps)} шагов")

        # ----------------------
        # 3. Находим индекс текущего шага
        # ----------------------
        try:
            idx = steps.index(code) 
        except ValueError:
            idx = -1
            await log.log_warning(target="neuro_step", message=f"Step '{code}' отсутствует в steps.txt")

        # ----------------------
        # 4. Удаляем результаты всех шагов после текущего
        # ----------------------
        if idx >= 0:
            for later_step in steps[idx + 1:]:
                if later_step in out:
                    del out[later_step]
            await log.log_info(target="neuro_step", message=f"Step '{code}': удалены последующие шаги из out")

        # ----------------------
        # 5. Сохраняем обновленный out
        # ----------------------
        pickle_client.save(out)
        await log.log_info(target="neuro_step", message=f"Step '{code}': состояние out сохранено в pickle")

        # ----------------------
        # 6. Проверяем, нужно ли выполнить шаг через LoadService
        # ----------------------
        if code in ("transcribe", "detect", "index"):
            await log.log_info(target="neuro_step", message=f"Step '{code}': выполняется через LoadService")

            if code == "transcribe":
                step_response = await LoadService.transcribe(request)
            elif code == "detect":
                step_response = await LoadService.detect(request)
            elif code == "index":
                step_response = await LoadService.index(request)

            # ----------------------
            # Преобразуем результат в словарь
            # ----------------------
            if hasattr(step_response, "model_dump"):
                result_dict = step_response.model_dump()
            elif isinstance(step_response, dict):
                result_dict = step_response
            else:
                result_dict = {"status": "success", "result": str(step_response)}

            # ----------------------
            # Сохраняем результат в out и pickle
            # ----------------------
            out[code] = result_dict
            pickle_client.save(out)
            await log.log_info(target="neuro_step", message=f"Step '{code}': результат сохранен в pickle через LoadService")

            # ----------------------
            # Возвращаем результат в формате NeuroNextResponse
            # ----------------------
            return {
                "code": code,
                "title": result_dict.get("title", code),
                "status": result_dict.get("status", "success"),
                "result": result_dict.get("result", ""),
                "progress": 0,
                "is_complete": code == steps[-1] if steps else True
            }

        # ----------------------
        # 7. Обработка для всех остальных шагов
        # ----------------------

        # Отправляем промежуточный прогресс 50%
        if ws_manager:
            message = WSMessage(
                type="progress",
                code=code,
                title=code,
                status="processing",
                progress=50,
                result='',
                is_complete=False
            )
            await ws_manager.broadcast(message.model_dump())

        # Выполняем шаг через GPT
        result_dict = await NeuroService.step_proc(request, code, out)

        result = {
            "code": result_dict["code"],
            "title": result_dict["title"],
            "status": result_dict.get("status", "success"),
            "result": result_dict.get("result", ""),
            "progress": 100,
            "is_complete": code == steps[-1] if steps else True
        }
        out[code] = result
        pickle_client.save(out)
        await log.log_info(target="neuro_step", message=f"Step '{code}': результат сформирован: {result}")

        # ----------------------
        # 8. Возвращаем результат
        # ----------------------
        return result

    @staticmethod
    async def step_proc(request, code: str, out: Dict) -> Dict:
        """
        Процессор шага анализа.

        Алгоритм:
        1. Загружает markdown-файл промпта по коду шага
        2. Парсит блок "## Параметры"
        3. Извлекает:
            - title   — название шага
            - index   — имя txt-файла для FAISS
            - user    — список предыдущих кодов
        4. Формирует prompt
        5. Вызывает GPT
        6. Сохраняет результат
        """

        log = request.app.state.log
        pickle_client = request.app.state.pickle

        # --------------------------------------------------
        # 1. Читаем файл промпта
        # --------------------------------------------------
        prompt_file = NeuroService.PROMPT_DIR / f"{code}.md"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Файл промпта не найден: {prompt_file}")

        text = prompt_file.read_text(encoding="utf-8")

        # --------------------------------------------------
        # 2. Парсим параметры
        # --------------------------------------------------
        title = code
        index_file: Optional[str] = None
        user_codes: Optional[List[str]] = None

        param_block = re.search(r"## Параметры(.*?)## system", text, re.S)

        if param_block:
            params_text = param_block.group(1)

            title_match = re.search(r"title:\s*(.+)", params_text)
            index_match = re.search(r"index:\s*(.+)", params_text)
            user_match = re.search(r"user:\s*(.+)", params_text)

            if title_match:
                title = title_match.group(1).strip()

            if index_match:
                index_file = index_match.group(1).strip() + '.txt'

            if user_match:
                user_codes = user_match.group(1).strip().split()

        # --------------------------------------------------
        # 3. Извлекаем system-промпт
        # --------------------------------------------------
        system_match = re.search(r"## system(.*)", text, re.S)
        if not system_match:
            raise ValueError(f"В файле {code}.md отсутствует блок '## system'")

        system_prompt = system_match.group(1).strip()

        # --------------------------------------------------
        # 4. Формируем данные для анализа
        # --------------------------------------------------
        user_content = ""

        # ---------- FAISS ----------
        if index_file:
            try:
                # Получаем несколько релевантных чанков
                faiss = FaissService(text_file=index_file)                
                results = faiss.search(query="анализ диалога", k=20)

                chunks = [r["content"] for r in results]
                user_content = "\n\n".join(chunks)

            except Exception as e:
                raise RuntimeError(f"Ошибка FAISS для шага {code}: {e}")

        # ---------- Предыдущие шаги ----------
        if user_codes:
            collected = []
            for uc in user_codes:
                if uc in out:
                    collected.append(out[uc]["result"])

            user_content = "\n\n".join(collected)

        # --------------------------------------------------
        # 5. Формируем финальный prompt
        # --------------------------------------------------
        
        separator = "\n\nДАННЫЕ ДЛЯ АНАЛИЗА:\n\n"

        full_prompt = system_prompt + separator + user_content

        # --------------------------------------------------
        # 6. Вызов GPT
        # --------------------------------------------------
        try:
            gpt_answer = await call_gpt(prompt=full_prompt, temperature=0.3, request=request)
        except Exception as e:
            raise RuntimeError(f"Ошибка GPT для шага {code}: {e}")

        result_text = gpt_answer.strip()

        # --------------------------------------------------
        # 7. Формируем результат
        # --------------------------------------------------
        result_dict = {
            "code": code,
            "title": title,
            "status": "success",
            "result": result_text,
            "progress": 100
        }

        # --------------------------------------------------
        # 8. Сохраняем
        # --------------------------------------------------
        out[code] = result_dict
        pickle_client.save(out)

        await log.log_info(
            target="neuro_step",
            message=f"Шаг '{code}' выполнен успешно"
        )

        return result_dict

    @staticmethod
    async def start(request: Request, file: UploadFile) -> NeuroStartResponse:
        """
        Начало анализа аудио

        Алгоритм:
        1. Вызывает LoadService.upload для сохранения файла
        2. Сохраняет результат в pickle (out['upload'])
        3. Возвращает NeuroStartResponse — готовую структуру для фронтенда
        """

        log = request.app.state.log                 # логгер приложения
        pickle_client = request.app.state.pickle    # клиент для сохранения состояния

        try:
            # ----------------------
            # 1. Загружаем файл через LoadService
            # ----------------------
            upload_response = await LoadService.upload(request, file)

            # ----------------------
            # 2. Сохраняем состояние в pickle
            # ----------------------
            # Сохраняем как словарь с ключом 'upload' для последующих шагов
            out = {"upload": upload_response.model_dump()}
            pickle_client.save(out)
            request.app.state.out = out

            # ----------------------
            # 3. Логирование успешного сохранения
            # ----------------------
            await log.log_info(
                target="neuro_start",
                message=f"Файл {file.filename} успешно загружен и сохранён в pickle"
            )

            # ----------------------
            # 4. Возвращаем результат в виде NeuroStartResponse
            # ----------------------
            # Добавляем поле is_complete=False, так как это стартовый шаг
            response = NeuroStartResponse(
                code=upload_response.code,                
                title=upload_response.title,
                status=upload_response.status,
                result=upload_response.result,
                progress=upload_response.progress,
                is_complete=False
            )
            return response

        except Exception as e:
            # ----------------------
            # 5. Логирование ошибки
            # ----------------------
            await log.log_error(
                target="neuro_start",
                message=f"Ошибка старта нейроанализа: {str(e)}"
            )
            raise

    @staticmethod
    async def process_next(request: Request) -> NeuroNextResponse:
        """
        Выполняет следующий шаг анализа по steps.txt

        Алгоритм:
        1. Читаем текущий словарь out из app.state.pickle
        2. Читаем base/steps.txt 
        3. Ищем первый шаг, которого ещё нет в out
        4. Если шаг transcribe/detect/index — вызываем соответствующий метод LoadService
        5. Иначе вызываем NeuroService.step(code)
        6. Сохраняем результат в out[code]
        7. Сохраняем out через pickle
        8. Возвращаем результат с is_complete=True только если выполнен последний шаг
        """
        log = request.app.state.log
        pickle_client = request.app.state.pickle

        # ----------------------
        # 1. Загружаем состояние
        # ----------------------
        out = pickle_client.read() or {}
        await log.log_info(target="neuro_next", message="Состояние out загружено из pickle")

        # ----------------------
        # 2. Читаем шаги
        # ----------------------
        if not NeuroService.STEPS_FILE.exists():
            await log.log_error(target="neuro_next", message="Файл steps.txt не найден")
            raise FileNotFoundError("base/steps.txt не найден")

        steps = [line.strip() for line in NeuroService.STEPS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

        # ----------------------
        # 3. Находим первый непроработанный шаг
        # ----------------------
        next_step = None
        for step_code in steps:
            if step_code not in out:
                next_step = step_code
                break

        if not next_step:
            await log.log_info(target="neuro_next", message="Все шаги уже выполнены")
            return NeuroNextResponse(
                title="Neuro Next",
                status="success",
                result="Все шаги уже выполнены",
                progress=100,
                is_complete=True
            )

        await log.log_info(target="neuro_next", message=f"Выполняется шаг: {next_step}")

        # ----------------------
        # 4. Выполняем шаг
        # ----------------------
        if next_step == "transcribe":
            step_response = await LoadService.transcribe(request)
        elif next_step == "detect":
            step_response = await LoadService.detect(request)
        elif next_step == "index":
            step_response = await LoadService.index(request)
        else:
            step_response = await NeuroService.step(request, next_step)

        # ----------------------
        # 5. Преобразуем в словарь для безопасного обращения
        # ----------------------
        if hasattr(step_response, "model_dump"):
            result_dict = step_response.model_dump()
        elif isinstance(step_response, dict):
            result_dict = step_response
        else:
            result_dict = {"status": "success", "result": str(step_response)}

        # ----------------------
        # 6. Сохраняем результат в out
        # ----------------------
        out[next_step] = result_dict

        # ----------------------
        # 7. Сохраняем состояние через pickle
        # ----------------------
        pickle_client.save(out)
        await log.log_info(target="neuro_next", message=f"Шаг {next_step} завершен и сохранен в pickle")

        # ----------------------
        # 8. Проверяем, выполнен ли последний шаг
        # ----------------------
        is_complete = next_step == steps[-1]

        # ----------------------
        # 9. Возвращаем результат
        # ----------------------
        return NeuroNextResponse(
            code=next_step,
            title=result_dict.get("title", next_step),  
            status=result_dict.get("status", "success"),
            result=result_dict.get("result", ""),
            progress=0,
            is_complete=is_complete
        )

    @staticmethod
    async def get_state(request: Request) -> Dict:
        """
        Возвращает текущее состояние out из pickle.

        Алгоритм:
        1. Читаем состояние через pickle.read()
        2. Если пусто — возвращаем {}
        3. Логируем операцию
        4. Возвращаем out
        """
        log = request.app.state.log
        pickle_client = request.app.state.pickle

        # 1. Читаем состояние
        out = pickle_client.read() or {}

        # 2. Логирование
        await log.log_info(
            target="neuro_state",
            message="Состояние out прочитано из pickle"
        )

        # 3. Возвращаем состояние
        return out

    @staticmethod
    async def auto(request: Request, code: str) -> Dict:
        """
        Автоматически выполняет все шаги начиная с указанного кода
        до конца списка steps.txt.
        Если code == '' или не найден, начинает с первого невыполненного шага.
        """

        log = request.app.state.log
        ws_manager = request.app.state.ws
        pickle_client = request.app.state.pickle

        await log.log_info(
            target="neuro_auto",
            message=f"Запуск авто-режима начиная с шага: '{code}'"
        )

        # ----------------------
        # Проверяем наличие файла steps.txt
        # ----------------------
        if not NeuroService.STEPS_FILE.exists():
            await log.log_error(target="neuro_auto", message="Файл base/steps.txt не найден")
            raise FileNotFoundError("base/steps.txt не найден")

        # ----------------------
        # Читаем список шагов
        # ----------------------
        steps = [
            line.strip()
            for line in NeuroService.STEPS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # ----------------------
        # Загружаем текущее состояние out
        # ----------------------
        out = pickle_client.read() or {}

        # ----------------------
        # Определяем стартовый шаг
        # ----------------------
        start_index = 0
        if code and code in steps:
            start_index = steps.index(code)
        else:
            # ищем первый невыполненный шаг
            next_step = None
            for idx, step_code in enumerate(steps):
                if step_code not in out:
                    next_step = step_code
                    start_index = idx
                    break
            if not next_step:
                await log.log_info(target="neuro_auto", message="Все шаги уже выполнены")
                return {
                    "status": "success",
                    "started_from": code,
                    "is_complete": True
                }

        remaining_steps = steps[start_index:]
        await log.log_info(target="neuro_auto", message=f"К выполнению: {remaining_steps}")

        # ----------------------
        # Выполняем шаги последовательно
        # ----------------------
        for step_code in remaining_steps:

            await log.log_info(target="neuro_auto", message=f"Начало выполнения шага: {step_code}")
            message = WSMessage(
                type="start",
                code=step_code,
                title=step_code.capitalize(),
                status="started",
                progress=0,
                is_complete=False
            )
            await ws_manager.broadcast(message.model_dump())

            # Выполнение шага
            response = await NeuroService.step(request, step_code)

            # Сохраняем результат в out и pickle
            out[step_code] = response
            pickle_client.save(out)

            # WS сообщение с результатом
            message = WSMessage(
                type="end",
                code=response.get("code", step_code),
                title=response.get("title", step_code.capitalize()),
                status=response.get("status", "success"),
                result=response.get("result"),
                progress=response.get("progress", 0),
                is_complete=response.get("is_complete", True)
            )
            await ws_manager.broadcast(message.model_dump())
            await log.log_info(target="neuro_auto", message=f"Шаг {step_code} завершен и сохранен в pickle")

        await log.log_info(target="neuro_auto", message="Авто-режим завершен успешно")

        return {
            "status": "success",
            "started_from": code or remaining_steps[0],
            "is_complete": True
        }
