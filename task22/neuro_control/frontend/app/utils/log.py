# app/utils/log.py
# Логирование событий

import os
import datetime
from aiologger import Logger
from aiologger.handlers.files import AsyncFileHandler
import logging

class Log:
    def __init__(self):
        self.log_dir = "log"
        os.makedirs(self.log_dir, exist_ok=True)
        self.handlers = {}
        self.log_print = os.getenv("LOG_PRINT", "0").lower() in ("1", "true", "yes")

    def build_log_path(self, target: str, now: datetime.datetime) -> str:
        """
        Формируем путь к лог-файлу:
        app/log/2025/10/04.log
        """
        year = f"{now.year}"
        month = f"{now:%m}"
        day = f"{now:%d}"

        base_dir = os.path.join(self.log_dir, year, month)
        os.makedirs(base_dir, exist_ok=True)

        filename = f"{day}.log"
        return os.path.join(base_dir, filename)

    async def get_logger(self, target: str, now: datetime.datetime) -> Logger:
        """Асинхронный логгер для target."""
        log_path = self.build_log_path(target, now)

        if target not in self.handlers or self.handlers[target]["path"] != log_path:
            handler = AsyncFileHandler(filename=log_path, mode="a", encoding="utf-8")
            target_logger = Logger(name=f"logger_{target}")
            target_logger.add_handler(handler)

            if target in self.handlers:
                try:
                    await self.handlers[target]["logger"].shutdown()
                except Exception:
                    pass

            self.handlers[target] = {
                "path": log_path,
                "logger": target_logger,
                "handler": handler,
            }

        return self.handlers[target]["logger"]

    async def unescape_newlines(self, obj):
        """Рекурсивно заменяет '\\n' на реальные переносы."""
        if isinstance(obj, dict):
            return {k: await self.unescape_newlines(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [await self.unescape_newlines(x) for x in obj]
        elif isinstance(obj, str):
            return obj.replace("\\n", "\n")
        else:
            return obj

    # Асинхронное
    async def log_info(
        self,
        target: str = "",
        message: str = "",
        data: dict | None = None,
        is_console: bool = None,
    ):
        if data is None:
            data = {}

        now = datetime.datetime.now()

        log_data_str = f"{now:%d.%m.%Y %H:%M:%S} {target}: {message}"
        if data:
            log_data_str += f": {self.safe_serialize(data)}"

        target_logger = await self.get_logger(target, now)
        await target_logger.info(log_data_str)

        should_print = self.log_print if is_console is None else is_console
        if should_print:
            print(log_data_str)

    async def log_error(
        self, target: str = "", message: str = "", data: dict | None = None, is_console: bool = True
    ):
        await self.log_info(target, f"ERROR: {message}", data, is_console)

    # Синхронное
    def log_info_sync(
        self, target: str = "", message: str = "", data: dict | None = None, is_console: bool = None
    ):
        if data is None:
            data = {}

        now = datetime.datetime.now()
        log_path = self.build_log_path(target, now)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        log_data_str = f"{now:%d.%m.%Y %H:%M:%S} {target}: {message}"
        if data:
            log_data_str += f": {self.safe_serialize(data)}"

        logger = logging.getLogger(f"sync_logger_{target}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.info(log_data_str)

        should_print = self.log_print if is_console is None else is_console
        if should_print:
            print(log_data_str)

    def log_error_sync(
        self, target: str = "", message: str = "", data: dict | None = None, is_console: bool = None
    ):
        self.log_info_sync(target, f"ERROR: {message}", data, is_console)

    def safe_serialize(self, obj):
        """
        Преобразуем объект в сериализуемый вид для JSON/log:
        - dict, list, tuple рекурсивно
        - Pydantic модели через model_dump
        - любые несериализуемые объекты → строка с типом
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: self.safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [self.safe_serialize(v) for v in obj]
        elif hasattr(obj, "model_dump"):  # Pydantic
            return self.safe_serialize(obj.model_dump())
        elif hasattr(obj, "__dict__"):
            # игнорируем приватные атрибуты
            return {k: self.safe_serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
        else:
            # для всего остального выводим тип объекта
            return f"<{type(obj).__name__}>"

    async def shutdown(self):
        for h in list(self.handlers.values()):
            try:
                await h["logger"].shutdown()
            except Exception:
                pass

    # WARNING: асинхронный вариант
    async def log_warning(self, target: str = "", message: str = "", data: dict | None = None, is_console: bool = None):
        await self.log_info(target, f"WARNING: {message}", data, is_console)

    # WARNING: cинхронный вариант
    def log_warning_sync(self, target: str = "", message: str = "", data: dict | None = None, is_console: bool = None):
        self.log_info_sync(target, f"WARNING: {message}", data, is_console)