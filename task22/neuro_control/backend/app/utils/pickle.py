# app/utils/pickle.py

"""
Утилита для работы с pickle-файлом.
Файл хранится в base/dump.pkl
Методы:
- read()  — возвращает данные из файла
- save(data) — сохраняет данные в файл
- clear() — очищает файл (удаляет содержимое)
"""

import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "base"
BASE_DIR.mkdir(parents=True, exist_ok=True)

DUMP_FILE = BASE_DIR / "dump.pkl"


class PickleStorage:
    """Класс для работы с pickle-хранилищем"""

    def read(self):
        """Читает объект из файла, возвращает None если файла нет"""
        if not DUMP_FILE.exists():
            return None
        try:
            with DUMP_FILE.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def save(self, data):
        """Сохраняет объект в файл"""
        with DUMP_FILE.open("wb") as f:
            pickle.dump(data, f)

    def clear(self):
        """Удаляет файл или очищает содержимое"""
        if DUMP_FILE.exists():
            DUMP_FILE.unlink()
