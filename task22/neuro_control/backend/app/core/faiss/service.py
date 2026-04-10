# app/core/faiss/service.py

"""
Сервис для работы с FAISS-индексами текстовых баз знаний.

Этот модуль предоставляет класс TextFaissService для:

1. Загрузки текста из *.txt файлов
2. Разбиения текста на чанки фиксированного размера (chunk_size)
3. Построения FAISS индекса на основе эмбеддингов OpenAI
4. Сохранения и загрузки индекса с диска
5. Выполнения поиска по индексу
6. Полного пересоздания индекса (удаление + построение нового)

Используется в роутерах FAISS:
- /faiss/build
- /faiss/rebuild
- /faiss/search
"""

import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from ...config import settings

from dotenv import load_dotenv
load_dotenv()

# Корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Директория хранения текстовых файлов
BASE_AUDIO_DIR = BASE_DIR / "base" / "audio"

# Директория хранения FAISS индексов
FAISS_BASE_DIR = BASE_DIR / "base" / "faiss"

class FaissService:
    """
    Сервис для работы с FAISS-индексами текстовых баз знаний.

    Основные функции:
    - load_text       — чтение текстового файла
    - split_chunks    — разбиение текста на чанки фиксированного размера
    - build_index     — создание FAISS индекса и сохранение на диск
    - save_index      — сохранение индекса на диск
    - load_index      — загрузка индекса с диска или построение нового
    - search          — поиск по индексу
    - rebuild_index   — полное пересоздание индекса
    """

    def __init__(self, text_file: str, chunk_size: int = 1000, base_dir: str = "app/base/faiss"):
        """
        Инициализация сервиса.

        Параметры:
        - text_file (str): имя текстового файла базы знаний (*.txt)
        - chunk_size (int, optional): размер чанка в символах при разбиении текста
        - base_dir (str, optional): базовая директория для хранения FAISS индексов
        """
        # Путь к папке с текстовыми файлами
        self.text_file = BASE_AUDIO_DIR / text_file
        if not self.text_file.exists():
            raise FileNotFoundError(f"Файл базы знаний не найден: {self.text_file}")

        # Размер чанка в символах
        self.chunk_size = chunk_size

        # Путь к папке хранения индекса
        self.base_dir = FAISS_BASE_DIR
        self.index_name = self.text_file.stem
        self.index_path = self.base_dir / self.index_name
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Инициализация эмбеддингов и состояния индекса
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.LLM_API_KEY)
        self.index = None
        self.chunks = []

    def load_text(self) -> str:
        """Чтение текстового файла и возврат его содержимого в виде строки"""
        with open(self.text_file, "r", encoding="utf-8") as f:
            return f.read()

    def split_chunks(self, text: str):
        """
        Разбивает текст на чанки фиксированного размера self.chunk_size.

        Каждый чанк создается как объект Document для последующего построения FAISS индекса.

        Параметры:
        - text (str): текст из файла

        Возвращает:
        - List[Document]: список объектов Document, представляющих чанки
        """
        self.chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            self.chunks.append(Document(page_content=chunk_text))
            start = end
        return self.chunks

    def build_index(self):
        """
        Создание FAISS индекса из текстового файла и сохранение на диск.

        Процесс:
        1. Чтение текста из файла
        2. Разбиение на чанки
        3. Создание FAISS индекса
        4. Сохранение индекса в index_path

        Возвращает:
        - FAISS: объект индекса
        """
        text = self.load_text()
        self.split_chunks(text)
        self.index = FAISS.from_documents(self.chunks, self.embeddings)
        self.save_index()
        return self.index

    def save_index(self):
        """Сохраняет FAISS индекс в локальную папку index_path"""
        if self.index is None:
            raise ValueError("Индекс ещё не создан. Сначала вызови build_index()")
        self.index.save_local(str(self.index_path))

    def load_index(self):
        """
        Загружает индекс с диска, если он существует,
        иначе строит новый индекс на основе текстового файла.
        
        Возвращает:
        - FAISS: объект индекса
        """
        if (self.index_path / "index.faiss").exists():
            self.index = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.build_index()
        return self.index

    def search(self, query: str, k: int = 5):
        """
        Поиск по FAISS индексу.

        Параметры:
        - query (str): поисковый запрос
        - k (int): количество релевантных результатов

        Возвращает:
        - List[dict]: список словарей с ключами 'content' и 'score'
        """
        if self.index is None:
            self.load_index()
        results = self.index.similarity_search_with_score(query, k=k)
        return [{"content": doc.page_content, "score": float(score)} for doc, score in results]

    def rebuild_index(self):
        """
        Полное пересоздание индекса: удаление старого и создание нового.

        Возвращает:
        - FAISS: объект нового индекса
        """
        if self.index_path.exists():
            shutil.rmtree(self.index_path, ignore_errors=True)
        return self.build_index()