import os
import re
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import logging



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  

# Функция загрузки документа
def load_document_text(url: str) -> str:
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)

    # Загрузка документа и преобразование в текст
    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    text = response.text
    return text


# Функция деления на чанки
def split_text(text: str, chunk_size=1024, chunk_overlap=100):
    """
    Разделение текста на чанки с метаданными.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source_chunks = []
    for chunk in splitter.split_text(text):
        # Добавляем метаданные к каждому чанку
        source_chunks.append(Document(page_content=chunk, metadata={"meta": "data"}))
    return source_chunks


# Функция создания индексной базы
def create_db_index(url: str, path: str = '', index_name: str = 'db_index'):
    """
    Создание индексной базы знаний из текста по URL.
    """
    text = load_document_text(url)
    chunks = split_text(text)
    db_index = FAISS.from_documents(chunks, OpenAIEmbeddings())
    db_index.save_local(folder_path=path, index_name=index_name)
    logging.info(f"Индекс '{index_name}' сохранен в '{path}'.")

# Функция загрузки ранее сохраненной индексной базы знаний.
def load_db_index(index_name='db_index', path=''):

    return FAISS.load_local(
        folder_path=path,
        embeddings=OpenAIEmbeddings(),
        index_name=index_name,
        allow_dangerous_deserialization=True
    )

