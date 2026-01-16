from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import requests
import re
from langchain.docstore.document import Document
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", default='https://api.proxyapi.ru/openai/v1')

def load_document_text(url):
    # Extract the document ID from the URL
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)
    # Download the document as plain text
    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    text = response.text
    return text

# Загружаем первую БЗ
db_1=load_document_text("https://docs.google.com/document/d/1CadgV8oI_MgYZBfaIEa5vAnrQXyme1qK-qIHFwlZL58/edit?usp=sharing")
# print(db_1[:800])

# Загружаем 2 БЗ:
db_2=load_document_text("https://docs.google.com/document/d/1VnnEk1-DASWaBUa5dMTeZvuAMjqgpY_K86trhCvx5TI/edit?usp=sharing")
# print(db_2[:800])

# Делим на чанки 1 БЗ:
db_1_chunks=[]
splitter = CharacterTextSplitter(separator='<Chunk>')
for chunk in splitter.split_text(db_1):
    db_1_chunks.append(Document(page_content=chunk, metadata={"meta":"data"}))
# print("Общее количество чанков: ", len(db_1_chunks))

# Делим на чанки 2 БЗ:
db_2_chunks=[]
splitter = CharacterTextSplitter(separator='<Chunk>')
for chunk in splitter.split_text(db_2):
    db_2_chunks.append(Document(page_content=chunk, metadata={"meta":"data"}))
# print("Общее количество чанков: ", len(db_2_chunks))

# Создаем 2 векторные базы:
embeddings = OpenAIEmbeddings()
db_1_faiss = FAISS.from_documents(db_1_chunks, embeddings)
db_2_faiss = FAISS.from_documents(db_2_chunks, embeddings)

# Вариант 2

# Создание пустой векторной базы (один пустой чанк)
merged_db = FAISS.from_documents([Document(page_content='', metadata={})], embeddings)

# Этот вариант удобен, если объединять базы, добавляя новые в цикле
for db in [db_1_faiss, db_2_faiss]:
    merged_db.merge_from(db)

# вот так можно посмотреть на объединенную базу (на все чанки)
print(len(merged_db.docstore._dict))
print(merged_db.docstore._dict)