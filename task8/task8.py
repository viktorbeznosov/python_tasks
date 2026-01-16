from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
import re
import requests
import textwrap
from glob import glob
import matplotlib.pyplot as plt
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
import os
from dotenv import load_dotenv

# Функция для загрузки .env переменных дял OpenAI
def load_openai_config():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", default='https://api.proxyapi.ru/openai/v1')

    if not api_key:
        print("Ошибка: Переменная окружения OPENAI_API_KEY не найдена.")
        print("Создайте файл .env в корне проекта и добавьте:")
        print("OPENAI_API_KEY=ваш_ключ_здесь")
        return False

    return {'api_key': api_key, 'base_url': base_url}

# функция для загрузки документа Docx по ссылке из гугл драйв
def download_google_doc(url: str) -> str:
    # Извлекаем ID документа из URL
    match_ = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
    if not match_:
        raise ValueError("Invalid Google Docs URL")
    doc_id = match_.group(1)

    try:        
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()
        text = response.text
    except requests.exceptions.HTTPError as e:
         print(f"HTTP ошибка: {e}")
         return False
    except requests.exceptions.Timeout:
        print("Таймаут соединения")
        return False
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        return False

    return text

# Функция для форматирования текста по абзацам
def format_text(text, width=120):
    # Разделяем текст на абзацы
    paragraphs = text.split('\n')
    # Форматируем каждый абзац отдельно
    formatted_paragraphs = []
    for paragraph in paragraphs:
        # Используем textwrap.fill для форматирования абзаца, чтобы длина строки не превышала width
        formatted_paragraph = textwrap.fill(paragraph, width)
        formatted_paragraphs.append(formatted_paragraph)
    # Объединяем абзацы с символом новой строки
    return '\n'.join(formatted_paragraphs)

def split_kasko_text(text: str,
                    chunk_size=900,  # Ограничение к-ва символов в чанке
                    chunk_overlap=150): # к-во символов перекрытия в чанке
    # Удалить пустые строки и лишние пробелы
    text = re.sub(r' {1,}', ' ', re.sub(r'\n\s*\n', '\n', text))
    # Создаем экземпляр RecursiveCharacterTextSplitter с заданными парамаетрами
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                                separators=[
                                    "# ",   # Сначала по H1
                                    "## "   # Затем по H2        
                                    "### ", # Затем по H3
                                    "\n",   # Затем по одинарным
                                    " ",    # Затем по пробелам
                                ]
                            )
    # Разбиваем текст на чанки с помощью созданного сплиттера
    chunks = text_splitter.split_text(text)
    # Преобразование фрагментов текста в формат LangChain Document (без метаданных)
    chunks = [Document(page_content=chunk) for chunk in chunks]
    return chunks # Возвращаем список чанков в формате LangChain Document

# Создание индексной (векторной) базы из чанков в формате LangChain Document и сохранение на диск
def create_db_index_from_documents_save(chunks_documents,
                                        index_name, # имя для индексной базы
                                        path):  # путь к папке
    # Создаем индексную базу с использованием FAISS ---------------------------------
    db_index = FAISS.from_documents(chunks_documents,
                                    OpenAIEmbeddings())
    # сохраняем индексную базу ------------------------------------------------------
    db_index.save_local(folder_path=path, # путь к папке (path)
                        index_name=index_name)  # имя для индексной базы (index_name)
    # --------------------------------------------------------------------------------
    return db_index

# Загрузка векторной базы с диска
def load_db_vector(folder_path_db_index,  # путь к сохраненной векторной базе
                   index_name):           # имя сохраненной векторной базы
    return FAISS.load_local(
                allow_dangerous_deserialization=True, # Разрешает потенциально небезопасную десериализацию
                embeddings=OpenAIEmbeddings(),  # Указывает векторные представления
                folder_path=folder_path_db_index, # путь к сохраненной векторной базе
                index_name=index_name) # имя сохраненной векторной базы

# Загружаем ранее сохраненные на ГуглДиске векторные базы из папки и объединяем в одну
def get_dbs_vector(path_dbs, verbose=False):
    # Для объединения нескольких баз в цикле будем присоединять существующие базы к новой (пустой)
    # Создание пустой векторной базы -----------------------------------------------------
    db_index_new = FAISS.from_documents([Document(page_content='', metadata={})],
                                        OpenAIEmbeddings())  # FAISS.from_documents
    # или для текстового формата чанков:
    # db_index_new = FAISS.from_texts([''], OpenAIEmbeddings())  # FAISS.from_texts
    # ------------------------------------------------------------------------------------
    # пробегаем по всем файлам векторных баз в нашей папке
    for file in glob(f'{path_dbs}*.faiss'):
        index_name = file.split('/')[-1].split('.')[0]  # Имя векторной базы на диске
        # Загрузка ранее сохраненной векторной базы
        db_index_loaded = load_db_vector(path_dbs, index_name)
        if (verbose):
            print(f'Векторная база: "{index_name}" загружена')
        # Слияние баз в одну db_index_new ------------------
        db_index_new.merge_from(db_index_loaded)
        # --------------------------------------------------
    return db_index_new

# Обернем в функцию
# Создание цепочки модели с использованием ретривера для поиска по векторной базе данных
def create_model_chain(db_index, # векторная база знаний
                       k=3,      # используемое к-во чанков
                       model='gpt-4o-mini',
                       temp=0.1):
    try:
        llm = ChatOpenAI(model=model, temperature=temp)
        retriever = db_index.as_retriever(search_type="similarity",
                                        search_kwargs={"k": k})
        system_prompt = """Ответь на вопрос пользователя используя отрезки текста.
        Context: {context}"""
        prompt = ChatPromptTemplate.from_messages(
                                    [("system", system_prompt),
                                    ("human", "{input}")])
        return create_retrieval_chain(retriever,
                                    create_stuff_documents_chain(llm, prompt))
    except Exception as e:
        print(f"Ошибка при создании цепочки: {e}")
        return None


def generate_answer(index, query):
    # Используем c векторной базой данных `db_postoplan_merged`
    new_chain = create_model_chain(index)

    ans = new_chain.invoke({"input": query})

    return ans

def neuro_consultant(index):
    history = ""
    while True:
        query = input('Вопрос пользователя: ')
        if query == 'стоп': break
        query += f"""
        История диалога: {history}
        """
        # выход из цикла, если пользователь ввел: 'стоп'
        # ответ от OpenAI
        answer = generate_answer(index, query)
        print(f'Ответ:\n{format_text(answer['answer'])}\n')
        # Запись истории диалога
        history += f'Вопрос пользователя: {answer['input']}. \nОтвет: {answer['answer']}\n'

url_base1_simble = 'https://docs.google.com/document/d/1Z7eZLIPG9URgOFz-yqtJAup-WXhKFIiF'
url_base2_simble = 'https://docs.google.com/document/d/1qxJXwHtYNxx6ecf35zhqFYBxSjoA5Mhr'

load_openai_config()

path_db_index = './db_index_kasko/'

# Базы знаний Kasko
for i, url in enumerate([url_base1_simble,
                         url_base2_simble]):
    # Скачиваем базу знаний (с разбивкой Markdown) по ссылке
    text_content = download_google_doc(url)
    # Создание чанков из Markdown документа
    chunks = split_kasko_text(text_content)
    # Создание векторной базы и сохранение на диск
    create_db_index_from_documents_save(chunks,
                                        f'kasko_{i+1}', # имя для индексной базы
                                        path_db_index)  # ранее созданный путь на ГуглДиске
    
# Создаем объединенную векторную базу Postoplan
db_kasko_merged = get_dbs_vector(path_db_index)

neuro_consultant(db_kasko_merged)

'''
Примеры вопросов и ответов

Вопрос пользователя: Как оформить каско за 5 минут?
Ответ:
Чтобы оформить КАСКО за 5 минут, выполните следующие шаги:

1. Введите данные вашего автомобиля и водителей. Вам понадобятся госномер, СТС, ПТС, ваш паспорт и водительское
удостоверение.
2. Получите и выберите лучшее предложение. Страховые компании рассчитают цены специально для вас, и вы сможете их
сравнить.
3. Оплатите КАСКО онлайн. Это быстро и безопасно, без наценок.
4. Получите полис на ваш e-mail. Его можно сохранить на телефон и использовать как обычный бумажный полис.

Также вы можете воспользоваться приложением, которое поможет в общении со страховой и ответит на вопросы по тарифам.

Вопрос пользователя: Какой минимальный срок оформления каско?
Ответ:
Минимальный срок оформления КАСКО не указан в предоставленном контексте. Однако, КАСКО на короткий срок позволяет
оформить полис на нужный вам период, например, на время поездки или парковки, что может быть удобным для тех, кто не
хочет переплачивать за годовое КАСКО.

Вопрос пользователя: Как оформить каско онлайн?
Ответ:
Чтобы оформить КАСКО онлайн, выполните следующие шаги:

1. Введите данные вашего автомобиля и водителей. Вам понадобятся госномер, СТС, ПТС, ваш паспорт и водительское
удостоверение.
2. Получите и выберите лучшее предложение. Страховые компании рассчитают цены специально для вас, и вы сможете их
сравнить.
3. Оплатите КАСКО онлайн. Это быстро и безопасно, без наценок.
4. Получите полис на ваш e-mail. Его можно сохранить на телефон и использовать как обычный бумажный полис.

Также вы можете воспользоваться приложением, которое поможет в общении со страховой и ответит на вопросы по тарифам.

Вопрос пользователя: Какие вопросы я задавал?
Ответ:
Вы задавали следующие вопросы:

1. Как оформить каско за 5 минут?
2. Какой минимальный срок оформления каско?
3. Как оформить каско онлайн?
'''