# импортируем необходимые библиотеки
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import (MarkdownHeaderTextSplitter,
                                     RecursiveCharacterTextSplitter,
                                     CharacterTextSplitter)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
import re
import requests
import textwrap
import openai
import tiktoken
from glob import glob
import matplotlib.pyplot as plt
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
import os
from dotenv import load_dotenv

# Служебные функции

# функция для загрузки документа Docx по ссылке из гугл драйв
def download_google_doc(url: str) -> str:
    # Извлекаем ID документа из URL
    match_ = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
    if not match_:
        raise ValueError("Invalid Google Docs URL")
    doc_id = match_.group(1)
    download_url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'
    response = requests.get(download_url, stream=True)
    if response.status_code != 200:
        raise RuntimeError("Failed to download the document")
    return response.text

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


# Функция возвращает количество токенов в строке в зависимости от используемой модели
def num_tokens_from_string(string: str, model='gpt-4o-mini') -> int:
    # Получаем имя кодировки для указанной модели
    encoding_name = tiktoken.encoding_for_model(model).name
    # Получаем объект кодировки на основе имени кодировки
    encoding = tiktoken.get_encoding(encoding_name)
    # Кодируем строку и вычисляем количество токенов
    num_tokens = len(encoding.encode(string))
    # Возвращаем количество токенов
    return num_tokens


# Построение гистограммы распределения количества токенов по чанкам
def create_histogram(chunks):
    print("\nОбщее количество чанков: ", len(chunks))
    # Подсчет токенов для каждого чанка
    try: # для формата чанков LangChain Document
        chunk_token_counts = [num_tokens_from_string(chunk.page_content) for chunk in chunks]
    except: # для текстового формата чанков
        chunk_token_counts = [num_tokens_from_string(chunk) for chunk in chunks]
    # Строим гистограмму
    plt.figure(figsize=(7, 4)) # размер
    plt.hist(chunk_token_counts, bins=10, alpha=0.5, label='Чанки')
    plt.title('Распределение к-ва токенов по чанкам')  # Заголовок графика
    plt.xlabel('К-во токенов в чанке')  # Подпись оси X
    plt.ylabel('К-во чанков')  # Подпись оси Y
    plt.show()  # Отображаем график

    # Сохраняем график в файл
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
    plt.savefig('histogram.pdf')  # или в PDF
    print("График сохранен как 'histogram.png'")
    plt.close()  # закрываем фигуру для освобождения памяти

# (MarkdownHeaderTextSplitter)
# Формируем чанки в формат LangChain Document из текста с Markdown разметкой
def split_markdown_text(markdown_text,
                        strip_headers=False): # НЕ удалять заголовки под '#..' из page_content
    # Удалить пустые строки и лишние пробелы
    markdown_text = re.sub(r' {1,}', ' ', re.sub(r'\n\s*\n', '\n', markdown_text))
    # Определяем заголовки, по которым будем разбивать текст
    headers_to_split_on = [("#", "Header 1"),   # Заголовок первого уровня
                           ("##", "Header 2"),  # Заголовок второго уровня
                           ("###", "Header 3")] # Заголовок третьего уровня
    # Создаем экземпляр MarkdownHeaderTextSplitter с заданными заголовками
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on,
                                                   strip_headers=strip_headers)
    # Разбиваем текст на чанки в формат LangChain Document
    chunks = markdown_splitter.split_text(markdown_text)
    return chunks # Возвращаем список чанков в формате LangChain Document

# (RecursiveCharacterTextSplitter)
# Формируем чанки в формат LangChain Document из текста по количеству символов
def split_hard_text(text: str,
                    chunk_size=3000,  # Ограничение к-ва символов в чанке
                    chunk_overlap=100): # к-во символов перекрытия в чанке
    # Удалить пустые строки и лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Создаем экземпляр RecursiveCharacterTextSplitter с заданными парамаетрами
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            # separators=["\n\n", "\n", " "]  # Приоритет разделителей
                            )
    # Разбиваем текст на чанки с помощью созданного сплиттера
    chunks = text_splitter.split_text(text)
    # Преобразование фрагментов текста в формат LangChain Document (без метаданных)
    chunks = [Document(page_content=chunk) for chunk in chunks]
    return chunks # Возвращаем список чанков в формате LangChain Document

# (CharacterTextSplitter) Формируем чанки из текста по количеству символов
def split_text(text: str,
               chunk_size=2000,    # Ограничение к-ва символов в чанке
               chunk_overlap=200): # к-во символов перекрытия в чанке
    # Удалить пустые строки и лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Создаем экземпляр CharacterTextSplitter с заданными парамаетрами
    splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     separator=" ") # разделитель по словам (по пробелу)
    return splitter.split_text(text) # список текстовых чанков

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
def get_dbs_vector(path_dbs):
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
        print(f'Векторная база: "{index_name}" загружена')
        # Слияние баз в одну db_index_new ------------------
        db_index_new.merge_from(db_index_loaded)
        # --------------------------------------------------
    return db_index_new

# Функция запроса и ответа от OpenAI с поиском по векторной базе данных
def generate_answer(query,    # запрос пользователя
                    db_index, # векторная база знаний
                    k=5,      # используемое к-во чанков
                    verbose=True, # выводить ли на экран выбранные чанки
                    model='gpt-4o-mini',
                    temp=0.1):
    # Поиск чанков по векторной базе данных
    similar_documents = db_index.similarity_search(query, k=k)
    # Формирование текстового контента из выбранных чанков для модели
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'Отрывок документа № {i+1}:\n' + doc.page_content
        for i, doc in enumerate(similar_documents)]))
    if verbose:
        print(message_content) # печать на экран выбранных чанков

    messages = [{"role": "system", "content":
                 f'Ответь на вопрос пользователя на основании информации из базы знаний: \n{message_content}'},
                {"role": "user", "content":
                 f'Вопрос пользователя: {query}'}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp)
    return response.choices[0].message.content

# Создание цепочки модели с использованием ретривера для поиска по векторной базе данных
def create_model_chain(db_index, # векторная база знаний
                       k=3,      # используемое к-во чанков
                       model='gpt-4o-mini',
                       temp=0.1):
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

# Ссылки на базы знаний, которые будем использовать:

# База знаний компании Postoplan - часть 1 (с разбивкой MarkDown)
url_base1_postoplan = 'https://docs.google.com/document/d/13WKOhsA2H7nWjZkuy2wDJ2oDv2Y-m8yt'
# База знаний компании Postoplan - часть 2 (с разбивкой MarkDown)
url_base2_postoplan = 'https://docs.google.com/document/d/1UDNfJvWGPQRz97Xnzf-y3AcEWtFJFT7U'

# База знаний компании Simble - часть 1 (без разбивки MarkDown)
url_base1_simble = 'https://docs.google.com/document/d/1Z7eZLIPG9URgOFz-yqtJAup-WXhKFIiF'
# База знаний компании Simble - часть 2 (без разбивки MarkDown)
url_base2_simble = 'https://docs.google.com/document/d/1qxJXwHtYNxx6ecf35zhqFYBxSjoA5Mhr'

# # Скачиваем базу знаний (.docx с разбивкой Markdown) по ссылке
# text_md_content = download_google_doc(url_base1_postoplan)

# # Пример создания чанков
# chunks_md = split_markdown_text(text_md_content)

# # Пример одного чанка
# print(f'\npage_content: \n{chunks_md[5].page_content}')
# print(f'\nmetadata: \n{chunks_md[5].metadata}')
# Гистограмма распределения количества токенов по чанкам
# create_histogram(chunks_md)

# # Скачиваем базу знаний (без разбивки Markdown) по ссылке
# text_content = download_google_doc(url_base1_simble)
# # Пример создания чанков
# chunks = split_hard_text(text_content)

# Пример одного чанка
# print(f'\npage_content: \n{format_text(chunks[0].page_content)}')
# Поле metedata не использовалось
# Гистограмма распределения количества токенов по чанкам
# create_histogram(chunks)

load_dotenv()

# Инициализирум модель эмбеддингов
embeddings = OpenAIEmbeddings()

path_db_index_md = './db_index_md/'     # для векторных баз из Markdown документов
path_db_index_text = './db_index_text/' # для векторных баз из документов без MD разбивки

##########################################################################################################################

# # Создаем индексные (векторные) базы и сохраняем на Диск

# # Базы знаний Postoplan
# for i, url in enumerate([url_base1_postoplan,
#                          url_base2_postoplan]):
#     # Скачиваем базу знаний (с разбивкой Markdown) по ссылке
#     text_md_content = download_google_doc(url)
#     # Создание чанков из Markdown документа
#     chunks_md = split_markdown_text(text_md_content)
#     # Создание векторной базы и сохранение на диск
#     create_db_index_from_documents_save(chunks_md,
#                                         f'postoplan_{i+1}', # имя для индексной базы
#                                         path_db_index_md)  # ранее созданный путь на ГуглДиске


# # Базы знаний Simble
# for i, url in enumerate([url_base1_simble,
#                          url_base2_simble]):
#     # Скачиваем базу знаний по ссылке
#     text_content = download_google_doc(url)
#     # Создание чанков из документа
#     chunks = split_hard_text(text_content)
#     # Создание векторной базы и сохранение на диск
#     create_db_index_from_documents_save(chunks,
#                                         f'simble_{i+1}', # имя для индексной базы
#                                         path_db_index_text)  # ранее созданный путь на ГуглДиске

##########################################################################################################################

# ids = db.add_texts(["Пример текста 1", "Пример текста 2"],
#                    metadatas=[{"author": "user1"}, {"author": "user2"}])

# # в итоге получаем ID добавленных чанков:
# print(ids)

##########################################################################################################################

# # Предположим, у нас есть некоторый список документов следующего вида:
# documents = [
#     Document(page_content="Текст документа 1", metadata={"author": "Автор 1"}),
#     Document(page_content="Текст документа 2", metadata={"author": "Автор 2"})
# ]
# # Используем метод add_documents для добавления документов в хранилище
# added_ids = db_md.add_documents(documents)

# # Теперь added_ids содержит идентификаторы добавленных текстов
# print(added_ids)

##########################################################################################################################

# # similarity_search
# similar_documents = db_md.similarity_search("Интересные факты о маркетинге социальных сетей", k=3)

# message_content = re.sub(r'\n{2}', ' ', '\n '.join(
#     [f'\nОтрывок документа № {i+1}:\n' + doc.page_content + f'\nMetedata: {doc.metadata}'
#      for i, doc in enumerate(similar_documents)]))
# print(format_text(message_content))

##########################################################################################################################

# # similarity_search_with_score
# similar_documents = db_md.similarity_search_with_score("Интересные факты о маркетинге социальных сетей", k=3)

# message_content = re.sub(r'\n{2}', ' ', '\n '.join(
#     [f'\nОтрывок документа № {i+1}:\n' + doc.page_content + f'\nMetedata: {doc.metadata}' + f'\nScore: {score}'
#      for i, (doc, score) in enumerate(similar_documents)]))
# print(format_text(message_content))

# # max_marginal_relevance_search
# # найти документы, максимально релевантные и разнообразные для вашего запроса:
# query = "Интересные факты о маркетинге социальных сетей"

##########################################################################################################################

# # Используем метод max_marginal_relevance_search для поиска документов
# selected_documents = db_md.max_marginal_relevance_search(
#     query=query,
#     k=3,             # хотим получить 3 докуменеа
#     fetch_k=20,      # рассматриваем 20 документов для MMR
#     lambda_mult=0.5, # балансируем между релевантностью и диверсификацией
# )
# # Перебираем список выбранных документов
# for i, doc in enumerate(selected_documents):
#     # Выводим информацию о каждом документе.
#     print(f"Document {i + 1}:")
#     print(f"Content: {format_text(doc.page_content)}")
#     print(f"Metadata: {doc.metadata}")
#     print("-"*30 + "\n")

##########################################################################################################################

# # similarity_search_with_relevance_scores
# query = "Интересные факты о маркетинге социальных сетей"

# k = 3  # мы хотим получить до 3 результатов
# score_threshold = 0.75  # мы хотим видеть только документы, релевантность которых не менее 0.75

# results = db_md.similarity_search_with_relevance_scores(query, k,
#                                                         score_threshold=score_threshold)

# # Выведем результаты
# for doc, similarity_score in results:
#     print(f"Document: {format_text(doc.page_content)} \
#             \nMetadata: {doc.metadata} \
#             \nScore_threshold: {similarity_score}")
#     print("-"*30 + "\n")

# Создаем объединенную векторную базу Postoplan
db_postoplan_merged = get_dbs_vector(path_db_index_md)
# Создаем объединенную векторную базу Simble
db_simble_merged = get_dbs_vector(path_db_index_text)

##########################################################################################################################

# # similarity_search_with_score
# # Используем векторную базу: db_postoplan_merged
# similar_documents = db_postoplan_merged.similarity_search_with_score("Интересные факты о маркетинге социальных сетей", k=3)

# message_content = re.sub(r'\n{2}', ' ', '\n '.join(
#     [f'\nОтрывок документа № {i+1}:\n' + doc.page_content + f'\nMetedata: {doc.metadata}' + f'\nScore: {score}'
#      for i, (doc, score) in enumerate(similar_documents)]))
# print(format_text(message_content))

##########################################################################################################################

# similarity_search_with_score
# Используем векторную базу: db_simble_merged

# similar_documents = db_simble_merged.similarity_search_with_score("КАСКО на короткий срок", k=3)

# message_content = re.sub(r'\n{2}', ' ', '\n '.join(
#     [f'\nОтрывок документа № {i+1}:\n' + doc.page_content[:1500] + f'\nScore: {score}'
#      for i, (doc, score) in enumerate(similar_documents)]))
# print(format_text(message_content))

# query = "Интересные факты о маркетинге социальных сетей"

# # Используем векторную базу: db_postoplan_merged
# answer = generate_answer(query, db_postoplan_merged, k=5, verbose=False)
# print('Ответ модели:\n')
# print(format_text(answer))

##########################################################################################################################

# # "similarity"
# # Используем векторную базу: db_postoplan_merged
# retriever = db_postoplan_merged.as_retriever(search_type="similarity",
#                                              search_kwargs={"k": 2})

# similar_documents = retriever.invoke("Интересные факты о маркетинге социальных сетей")

# message_content = re.sub(r'\n{2}', ' ', '\n '.join(
#     [f'\nОтрывок документа № {i+1}:\n' + doc.page_content + f'\nMetedata: {doc.metadata}'
#      for i, doc in enumerate(similar_documents)]))
# print(format_text(message_content))

##########################################################################################################################

# # "mmr"
# # Используем векторную базу: db_postoplan_merged
# retriever = db_postoplan_merged.as_retriever(
#                                     search_type="mmr",
#                                     search_kwargs={'k': 3,
#                                                    'lambda_mult': 0.5})

# similar_documents = retriever.invoke("Интересные факты о маркетинге социальных сетей")

# message_content = re.sub(r'\n{2}', ' ', '\n '.join(
#     [f'\nОтрывок документа № {i+1}:\n' + doc.page_content + f'\nMetedata: {doc.metadata}'
#      for i, doc in enumerate(similar_documents)]))
# print(format_text(message_content))

##########################################################################################################################

# # Инициализация языковой модели OpenAI
# llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)

# # Создание объекта `retriever` для поиска по базе данных `db_postoplan_merged`
# # с использованием метода поиска по схожести (similarity) и выборкой из 3 результатов
# retriever = db_postoplan_merged.as_retriever(search_type="similarity",
#                                              search_kwargs={"k": 3})

# # Определение системного промпта для цепочки обработки
# # Включает в себя обязательный шаблон с переменной `{context}`
# system_prompt = """
# Используй отрезки текста для ответа на вопрос.
# Context: {context}"""  # Context: {context} - обязателен

# # Создание шаблона промпта
# prompt = ChatPromptTemplate.from_messages(
#                         [("system", system_prompt),
#                          ("human", "{input}")]
#                         )

# # Создание цепочки для обработки запросов (retrieval chain)
# chain = create_retrieval_chain(
#         retriever,  # Ретривер для извлечения релевантного контекста из векторной базы данных
#         create_stuff_documents_chain(llm, prompt))  # Логика генерации ответа на основе модели `llm` и шаблона `prompt`

# query = "Интересные факты о маркетинге социальных сетей"
# ans = chain.invoke({"input": query})

# print('Ответ модели:\n')
# print(format_text(ans['answer']), '\n')

##########################################################################################################################

# # Используем c векторной базой данных `db_postoplan_merged`
# new_chain = create_model_chain(db_postoplan_merged)

# query = "Интересные факты о маркетинге социальных сетей" 
# ans = new_chain.invoke({"input": query})

# print('Ответ модели:\n')
# print(format_text(ans['answer']), '\n')

##########################################################################################################################