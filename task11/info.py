#@title Импортируем библиотеки

import yt_dlp
import whisper
from pydub import AudioSegment
import textwrap
import matplotlib.pyplot as plt
from IPython.display import Audio
import os
import re
import time
import tiktoken
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
import os
import shutil
from pprint import pprint as pp
from dotenv import load_dotenv

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

# Извлекаем Аудио-дорожку из Видео контента (Youtube, VK, RuTube)
def get_audio_from_video(url: str,
                         folder: str,
                         audio_file_name: str):
    path_file = f'{folder}{audio_file_name}'
    ydl_opts = {'format': 'm4a/bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio',
                                    'preferredcodec': 'm4a'}],
                'outtmpl': path_file}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(url)
            if error_code == 0:
                print('Загрузка файла прошла успешно!')
            else:
                print(f'Код ошибки: {error_code}')
        print(f"\nАудиофайл загружен: {path_file}.m4a")
        return f'{path_file}.m4a'
    except:
        return None
    
# Информация об аудио файлe
def audio_info(audio_file):
    audio = AudioSegment.from_file(audio_file)
    print(f'\nПродолжительность: {audio.duration_seconds / 60} min.')
    print(f'Частота дискретизаци: {audio.frame_rate}')
    print(f'Количество каналов: {audio.channels}')
    return audio

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
    num_tokens = len(encoding.encode(string)) + 10
    # Возвращаем количество токенов
    return num_tokens


# Построение гистограммы распределения количества токенов по чанкам (если нужно)
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


# (CharacterTextSplitter) Формируем чанки из текста по количеству символов
def split_text(text: str,
               chunk_size=30000,    # Ограничение к-ва символов (не токенов) в чанке
               chunk_overlap=1000): # к-во символов перекрытия в чанке
    # Удалить пустые строки и лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Создаем экземпляр CharacterTextSplitter с заданными парамаетрами
    splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     separator=" ")
    return splitter.split_text(text) # список текстовых чанков


# (MarkdownHeaderTextSplitter)
# Формируем чанки в формат LangChain Document из текста с Markdown разметкой
def split_markdown_text(markdown_text,
                        strip_headers=False): # НЕ удалять заголовки под '#..' из page_content
    # strip_headers=False  - Заголовки будут сохраняться в page_content и в metadata
    # Определяем заголовки, по которым будем разбивать текст
    headers_to_split_on = [("#", "Header 1"),   # Заголовок первого уровня
                           ("##", "Header 2"),  # Заголовок второго уровня
                           ("###", "Header 3")] # Заголовок третьего уровня
    # Создаем экземпляр MarkdownHeaderTextSplitter с заданными заголовками
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on,
                                                   strip_headers=strip_headers)
    # Разбиваем текст на чанки в формат LangChain Document
    return markdown_splitter.split_text(markdown_text)


# Функция получения ответа от модели
def generate_answer(system, user, text, temp=0.3, model='gpt-4o-mini'):
    _ = load_openai_config()
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user + '\n' + text}]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp)
    return completion.choices[0].message.content


# Обработка текстовых чанков по очереди (если нужно)
def process_text_chunks(text_chunks, system, user):
    processed_text = ''
    for chunk in text_chunks:
        # получение ответа от модели для каждого чанка
        answer = generate_answer(system, user, chunk)
        processed_text += f'{answer}\n\n'  # Добавляем ответ в результат
    return processed_text

# Транскрибация аудио в текст (OpenAI - whisper)
def transcribe_audio_whisper(audio_path,
                             file_title,
                             save_folder_path,
                             max_duration=10*60*1000):  # 10 минут
    """
    Функция для транскрибации аудиофайла по частям, чтобы соответствовать ограничениям размера API.
    """
    # Создание папки для сохранения результатов, если она ещё не существует
    os.makedirs(save_folder_path, exist_ok=True)
    # Загрузка аудиофайла
    audio = AudioSegment.from_file(audio_path)
    # Создание временной папки в колабе для хранения аудио фрагментов
    temp_dir = os.path.join('./content/', "temp_audio_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    # Инициализация переменных для обработки аудио фрагментов
    current_start_time = 0  # Текущее время начала фрагмента
    chunk_index = 1         # Индекс текущего фрагмента
    transcriptions = []     # Список для хранения всех транскрибаций

    # Обработка аудиофайла частями
    while current_start_time < len(audio):
        # Выделение фрагмента из аудиофайла
        chunk = audio[current_start_time:current_start_time + max_duration]
        # Формирование имени и пути файла фрагмента
        chunk_name = f"chunk_{chunk_index}.mp3"
        chunk_path = os.path.join(temp_dir, chunk_name)
        # Экспорт фрагмента
        chunk.export(chunk_path, format="mp3")

        # Проверка размера файла фрагмента на соответствие лимиту API
        if os.path.getsize(chunk_path) > 26000000:  # почти 25 MB
            print(f"Chunk {chunk_index} exceeds the maximum size limit for the API. Trying a smaller duration...")
            max_duration = int(max_duration * 0.8)  # Уменьшение длительности фрагмента
            os.remove(chunk_path)  # Удаление фрагмента, превышающего лимит
            continue

        # Открытие файла фрагмента для чтения в двоичном режиме
        with open(chunk_path, "rb") as src_file:
            print(f"Transcribing {chunk_name}...")
            try:
                # Запрос на транскрибацию фрагмента с использованием модели Whisper
                transcript_response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=src_file)
                # Добавление результата транскрибации в список транскрипций
                transcriptions.append(transcript_response.text)
            except openai.BadRequestError as e:
                print(f"An error occurred: {e}")
                break
        # Переход к следующему фрагменту
        current_start_time += max_duration
        chunk_index += 1

    # Сохранение всех транскрибаций в один текстовый файл
    result_text = "\n".join(transcriptions)
    result_path = os.path.join(save_folder_path, f"{file_title}.txt")
    with open(result_path, "w") as txt_file:
        txt_file.write(result_text)
    print(f"Transcription saved to {result_path}")

    # Удаляем временную папку и все файлы в ней
    shutil.rmtree(temp_dir)
    return result_text

# Обработка каждого чанка (документа) для формирования методички
def process_documents(path_drive, documents, system, user):
    """
    Функция принимает папку для сохранения файла, чанки, system, user
    Она обрабатывает каждый документ, конкатенирует результаты в один текст и сохраняет в файл .txt.
    В итоге мы получаем методичку по лекции.
    """
    processed_text_for_handbook = ""  # Строка для конкатенации обработанного текста
    for document in documents:
        # Получаем ответ от модели для каждого документа
        answer = generate_answer(system, user, document.page_content)
        # Добавляем обработанный текст в общую строку
        processed_text_for_handbook += f"{answer}\n\n"
    # Записываем полученный текст в файл
    with open(os.path.join(path_drive, 'short_tutorial.txt'), 'w', encoding='utf-8') as f:
        f.write(processed_text_for_handbook)
    return processed_text_for_handbook

#@title Функции и создание векторной базы

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
    return db_index


# Загрузка векторной базы с диска
def load_db_vector(folder_path_db_index,  # путь к сохраненной векторной базе
                   index_name):           # имя сохраненной векторной базы
    return FAISS.load_local(
                allow_dangerous_deserialization=True, # Разрешает потенциально небезопасную десериализацию
                embeddings=OpenAIEmbeddings(),  # Указывает векторные представления
                folder_path=folder_path_db_index, # путь к сохраненной векторной базе
                index_name=index_name) # имя сохраненной векторной базы


# Функция запроса и ответа от OpenAI с поиском по векторной базе данных
def generate_db_answer(query,    # запрос пользователя
                       db_index, # векторная база знаний
                       k=3,      # используемое к-во чанков
                       verbose=True, # выводить ли на экран выбранные чанки
                       model='gpt-4o-mini',
                       temp=0.3):
    # Поиск чанков по векторной базе данных
    similar_documents = db_index.similarity_search(query, k=k)
    # Формирование текстового контента из выбранных чанков для модели
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'Отрывок документа № {i+1}:\n' + doc.page_content
        for i, doc in enumerate(similar_documents)]))
    if verbose:
        print(message_content) # печатать на экран выбранные чанки

    messages = [{"role": "system",
                 "content": f'Ответь подробно на вопрос пользователя на основании информации из базы знаний: \n{message_content}'},
                {"role": "user",
                 "content": f'Вопрос пользователя: {query}'}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp)
    return response.choices[0].message.content

_ = load_openai_config()

# Пример: Получаем Аудио из VK видео
url= 'https://vkvideo.ru/video-123851409_456239255'
folder = './content/'
audio_file_name = 'Квантовая физика - от головоломки до технологий'

# _ = get_audio_from_video(url, folder, audio_file_name)

# # Транскрибация с сохранением текстового файла на диск

# # Аудио файл с гугл диска
# audio_path = os.path.join(folder, "Квантовая физика - от головоломки до технологий.m4a")
# file_title = 'Квантовая физика - от головоломки до технологий'
# start_time = time.time()

# res_text = transcribe_audio_whisper(audio_path, file_title, folder)

# print(f'\nВремя выпонения: {time.time() - start_time} секунд.\n')
# # Фрагмент текста
# pp(res_text[:1000])

# audio_path = os.path.join(folder, "Квантовая физика - от головоломки до технологий.m4a")
# _ = audio_info(audio_path)

# @title Загрузка текста и определения размера
# Текстовый файл с гугл диска
text_file = os.path.join(folder, "Квантовая физика - от головоломки до технологий.txt")
with open(text_file, 'r', encoding='utf-8') as txt_file:
    text = txt_file.read()

# # Количество токенов во всем тексте
# print(f'Количество токенов во всем тексте: {num_tokens_from_string(text)}\n')

# @title Обработка текста

# В зависимости от размера контекстного окна (в токенах) для разных моделей мы можем подать разный объем текста
# Если текст небольшой, мы можем весь текст отправить в модель, без деления на чанки
# И получим ответ примерно на такое же количество токенов (по условиям промпта - сохранение 100% текста)

# system = """Вы гений текста, копирайтинга, писательства. Ваша задача распознать разделы в тексте
# и разбить его на эти разделы сохраняя весь текст на 100%"""

# user = """Пожалуйста, давайте подумаем шаг за шагом: Подумайте, какие разделы в тексте вы можете
# распознать и какое название по смыслу можно дать каждому разделу. Далее напишите ответ по всему
# предыдущему ответу и оформи в порядке:
# ## Название раздела, после чего весь текст, относящийся к этому разделу. Текст:"""

# # Подали весь текст в модель без деления на чанки
# processed_md_text = generate_answer(system, user, text)
# print(processed_md_text[100:])
# # Записываем processed_md_text в файл на гугл диск
# with open(os.path.join(folder, 'processed_md_text.txt'), "w") as txt_file:
#     txt_file.write(processed_md_text)

# Если размер всего текста превышает размер контекстного окна модели в токенах, разделим его на чанки

# # Разбиваем текст на чанки
# text_chunks = split_text(text, chunk_size=30000, chunk_overlap=1000)
# # Гистограмма распределения количества токенов по чанкам
# create_histogram(text_chunks)
# # Обработка текстовых чанков по очереди
# processed_md_text = process_text_chunks(text_chunks, system, user)
# # Записываем processed_md_text в файл на гугл диск
# with open(os.path.join(folder, 'processed_md_text.txt'), "w") as txt_file:
#     txt_file.write(processed_md_text)

# # Текстовый файл processed_md_text.txt с гугл диска
# text_file = os.path.join(folder, "processed_md_text.txt")
# with open(text_file, 'r', encoding='utf-8') as txt_file:
#     text_md = txt_file.read()

# # # Получаем список документов, разбитых по заголовкам
# chunks_md_splits = split_markdown_text(text_md)

# # print('Заголовки: \n')
# for chunk in chunks_md_splits:
#     print(chunk.metadata['Header 2'])

#@title Формируем методичку

# system = """Ты гений копирайтинга. Ты получаешь раздел необработанного текста по определенной теме.
# Нужно из этого текста выделить самую суть, только самое важное, сохранив все нужные подробности и детали,
# но убрав всю "воду" и слова (предложения), не несущие смысловой нагрузки."""

# user = """Из данного текста выдели только ключевую и ценную с точки зрения темы раздела информацию.
# Удали всю "воду". В итоге у тебя должен получится раздел для методички по указанной теме. Опирайся
# только на данный тебе текст, не придумывай ничего от себя. Ответ нужен в формате:
# ## Название раздела, и далее выделенная тобой ценная информация из текста."""

# # Создание методички и запись текста методички на Гугл Диск
# short_tutorial = process_documents(folder, chunks_md_splits, system, user)
# print(f"Обработанный текст сохранен в файле: {os.path.join(folder, 'short_tutorial.txt')}")

# # Вывод содержимого методички:
# with open(os.path.join(folder, 'short_tutorial.txt'), 'r', encoding='utf-8') as f:
#     processed_text = f.read()

# print(format_text(processed_text))

# # Текстовый файл processed_md_text.txt с гугл диска
# text_file = os.path.join(folder, "short_tutorial.txt")
# with open(text_file, 'r', encoding='utf-8') as txt_file:
#     text_md = txt_file.read()

# # # Получаем список документов, разбитых по заголовкам
# chunks_md_splits = split_markdown_text(text_md)

# # Создание и сохранение векторной базы на ГуглДиск
# db_vector = create_db_index_from_documents_save(chunks_md_splits, 'db_vector', folder)

# # Загрузка векторной базы с ГуглДиска
# db_vector = load_db_vector(folder, 'db_vector')

# system = """Ты - преподаватель, эксперт по Квантовой физике. Твоя задача - ответить
# на вопрос только на основе представленных тебе документов, не добавляя ничего от себя."""

# query = "Что такое квантовый компьютер?"

# answer = format_text(generate_db_answer(query, db_vector))
# print(f'\nОтвет косультанта: \n\n{answer}')