import os
from langchain_text_splitters import CharacterTextSplitter
import tiktoken
import textwrap
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import re
import json
import pickle
from dotenv import load_dotenv
import shutil
import requests
import zipfile
from pathlib import Path

#@title Функии

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

# Функция для чтения файла
def load_document_text(file_path) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# Запись словаря на диск в '/content/' в бинарном режиме
def save_dict_data(dict_data):
    with open('./content/task14/dict_data.pkl', 'wb') as file:
        pickle.dump(dict_data, file)


# Загрузка словаря с диска из '/content/' для чтения в бинарном режиме
def load_dict_data():
    with open('./content/task14/dict_data.pkl', 'rb') as file:
        return pickle.load(file)


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

# Формируем чанки из текста (CharacterTextSplitter) и создаем векторную базу
def create_db_index(text: str,
                    chunk_size=2048,     # Ограничение к-ва символов в чанке
                    chunk_overlap=0,     # к-во символов перекрытия в чанке
                    path = None,         # Путь сохранения
                    index_name='index'): # Название БД    
    splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     separator="\n") # по переносу строки
    text_chunks = splitter.split_text(text) # список текстовых чанков
    print(f'Количество чанков: {len(text_chunks)}.\n')
    db_index = FAISS.from_texts(text_chunks, OpenAIEmbeddings())
    if (path):
        db_index.save_local(path, index_name)
    return db_index


"""Функция answer_index выполняет поиск по векторной базе для получения k
наиболее релевантных документов по заданной теме, формирует
сообщения для модели и генерирует ответ с использованием OpenAI API """
def answer_index(system,        # инструкция system
                 instructions,  # инструкция для формирования роли user
                 topic_phrase,  # контент для поиска чанков в векторной базе
                 db_index,      # индексная база
                 k,             # количество релевантных чанков
                 example='',    # пример ответа
                 format='text',       # формат примера ответа 'json' или 'text'
                 model='gpt-4o-mini', # модель GPT
                 temp=0.1):     # температура
    docs = db_index.similarity_search_with_score(topic_phrase, k=k)
    response_format = None
    message_content = '\n '.join([f'Отрывок №{i+1}\n{doc[0].page_content}' for i, doc in enumerate(docs)])
    messages = [{"role": "system", "content": system}]
    if example != '':
        messages.append({"role": "user", "content": 'Ответь на вопрос' + ' и верни ответ в формате JSON' if format == 'json' else ''})
        messages.append({"role": "assistant", "content": example})
        if format == 'json': response_format = {'type': 'json_object'}
    messages.append({"role": "user", "content": f"{instructions}\n\nТексты для анализа:\n{message_content}"})
    completion = OpenAI().chat.completions.create(model=model,
                                                  messages=messages,
                                                  temperature=temp,
                                                  response_format=response_format)
    return completion.choices[0].message.content


"""Функция формирует сообщения для модели на основе документа, инструкций и результатов
анализа, затем генерирует ответ с использованием OpenAI API """
def answer_user_question_from_answer(system,                 # инструкция system
                                     instructions,           # инструкция для формирования роли user
                                     answers_content,        # результаты предыдущего анализа
                                     temp=0.1,               # температура
                                     model='gpt-4o-mini'):   # модель GPT
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"{instructions}\n\nИнформация для анализа:\n{answers_content}"}]
    completion = OpenAI().chat.completions.create(model=model, messages=messages, temperature=temp)
    return completion.choices[0].message.content

# Функция для загрузки содержимого по URL
def download_from_url(full_url, temp_dir):
    # Создание временной директории, если она не существует
    os.makedirs(temp_dir, exist_ok=True)
    
    # Создаём папку content если её нет
    content_dir = './content/task14'
    os.makedirs(content_dir, exist_ok=True)
    
    # Путь для Audio Record
    audio_record_path = './content/task14/Audio Record/'
    
    try:
        # Удаление существующих файлов и папок
        if os.path.exists("temp.zip"):
            os.remove("temp.zip")
        
        if os.path.exists(audio_record_path):
            shutil.rmtree(audio_record_path)
        
        # Загрузка файла по URL
        print(f"Загрузка из: {full_url}")
        response = requests.get(full_url, stream=True)
        response.raise_for_status()  # Проверка на ошибки HTTP
        
        # Сохраняем ZIP файл
        with open("temp.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Распаковываем ZIP
        with zipfile.ZipFile("temp.zip", 'r') as zip_ref:
            zip_ref.extractall(content_dir)
        
        print('Файлы успешно загружены!')
        
    except requests.RequestException as e:
        print(f"Ошибка загрузки: {e}")
    except zipfile.BadZipFile as e:
        print(f"Ошибка распаковки: {e}")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")

# Функция выбора векторной базы по зазванию
def get_db_index(name):
    db_index_path = './content/task14/faiss/'
    if name == 'Диалог': return load_db_vector(db_index_path, 'dialog_db_index')
    elif name == 'Client': return load_db_vector(db_index_path, 'client_db_index') 
    elif name == 'Manager': return load_db_vector(db_index_path, 'manager_db_index')

# Загрузка векторной базы с диска
def load_db_vector(folder_path_db_index,  # путь к сохраненной векторной базе
                   index_name):           # имя сохраненной векторной базы
    try:
        return FAISS.load_local(
                    allow_dangerous_deserialization=True, # Разрешает потенциально небезопасную десериализацию
                    embeddings=OpenAIEmbeddings(),  # Указывает векторные представления
                    folder_path=folder_path_db_index, # путь к сохраненной векторной базе
                    index_name=index_name) # имя сохраненной векторной базы
    except:
        return None

def get_db_vektor(folder_path_db_index, index_name, text): 
    db_vector = load_db_vector(folder_path_db_index, index_name)  
    if (not db_vector):
        db_vector = create_db_index(text, 2048, 0, folder_path_db_index, index_name)
    return db_vector

###########################################################################################################################################

load_dotenv()

###########################################################################################################################################

#@title Ссылка с Яндекс диска c текстовыми файлами диалога

folder_url_yadisk = "https://disk.yandex.ru/d/IqOQeMnvd6YSKg" # Ссылка на основную папку на Яндекс Диске
full_url = f'https://getfile.dokpub.com/yandex/get/{folder_url_yadisk}' # Построение полной ссылки для загрузки
temp_dir = 'temp' # Название временной директории

# Загрузка файлов
download_from_url(full_url, temp_dir)

###########################################################################################################################################

#@title Подсчет токенов. Создание векторных баз

text_dialog = load_document_text(r'./content/task14/Audio Record/merged_dialogue.txt')
dialog_db_index = get_db_vektor('./content/task14/faiss', 'dialog_db_index', text_dialog)

text_client = load_document_text(r'./content/task14/Audio Record/client_recogn.txt')
client_db_index = get_db_vektor('./content/task14/faiss', 'client_db_index', text_client)

text_manager = load_document_text(r'./content/task14/Audio Record/manager_recogn.txt')
manager_db_index = get_db_vektor('./content/task14/faiss', 'manager_db_index', text_manager)

###########################################################################################################################################

#@title 1\. Возражения клиента

objections = "\n".join([
    "возражение о том, что оучение слишком долгое",
    "возражение о том, что клиент не сможет справиться с темпом обучения и выделять достаточно времени"
])

system_prompt = """Ты лучше всех разбираешься в анализе общения менеджера по продажам и клиента.
Ты умеешь находить в текстах явные возражения клиента. Ты всегда точно следуешь порядку отчета."""

instructions = f"""Найди явные возражения, относящиеся только к возражениям вида {objections} в текстах 
и сделай об этом отчет в следующем формате:
напиши список явных возражений в текстах пунктами:
обобщенное название возражения, которое есть в текстах,
короткая цитата возражения из текстов,
время когда сказано возражение(23:19).
Примечание: повторяющиеся возражения исключи из отчета.
Примечание: фразы не являющиеся возражениями исключи из отчета.
Примечание: возражения, не относящиеся к возражениям вида {objections} исключи их отчета"""

topicphrase = ''

out = {}

db_index = get_db_index('Client')
num_fragment = 8
out['objections'] = answer_index(system_prompt, instructions, topicphrase, db_index, num_fragment)

print(format_text(out['objections']))
save_dict_data(out)

# ###########################################################################################################################################

#@title 2\. Негативные отзывы. Было ли возражение о том, что оучение слишком долгое

system_prompt = """Ты самый лучший сотрудник отдела контроля качества общения менеджера по продажам
и клиента. Менеджер работает в компании, которая продает обучение программированию на python и нейронным
сетям. Ты всегда очень точно следуешь порядку отчета. Ты знаешь, что возражение - это то, что не
устраивает клиента. Ты знаешь, что отработка возражения - это аргументация, которая нивелирует возражение."""

instructions = """Проанализируй: Было ли возражение о том, что обучение слишком долгое, в текстах? 
Если было, то как менеджер отработал это возражение? Напиши только то, что написано в порядке отчетов:
Первый отчет: напиши список, были ли названы возражения о том, что обучение слишком долгое, 
в текстах по всем текстам: "да / нет", "короткая цитата отработки возражения менеджером, 
если отработка возражения была", "время отрывка в котором была взята цитата (например 23:19)".
Второй отчет: "quality" - только одна общая оценка качества отработки возражения или 
nan (100% - убедительно отработано возражение, 0% - возражение было и его не отработали, 
"nan" - не было возражения и оно не было отработано).
Третий отчет: напиши, почему ты поставил такую оценку во втором отчете.. """

topicphrase = """Обучение слешком долгое"""

answer_example = """Строго следуй формату вывода:
'reports': [{'num_report':  int, 'name_report': str, 'text_report': str},...]"""
answer_format = "json"

db_index = get_db_index('Диалог')
num_fragment = 5
result = answer_index(system_prompt, instructions, topicphrase, db_index, num_fragment,
                      example=answer_example, format=answer_format)

out = {}

# if (not 'out' in globals()):
#     out = load_dict_data()

if result:
    out['objections_too_long'] = json.loads(result)
    out['objections_too_long_total'] = out['objections_too_long']['reports'][2]['text_report']

    # print(bool(out['objections_too_long']['reports'][1]['text_report']))
    print(format_text(out['objections_too_long_total']))
    save_dict_data(out)

# ###########################################################################################################################################

#@title 3\. Негативные отзывы. Было ли возражение о том, что клиент не сможет справиться с темпом обучения и выделять достаточно времени

system_prompt = """Ты самый лучший сотрудник отдела контроля качества общения менеджера по продажам
и клиента. Менеджер работает в компании, которая продает обучение программированию на python и нейронным
сетям. Ты всегда очень точно следуешь порядку отчета. Ты знаешь, что возражение - это то, что не
устраивает клиента. Ты знаешь, что отработка возражения - это аргументация, которая нивелирует возражение."""

instructions = """Проанализируй: Было ли возражение о том, что клиент не сможет справиться с темпом обучения 
и выделять достаточно времени, в текстах? 
Если было, то как менеджер отработал это возражение? Напиши только то, что написано в порядке отчетов:
Первый отчет: напиши список, были ли названы возражения о том, что клиент не сможет справиться с темпом обучения 
и выделять достаточно времени, в текстах по всем текстам: 
"да / нет", "короткая цитата отработки возражения менеджером, если отработка возражения была", 
"время отрывка в котором была взята цитата (например 23:19)".
Второй отчет: "quality" - только одна общая оценка качества отработки возражения 
или nan (100% - убедительно отработано возражение, 0% - возражение было и его не отработали, 
"nan" - не было возражения и оно не было отработано).
Третий отчет: напиши, почему ты поставил такую оценку во втором отчете... """

topicphrase = """Слишком высокий темп обучения, нет времени, много других дел"""

answer_example = """Строго следуй формату вывода:
'reports': [{'num_report':  int, 'name_report': str, 'text_report': str},...]"""
answer_format = "json"

db_index = get_db_index('Диалог')
num_fragment = 5
result = answer_index(system_prompt, instructions, topicphrase, db_index, num_fragment,
                      example=answer_example, format=answer_format)

if (not 'out' in globals()):
    out = load_dict_data()

if result:
    out['objections_no_time'] = json.loads(result)
    out['objections_no_time_total'] = out['objections_no_time']['reports'][2]['text_report']

    print(result)
    print(format_text(out['objections_no_time_total']))
    save_dict_data(out)

# ###########################################################################################################################################

#@title Отчет по возражениям: насколько хорошо менеджер отрабатывал возражения клиента

system_prompt = """Ты великолепно умеешь анализировать и делать выводы. Ты отлично умеешь выделять
значимые вещи в текстах для отчета руководителю отдела продаж. Твои отчеты всегда кратки и по существу.
Ты всегда очень точно следуешь порядку отчета."""

instructions = """Напиши 4 отчета насколько хорошо менеджер отрабатывал возражения клиента, строго
как указано в этих порядках отчетов: "результат работы менеджера по отработке возражений  клиента":
1 отчет: проанализируй сразу все полученные результаты и напиши кратко общий вывод, насколько хорошо
менеджер отрабатывал возражения клиента, а также  выдели ключевые моменты из результата анализа.
2 отчет:  на основании первого отчета напиши краткий список конкретных рекомендаций менеджеру для
улучшения работы с возражениями клиента.
3 отчет: Сделай разбор по эффективности менеджера по отработке возражений  клиента, судя по результатам
анализа, и напиши эти выводы.
4 отчет: "quality" - путем вычислений напиши число среднего процента качества по всем результатам
анализа от 0% до 100%. Если в информации для анализа говориться, что возражения не было озвучено в текстах 
и его оценка = "nan" - то не прибавляй его к общему количеству
Примечание: обязательно напиши только 4 отчета. """

data = {
    'objections': 'Список возражений клиента: ',
    'objections_too_long_total': 'Как менеджер обработал возражение о том, что обучение слишком долгое:',
    'objections_no_time_total': 'Как менеджер обработал возражение о том, что у клиента нет времени на обучение:',
}

if (not 'out' in globals()):
    out = load_dict_data()

answers = []

for i, key in enumerate(data.keys(), start=1):
    main_text = data[key]
    extra_text = out.get(key, '').strip()
    answer = f"Анализ №{i}. {main_text} {extra_text}"
    answers.append(answer)

out['objections_report'] = answer_user_question_from_answer(system_prompt, instructions, answers)

print(format_text(out['objections_report']))
save_dict_data(out)

# ###########################################################################################################################################

# @Результат 

"""
Загрузка из: https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/IqOQeMnvd6YSKg
Файлы успешно загружены!
### Отчет о явных возражениях клиента

1. **Возражение о том, что обучение слишком долгое**
   - **Цитата:** "Ну, это ерунда. И, ну, безусловно, кроме того, что это ерунда, плюс к всему нет гарантии, что я этому
обучусь и потом у меня уже будет работа. Ну, в общем, в пустую потрачу свое время."
   - **Время:** 6:15

2. **Возражение о том, что клиент не сможет справиться с темпом обучения и выделять достаточно времени**
   - **Цитата:** "Я пока не смогла найти, в чем бы я хотела реализоваться, и что мне будет комфортно. Ну, то есть я пока
не вижу такой сферы деятельности и то, чем можно обучиться с нуля."
   - **Время:** 6:06

### Примечание
Повторяющиеся возражения были исключены из отчета, а также фразы, не являющиеся возражениями.
Возражение о длительности обучения не было озвучено в текстах.
{
  "reports": [
    {
      "num_report": 1,
      "name_report": "Возражение о темпе обучения",
      "text_report": {
        "возражение": "да",
        "цитата": "у нас еще такая программа обучения, где вы можете двигаться в собственном темпе и не зависеть от потока, там, от того, как идет.",
        "время": "18:39"
      }
    },
    {
      "num_report": 2,
      "name_report": "Оценка качества отработки возражения",
      "text_report": 100
    },
    {
      "num_report": 3,
      "name_report": "Обоснование оценки",
      "text_report": "Менеджер убедительно отработал возражение, предоставив информацию о возможности обучения в собственном темпе, что позволяет клиенту не беспокоиться о недостатке времени и темпе обучения."
    }
  ]
}
Менеджер убедительно отработал возражение, предоставив информацию о возможности обучения в собственном темпе, что
позволяет клиенту не беспокоиться о недостатке времени и темпе обучения.
### Отчет 1: Общий вывод по отработке возражений клиента

На основании анализа представленных данных, можно сделать вывод, что менеджер в целом хорошо отработал возражения
клиента, однако не все возражения были озвучены. Ключевые моменты анализа:

- Возражение о длительности обучения не было озвучено, что затрудняет оценку работы менеджера по этому пункту.
- Менеджер успешно обработал возражение о нехватке времени на обучение, предложив гибкий подход к обучению в собственном
темпе.

### Отчет 2: Рекомендации для менеджера

1. Убедиться, что все возможные возражения клиентов заранее проработаны и подготовлены ответы на них.
2. Активно задавать вопросы клиентам, чтобы выявить возможные возражения, которые могут не быть озвучены.
3. Разработать дополнительные материалы, которые помогут клиентам лучше понять процесс обучения и его преимущества.
4. Продолжать использовать подходы, которые позволяют клиентам чувствовать себя комфортно, например, обучение в
собственном темпе.

### Отчет 3: Эффективность менеджера по отработке возражений

Менеджер продемонстрировал высокую эффективность в отработке возражений, особенно в случае с нехваткой времени на
обучение. Однако отсутствие информации о других возражениях, таких как длительность обучения, указывает на необходимость
улучшения в выявлении и проработке всех возможных возражений. Важно, чтобы менеджер не только реагировал на озвученные
возражения, но и активно искал их, чтобы повысить общую эффективность взаимодействия с клиентами.

### Отчет 4: Средний процент качества

В данном анализе было рассмотрено 2 возражения, из которых одно было успешно обработано, а другое не было озвучено.
Таким образом, расчет среднего процента качества:

- Обработанное возражение: 100%
- Неозвученное возражение: nan (не учитывается)

Общее количество возражений для анализа: 1 (только озвученное).
Средний процент качества = (100% / 1) = 100%.

**Итоговый процент качества: 100%**.
"""