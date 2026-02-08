from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


from io import BytesIO
import aiohttp
from PyPDF2 import PdfReader
from datetime import datetime
import requests
from urllib.parse import urlencode
import textwrap
import openai
import json
import time
import os
import re
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

# Функция для форматирования текста по абзацам
def format_text(text, width=120):
    # Разделяем текст на абзацы
    paragraphs = str(text).split('\n')
    # Форматируем каждый абзац отдельно
    formatted_paragraphs = []
    for paragraph in paragraphs:
        # Используем textwrap.fill для форматирования абзаца, чтобы длина строки не превышала width
        formatted_paragraph = textwrap.fill(paragraph, width)
        formatted_paragraphs.append(formatted_paragraph)
    # Объединяем абзацы с символом новой строки
    return '\n'.join(formatted_paragraphs)

# Функция записи логов в файл _log.txt
def add_log_file(text, title=''):
    time_now = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    with open(os.path.join(path, '_log.txt'), "a", encoding='utf-8') as file:
            file.write(f'\n\n{time_now}. {title}.\n\n{format_text(text)}')

# Чтение PDF файла
def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    # возвращаем текст построчно
    return ' '.join([line.extract_text() for line in reader.pages])

# Загрузка фала из YandexDisk
def download_from_yandex_disk(link, output_path):
    # link: ссылка на файл в Yandex Диске
    # output_path: путь, куда будет сохранен загруженный файл
    # Базовый URL для получения информации о ресурсе
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources?'
    # Формируем финальный URL для запроса информации о ресурсе
    final_url = base_url + urlencode(dict(public_key=link))
    # Делаем запрос для получения информации о ресурсе
    response = requests.get(final_url)
    resource_info = response.json()
    # Получаем ссылку на скачивание файла
    download_url = resource_info['file']
    # Получаем имя файла
    file_name = resource_info['name']
    # Скачиваем файл
    download_response = requests.get(download_url)
    with open(os.path.join(output_path, file_name), 'wb') as f:
        f.write(download_response.content)
    add_log_file(link, title='yandex_link') # Запись в лог-файл

# Формируем поля для парсига Вакансии для последующего использовании при поиске
# https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/
"""Класс Vacancy с использованием библиотеки Pydantic, который будет использоваться для парсинга
   вакансий. Класс имеет два поля: position, requirements. Описание
   для каждого поля включает инструкцию по извлечению соответствующих данных из текста вакансии."""

add_prompt = "Иначе, выведи ответ: None"
class Vacancy(BaseModel):
    position: str = Field(
        description = f'Найди в тексте вакансии название должности (позиции). {add_prompt}')
    requirements: list = Field(
        description = re.sub(r'\s+', ' ', f"""Найди в тексте вакансии 10 основных ключевык требований к соискателю. 
                      {add_prompt}"""))
    
# Парсинг текста с формированием полей по параметрам (parser_class=Vacancy или Resume)
# JSON parser https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/
def to_dict_parser(text, parser_class, model='gpt-4o') -> dict:
    # text: текст для парсинга
    # parser_class: класс парсера (например, Vacancy или Resume)
    # model: имя модели для генерации текста (по умолчанию 'gpt-4o')
    
    # Экземпляр JsonOutputParser (из langchain)
    parser = JsonOutputParser(pydantic_object=parser_class)
    prompt = PromptTemplate(
        input_variables = ["query"],
        # шаблон с инструкциями
        template = "Follow the instructions:\n"
                   "{format_instructions}\n"
                   "{query}\n",
        # встроенные инструкции JsonOutputParser по форматированию
        partial_variables = {"format_instructions": parser.get_format_instructions()})

    # цепочка (chain) из шаблона, модели и парсера
    # оператор '|' означает последовательное выполнение: шаблон -> модель -> парсер
    model = ChatOpenAI(model=model, temperature=0)

    chain = prompt | model | parser
    return chain.invoke({"query": text})

# Функция генерации ответа от OpenAI
def generate_answer(prompt_system, prompt_user, prompt_assistant='', model='gpt-4o-mini', temp=0.1):
    messages = [
        {"role": "system", "content": prompt_system},
        {'role': 'assistant', 'content': prompt_assistant},
        {"role": "user", "content": prompt_user}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        # max_tokens=4000
    )
    # Запись в лог-файл
    add_log_file(f'\n{messages}\n{response.choices[0].message.content}\n\n\n', title='generate_answer()')
    return response.choices[0].message.content

###########################################################################################################################################

load_openai_config()

# Формируем основной путь
path = './content/data/task13/'
# Формирование дополнительных путей
path_vacancy_pdf = f'{path}vacancy_pdf/'
path_vacancy_json = f'{path}vacancy_json/'
path_resume_pdf = f'{path}resume_pdf/'

###########################################################################################################################################

#@title Создам необходимые директории

folders = [path, path_vacancy_pdf, path_vacancy_json, path_resume_pdf]
# Создание необходимых папок: '/data/tesk13/...'
for folder in folders:
    os.makedirs(folder, exist_ok=True)

###########################################################################################################################################

#@title Ссылки на файлы вакансию и резюме

# Файл резюме
resume_link = 'https://disk.yandex.ru/i/ULV_Onsdrx7zrw'
download_from_yandex_disk(resume_link, path_resume_pdf)

# Файл вакансии
vacancy_link = 'https://disk.yandex.ru/i/Lg7ZhWiutBVpXg'
download_from_yandex_disk(vacancy_link, path_vacancy_pdf)

###########################################################################################################################################

# #@title Читаем pdf файл вакансии и выделаем в массив 10 ключевых требований и сохраняем в json-файл
vacancy_text = read_pdf(os.path.join(path_vacancy_pdf, 'HR_Director.pdf'))

vacancy_requirements = to_dict_parser(vacancy_text, Vacancy)

# Сохрание всех данных словаря в Json файл
with open(os.path.join(path_vacancy_json, f'HR_Director.json'), 'w', encoding='utf-8') as f:
    json.dump(vacancy_requirements, f, ensure_ascii=False)

###########################################################################################################################################

with open(os.path.join(path_vacancy_json, 'HR_Director.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
    vacancy_requirements = "\n".join(data['requirements'])

resume_text = read_pdf(os.path.join(path_resume_pdf, 'Resume HR_Director+.pdf'))

prompt_system = """
Ты — опытный рекрутер/техлид и строгий оценщик соответствия кандидата вакансии.
Тебе даны: текст вакансии, текст резюме кандидата, список из 10 ключевых требований (критериев) к вакансии.
"""

prompt_user = f"""
Задача: оценить, насколько кандидат подходит по каждому из 10 требований, 
используя только информацию из резюме и текста вакансии. 
Не придумывай факты. Если данных нет — так и напиши.

Входные данные:
Текст вакансии: {vacancy_text}
Текст резюме кандидата: {resume_text}
Ключевые требования к кандидату: \n
{vacancy_requirements}
Правила оценки:
- Оценивай каждое требование отдельно.
- Для каждого требования выдай:
Оценка соответствия: число от 0 до 5, где
0 — не подтверждено/противоречит,
1 — косвенно/очень слабо,
2 — частично,
3 — в целом подходит, но есть пробелы,
4 — хорошо подтверждено,
5 — полностью и явно подтверждено сильными примерами.

В конце:
- Итоговый балл: сумма по всем требованиям (макс 50).
- Краткая рекомендация: “Приглашать”, “Приглашать с оговорками”, или “Не приглашать”.
- Топ-3 сильные стороны и топ-3 слабые стороны (по фактам из резюме).
- Если есть критические требования (must-have) в вакансии, укажи, какие из них не подтверждены и почему.

Формат ответа (строго соблюдай)
- Оценка
- Вердикт
- Обоснование
- Пробелы/риски
"""

result = generate_answer(prompt_system=prompt_system, prompt_user=prompt_user)
print(result)

###########################################################################################################################################

# #@title Результат

"""
### Оценка соответствия кандидата по требованиям

1. **Опыт работы руководителем отдела/HRD от 1 года**
   - Оценка: 5
   - Вердикт: Полностью и явно подтверждено сильными примерами.
   - Обоснование: Кандидат имеет более 12 лет опыта работы в HR, включая должности директора по персоналу и руководителя блока подбора и удержания персонала.
   - Пробелы/риски: Нет.

2. **Опыт управления удаленными командами**
   - Оценка: 4
   - Вердикт: Хорошо подтверждено.
   - Обоснование: Кандидат упоминает управление HR-процессами в регионах и работу с удаленными сотрудниками, однако конкретные примеры управления удаленными командами не указаны.
   - Пробелы/риски: Отсутствие конкретных примеров.

3. **Опыт управления в рекрутменте**
   - Оценка: 5
   - Вердикт: Полностью и явно подтверждено сильными примерами.
   - Обоснование: Кандидат имеет значительный опыт в подборе персонала, включая массовый подбор и закрытие ТОП-вакансий.
   - Пробелы/риски: Нет.

4. **Опыт работы в IT-компаниях**
   - Оценка: 4
   - Вердикт: Хорошо подтверждено.
   - Обоснование: Кандидат работал в компании "Билайн", которая относится к IT-сектору, но не указано, что это был основной фокус работы.
   - Пробелы/риски: Неясно, насколько глубоко кандидат погружен в специфику IT.

5. **Высокий уровень самодисциплины**
   - Оценка: 5
   - Вердикт: Полностью и явно подтверждено сильными примерами.
   - Обоснование: Кандидат демонстрирует способность достигать результатов и управлять большими командами, что требует высокой самодисциплины.
   - Пробелы/риски: Нет.

6. **Ответственность и продуктивность**
   - Оценка: 5
   - Вердикт: Полностью и явно подтверждено сильными примерами.
   - Обоснование: Кандидат описывает успешные результаты своей работы, такие как снижение текучести и увеличение численности сотрудников.
   - Пробелы/риски: Нет.

7. **Умение принимать решения и нести за них ответственность**
   - Оценка: 5
   - Вердикт: Полностью и явно подтверждено сильными примерами.
   - Обоснование: Кандидат описывает множество примеров, где принимались решения, влияющие на бизнес-процессы и результаты.
   - Пробелы/риски: Нет.

8. **Аналитическое мышление**
   - Оценка: 5
   - Вердикт: Полностью и явно подтверждено сильными примерами.
   - Обоснование: Кандидат внедрял HR-дашборды и проводил аудит HR-функции, что требует аналитических навыков.
   - Пробелы/риски: Нет.

9. **Автоматизация процессов управления персоналом**
   - Оценка: 5
   - Вердикт: Полностью и явно подтверждено сильными примерами.
   - Обоснование: Кандидат упоминает внедрение автоматизации процессов на базе Talantix и корпоративного портала.
   - Пробелы/риски: Нет.

10. **Разработка долгосрочной системы мотивации**
    - Оценка: 4
    - Вердикт: Хорошо подтверждено.
    - Обоснование: Кандидат разработал системы KPI и нематериальной мотивации, но не указал на наличие долгосрочной системы.
    - Пробелы/риски: Неясно, насколько долгосрочные системы были разработаны.

### Итоговый балл: 46 из 50

### Рекомендация: Приглашать

### Топ-3 сильные стороны:
1. Обширный опыт работы в HR и управления персоналом.
2. Успешный опыт в рекрутменте и автоматизации HR-процессов.
3. Высокий уровень ответственности и продуктивности, подтвержденный результатами.

### Топ-3 слабые стороны:
1. Неясность в опыте управления удаленными командами.
2. Отсутствие конкретных примеров работы в IT-секторе.
3. Недостаточная информация о разработке долгосрочной системы мотивации.

### Критические требования (must-have), которые не подтверждены:
- Опыт управления удаленными командами: конкретные примеры отсутствуют, что может быть важным для удаленной работы в компании.
"""