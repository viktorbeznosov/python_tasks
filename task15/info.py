from urllib.parse import urlencode
import hmac
import hashlib

from langchain_text_splitters  import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pprint import pprint as pp
from IPython.display import clear_output
from googleapiclient.discovery import build
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import requests
import textwrap
import openai
import gdown
import json
import time
import os
import re
from dotenv import load_dotenv

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

###########################################################################################################################################

# Формируем основной путь
path = './content/task15/drive/MyDrive/data/tech_support/'
os.makedirs(path, exist_ok=True)

# для обработки Гугл таблиц
google_service_file = os.path.join(path, 'proj-202406.json') # json файл для доступа к гугл таблицам
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_service_file

# Cоздание или очистка файла логов
open(os.path.join(path, '_log.txt'), 'w').close()
clear_output() # очистка вывода

# Словарь для учета количества токенов и стоимости
TOKENS = {'price': 0.,
          'input': 0.,
          'output': 0.,
          'total': 0.}

###########################################################################################################################################

#@title II. Блок базовых функций

# преобразование строки в Python код
def try_eval(func_text):
    try: return eval(func_text)
    except Exception as ex:
        print(f'Error eval(): {ex}')
        try_eval(func_text)


# Функция преобразования текста в список строк
def text_to_list_lines(text):
    # список по разделителю переноса строки и без пустых строк
    return [line for line in text.split("\n") if line.strip() != '']


# Функция для форматирования текста
def format_text(text, width=120):
    try:
        return '\n'.join(textwrap.fill(line, width) for line in str(text).split('\n'))
    except Exception as ex:
        return f'Error format_text(): {ex}'


# Загрузка фала из GoogleDisk по открытой ссылке
def download_from_google_disk(link, output_path):
    # Идентификатор файла - это последовательность из 25 или более символов (букв, цифр, подчеркиваний, дефисов)
    id = re.findall(r'/d/([a-zA-Z0-9_-]{25,})', link)  # Находим идентификатор файла в ссылке
    direct_link = f'https://drive.google.com/uc?export=download&id={id[0]}'  # Формируем прямую ссылку
    add_log_file(direct_link, title='direct_link')
    gdown.download(direct_link, output_path, quiet=True)


# Функция записи логов в файл _log.txt
def add_log_file(text, title=''):
    time_now = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    with open(os.path.join(path, '_log.txt'), "a", encoding='utf-8') as file:
            file.write(f'\n\n{time_now}. {title}.\n\n{format_text(text)}')


# Очистка файла логов
def clear_log():
    open(os.path.join(path, '_log.txt'), 'w').close()


# Функция подсчета количества используемых токенов и стоимость
# https://openai.com/pricing
def tokens_count_and_price(completion, model, to_print=False):
    if model == "gpt-4o":
        # gpt-4o - Input: $5 / 1M tokens - Output: $15 / 1M tokens
        input_price, output_price = 5, 15
    if model == "gpt-4o-mini":
        # gpt-4o-mini - Input: $0.15 / 1M tokens - Output: $0.60 / 1M tokens
        input_price, output_price = 0.15, 0.60
    # цена запроса
    price = input_price * completion.usage.prompt_tokens / 1e6 + \
            output_price * completion.usage.completion_tokens / 1e6
    # словарь значений (для удобства)
    values = {'price': price,
              'input': completion.usage.prompt_tokens,
              'output': completion.usage.completion_tokens,
              'total': completion.usage.total_tokens}
    if to_print: # вывод информации о к-ве токенов и цене по текущему запросу
        print(f"Tokens used: {values['input']} + {values['output']} = {values['total']}. "
              f"*** {model} *** $ {round(price, 5)}")
    global TOKENS
    for key in TOKENS.keys():
        TOKENS[key] += values[key] # суммируем токены и стоимость


# Вывод информации о накопленном к-ве токенов и стоимости
def print_tokens_info():
    global TOKENS
    print(f"\nTokens used: {TOKENS['input']} + {TOKENS['output']} = {TOKENS['total']} "
          f"*** $ {round(TOKENS['price'], 5)}")


# Функция генерации ответа от OpenAI
def generate_answer(prompt_system, prompt_user, prompt_assistant='', model='gpt-4o-mini', temp=0.):
    messages = [
        {"role": "system", "content": prompt_system},
        {'role': 'assistant', 'content': prompt_assistant},
        {"role": "user", "content": prompt_user}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp)
    # Вывод количества используемых токенов и стоимость
    tokens_count_and_price(response, model=model)
    add_log_file(response.choices[0].message.content, title='generate_answer()')
    return response.choices[0].message.content

# Создание векторной базы из текстового файла с Markdown разметкой
def db_from_markdown_file(markdown_file):
    # Открываем и читаем содержимое Markdown файла
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_info = f.read()
    markdown_info = duplicate_lines(markdown_info) # дублирование строк с заголовками '#'
    # Определяем заголовки, по которым будет выполняться разбиение текста
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),]
    # Экземпляр MarkdownHeaderTextSplitter с указанными заголовками для разбиения текста
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # Разбиваем содержимое Markdown файла на чанки на основе заголовков
    chunks = splitter.split_text(markdown_info)
    # Создаем и возвращаем векторную базу данных FAISS
    return FAISS.from_documents(chunks, OpenAIEmbeddings())


# Создание векторной базы из текстового файла с Markdown разметкой
def split_markdown_text(markdown_file,
                        strip_headers=False): # НЕ удалять заголовки под '#..' из page_content
    # Открываем и читаем содержимое Markdown файла
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
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
    # Создаем и возвращаем векторную базу данных FAISS
    return FAISS.from_documents(chunks, OpenAIEmbeddings())


def similarity_info(question, db, k=3):
    # нахождение наиболее похожих документов на основе заданного вопроса
    docs = db.similarity_search(question, k=k)
    # из списока объектов базы знаний docs, извлекаем атрибуты page_content (содержание документов) из каждого docs,
    # добавляем заголовок с форматом '\nChank {i+1}:\n' и объединяем их в строку
    info = '\n'.join([f'\nChank {i+1}:\n' + doc.page_content for i, doc in enumerate(docs)])
    add_log_file(info, title='similarity_info()')
    return info


# Функция ответа на вопрос из баз знаний
def answer_db(question, db_uai, db_add, model='gpt-4o-mini'):
    doc_1 = similarity_info(question, db_uai, k=1)
    doc_2 = similarity_info(question, db_add, k=1)
    system_prompt = """Ты являешься сотрудником техподдержки Университета искусственного
    интеллекта. Ты отвечаешь на вопросы студентов, осуществляешь координаторские функции и помогаешь
    студентам в любых вопросах, связанных с процессом обучения и взаимодействия с Университетом.
    """
    user_prompt = f"""
    Вот информация из дополнительной базы знаний: {doc_2}.
    Вот информация из базы знаний Университета: {doc_1}.
    Вот вопрос студента: {question}.
    Ответь на вопрос студента на основании информации из баз знаний. Не придумывай ничего от себя.
    Выводи ответ в красиво форматированном виде с длинной строки не более 100 символов.
    """
    return generate_answer(system_prompt, user_prompt, model=model)


# Функция диалога (вопрос-ответ) и история диалога
def simple_dialog(questions: list) -> str:
    dialog = '\nОтветьте на уточняющие вопросы, пожалуйста:'
    print(dialog, '\n')
    for i, question in enumerate(questions, 1):
        formatted_question = format_text(f"{question}")
        print(formatted_question, '\n')
        answer = input("Ответ: ")
        print()
        dialog += f"Техподдержка: {formatted_question}\n\nСтудент: {answer}\n\n" # накапливаем историю диалога
    return dialog

###########################################################################################################################################

#@title III. Блок функций взаимодействия с Google Таблицами

service = build('sheets', 'v4') # from googleapiclient.discovery
# Для доступа к таблицам через gspread
# scope = ['https://spreadsheets.google.com/feeds',
#          'https://www.googleapis.com/auth/drive']
# client_gspread = gspread.authorize(
#                  ServiceAccountCredentials.from_json_keyfile_name(google_service_file, scope))

# # **************************************************************************************************
# # Запись и отмена записи на консультации

# # Гугл таблица записей на консультации
# link_tab = 'https://docs.google.com/spreadsheets/d/1rTsRErKwX_-lSYCOOq0O0uS3My5S9uF8l0DUQeUabi4/edit'
# spreadsheet_id = link_tab.split('/')[-2]

# # Словарь слотов времени к именам ячеек
# time_slots = {'10:00': 'A2',
#               '11:00': 'A3',
#               '12:00': 'A4',
#               '13:00': 'A5',
#               '14:00': 'A6',
#               '15:00': 'A7',
#               '16:00': 'A8',
#               '17:00': 'A9',
#               '18:00': 'A10'}


# # Проверка свободных слотов времени на выбранную дату
# def check_free_time_slots(sheet_name: str):
#     service_google = build('sheets', 'v4') # from googleapiclient.discovery
#     # sheet_name: str - имя листа таблицы (у нас имя листа - эта дата)
#     range_ = f'{sheet_name}!A2:A10' # лист и столбец слотов времени
#     response = service.spreadsheets().get(spreadsheetId=spreadsheet_id,
#                                         ranges=range_,
#                                         includeGridData=True).execute()
#     free_time_slots = []
#     for cell in response['sheets'][0]['data'][0]['rowData']:
#         # Если ячейка не закрашена
#         if cell['values'][0]['effectiveFormat']['backgroundColor']['red'] == 1:
#             free_time_slots.append(cell['values'][0]['formattedValue'])
#     if free_time_slots:
#         return {f'{sheet_name}': free_time_slots}
#     else:
#         return None


# # Получение id листа Гугл таблицы по его имени
# def get_sheet_id(sheet_name):
#     spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
#     sheet_index = None
#     for sheet in spreadsheet['sheets']:
#         if sheet['properties']['title'] == sheet_name:
#             sheet_id = sheet['properties']['sheetId']
#             break
#     return sheet_id


# # Раскраска ячеек слотов времени (при бронировании).
# # Только для ячеек времени в диапазоне от А2 до А10 !!!
# def paint_cell(one_cell, sheet_name, colored=True):
#     sheet_id = get_sheet_id(sheet_name) # id листа таблицы по его имени
#     number = int(one_cell[1:]) # номер ячейки А
#     if colored: # закраска
#         background_color = {'red': 0.8784314, 'green': 0.4, 'blue': 0.4}
#     else: # белый цвет
#         background_color = {'red': 1, 'green': 1, 'blue': 1}
#     # Запрос к Google Sheets API для закрашивания ячейки
#     request = {"updateCells": {"rows": [{"values": [
#                 {"userEnteredFormat": {"backgroundColor": background_color}}]}],
#               "fields": "userEnteredFormat.backgroundColor",
#               # для раскраски ячейки обращение только по индексу
#               "start": {"sheetId": sheet_id,    # id листа
#                         "rowIndex": number - 1, # Индекс строки
#                         "columnIndex": 0}}}     # Индекс столбца (A)
#     return service.spreadsheets().batchUpdate(
#                     spreadsheetId=spreadsheet_id,
#                     body={"requests": [request]}).execute()


# # Проверка свободных слотов времени для консультации из Гугл таблицы
# def check_time_slot(order_dict):
#     sheet_name = order_dict['date']
#     time_slot = order_dict['time']
#     # Проверяем, свободен ли каждый временной слот в Гугл Таблице
#     slot_free = True # начальное значение
#     try:
#         # Имя листа таблицы и имя ячейки слота времени (A..)
#         range_ = f'{sheet_name}!{time_slots[time_slot]}'
#     except: return None

#     # Запрос к Google Sheets API для получения информации
#     response = service.spreadsheets().get(
#                         spreadsheetId=spreadsheet_id,
#                         ranges=range_,
#                         includeGridData=True).execute()
#     color = response['sheets'][0]['data'][0]['rowData'][0]['values'][0]['effectiveFormat']['backgroundColor']
#     # если ячейка закрашена, значит временной слот занят
#     if color['red'] != 1: slot_free = False
#     return time_slot if slot_free else None


# # Запись консультации в Гугл таблицу
# def book_slots_time(order_dict, slot):
#     sheet_name = order_dict['date']
#     if slot:
#         paint_cell(time_slots[slot], sheet_name, colored=True) # закраска временных слотов
#         number = int(time_slots[slot][1:]) # номер строки из слота времени
#         # Имя листа таблицы и ячейки от B.. до Е.. в одной строке
#         range_ = f'{sheet_name}!B{number}:E{number}'
#         # данные для записи в таблицу
#         request_body = {'values': [[str(order_dict['student_id']),
#                                         order_dict['student_name'],
#                                         order_dict['topic'],
#                                         order_dict['course_title']]]}
#         response = service.spreadsheets().values().update(
#                             spreadsheetId=spreadsheet_id,
#                             range=range_,
#                             valueInputOption='RAW',
#                             body=request_body).execute()


# # Проверка существующих записей в таблице по id студента
# def check_records(student_id) -> list:
#     spreadsheet = client_gspread.open_by_url(link_tab)
#     student_records = []
#     # по всем листам (вкладкам) таблицы
#     for worksheet in spreadsheet.worksheets():
#         # все student_id в колонке B
#         column_b = worksheet.col_values(2)
#         # номера строк с нужным studen_id в колонке B
#         rows_with_student_id = [row for row, value in enumerate(column_b) if value == str(student_id)]
#         for row in rows_with_student_id:
#             student_records.append([worksheet.title] + worksheet.row_values(row + 1))
#     return student_records


# # Удаление записей на консультацию из таблицы
# def clear_record(order: dict, records: list):
#     # order: dict - ордер на удаление записи
#     # records: list - список существующих записей студента
#     sheet_name = order['date']
#     time_slot = order['time']
#     # пробегаем по существующим записям консультаций
#     for record in records:
#         # если дата в ордере на удаление равна дате в существующей записи
#         if sheet_name == record[0]:
#             # если время существующей записи есть в списке слотов на удаление
#             if record[1] == time_slot:
#                 slot = time_slots[record[1]] # ячейка слота времени А..
#                 number = int(slot[1:]) # номер строки
#                 paint_cell(slot, record[0], colored=False) # Убираем закраску ячейки
#                 # Имя листа таблицы и ячейки от B.. до Е.. в одной строке
#                 range_ = f'{record[0]}!B{number}:E{number}'
#                 # очистка строки
#                 response = service.spreadsheets().values().clear(
#                                     spreadsheetId=spreadsheet_id,
#                                     range=range_).execute()


# # Функция согласования времени для записи на консультацию
# def check_free_time(dialog: str, formated_date: str, model='gpt-4o-mini') -> str:
#     # dialog: str
#     # formated_date: str - формат: yyyy-mm-dd
#     free_time_slots = check_free_time_slots(formated_date) # вывод свободных слотов времени на дату
#     system_prompt = "Ты согласовываешь со студентом время проведения консультации"
#     user_prompt = f""": Вот свободные слоты по времени с интервалом в 60 мин. в виде
#     {{'дата': [список свободных временных слотов]}}: {free_time_slots}\n
#     Здесь история диалога: {dialog}\n
#     Если студент не указал год, используй год от текущей даты: {datetime.now().strftime("%Y-%m-%d")}.
#     - Проанализируй историю диалога.
#     - Проверь желаемое время начала консультации, указанное студентом.
#     - Если есть свободные слоты, сообщи студенту о возможности записи на время свободных слотов.
#     Попроси подтвердить дату и время начала консультации.
#     - Если желаемого времени нет, предложи свободные слоты для записи на эту дату."""
#     return generate_answer(system_prompt, user_prompt, model=model)


# # Функция вывода существующих записей на консультацию
# def existing_records(student_id, model='gpt-4o-mini'):
#     status = check_records(student_id) # существующие записи на консультации
#     user_prompt = f"""
#     Существующие записи на консультацию студента храниться в формате:
#     [['дата консультации (yyyy-mm-dd)',
#       'время консультации (HH:MM)', 'id студента', 'имя студента',
#       'тема консультации (номер урока, этапы диплома и т.д.)'
#       'название учебного курса, в рамках которого была консультация'], ...]
#     Существующие записи на консультацию студента находится здесь: \n{status}
#     Выведи сообщение студенту о его существующих записях без указания 'id студента'.
#     Если записей нет, сообщи студенту, что у него нет записей на консультации."""
#     return generate_answer('', user_prompt, model=model)


# # **************************************************************************************************
# # Получение материалов к Вебинарам

# # Гугл таблица вебинаров
# webinars_link_tab = 'https://docs.google.com/spreadsheets/d/1AtPhTDTpPPfn8udAXZlaHdrFsAaqUJKs7krGSQu6_Nc/edit'
# spreadsheet_webinars = client_gspread.open_by_url(webinars_link_tab)


# # Функция вывода столбца описания 'F' из таблице вебинаров
# def webinars_all_records(col_number=6, webinars_link_tab=webinars_link_tab) -> list:
#     worksheet = spreadsheet_webinars.get_worksheet(0)  # доступ к первому листу
#     column_f = worksheet.col_values(col_number) # список из 6-го столбеца F
#     return [(el, i + 1) for i, el in enumerate(column_f)][1:] # [(название, номер строки),...]


# # Функция вывода ссылок на материалы вебинаров по указанным номерам строк таблицы
# def find_rows_records(rows_number: list, webinars_link_tab=webinars_link_tab) -> list:
#     worksheet = spreadsheet_webinars.get_worksheet(0) # доступ к первому листу
#     return [worksheet.row_values(row) for row in rows_number] # строки по заданным номерам


# # Находим нужные номера строк запрашиваемых вебинаров в таблице
# def get_webinars_row(order, model='gpt-4o-mini') -> list:
#     webinars = webinars_all_records() # все описания (только столбец 'F')
#     prompt_user = f"""
#     # Из ордера на получение данных возьми только дату (если есть) и название вебинара:
#     {{'date': 'дата вебинара', 'title': 'название вебинара'}}
#     Вот ордер на получение данных: {order}
#     # Информация о вебинарах храниться в базе в формате:
#     [('дата и название вебинара', номер строки), ...]
#     Вот база с информацией о вебинарах: {webinars}
#     # Из ордера на получение данных определи вебинар из базы, который нужен (по названию).
#     Название вебинара в ордере может не точно соответсвовать его названию в базе.
#     Выведи список с номером или номерами строк из базы, соответсвующим запрашиваемым вебинарам.
#     Не выводи ничего лишнего. Выведи только список с номером или номерами строк в скобках: [...]"""
#     return try_eval(generate_answer('', prompt_user, model=model))


# # Вывод ссылок на материалы вебинара из списка в читабельном виде
# def webinars_links(rows_number: list, model='gpt-4o-mini') -> str:
#     rows_records = find_rows_records(rows_number) # строки из таблицы с нужными ссылками
#     prompt_user = f"""Список ссылок на материалы к вебинарам имеет вид:
#     [[ссылка на презентацию к вебинару,
#       ссылка на Google Colab к вебинару,
#       ссылка на видео к вебинару,
#       видео ВК,
#       ссылка на другие материалы,
#       название вебинара,
#       дата,
#       описание], ...]
#     Вот список ссылок на материалы к вебинарам: {rows_records}
#     Некоторые элементы списка могут быть пустыми.
#     Выведи список ссылок на материалы к запрашиваемым вебинарам."""
#     return generate_answer('', prompt_user, model=model)


# # **************************************************************************************************
# # Запись вопросов к координаторам

# # Гугл таблица вопросов координаторам
# coordinators_link_tab = 'https://docs.google.com/spreadsheets/d/15xrZZpT3YyNZvz80LaRd3RliTSljiUuZCNL46gWhEd0/edit'

# # Запись вопроса координаторам в таблицу
# def record_question(student_id, name, question):
#     spreadsheet = client_gspread.open_by_url(coordinators_link_tab)
#     worksheet = spreadsheet.get_worksheet(0)  # доступ к первому листу
#     date_time = datetime.now().strftime("%Y-%m-%d %H:%M")
#     # Данные для записи
#     data = [date_time, student_id, name, question]
#     # Запись строки в конец листа
#     worksheet.append_row(data, value_input_option='USER_ENTERED')