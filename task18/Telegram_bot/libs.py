import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import logger
import pandas as pd
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# Асинхронная функция обращения к модели с реализацией памяти
async def chat_with_memory(history, system, user):
    # Запрос к GPT с историей
    try:
        client = AsyncOpenAI()
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"""Вот история вашего диалога: {history}. 
                 Ответь на вопрос пользователя {user}"""}
            ],
            temperature=0.15,
        )

        if completion and completion.choices:
            answer = completion.choices[0].message.content
        else:
            answer = "Произошла ошибка при получении ответа от ассистента."

    except Exception as e:
        logging.error(f"Ошибка {e}")
        answer = "Произошла ошибка при получении ответа от ассистента."

    return answer

# Асинхронная функция обращения к модели с реализацией памяти
async def chat_with_ai(system):
    # Запрос к GPT с историей
    try:
        client = AsyncOpenAI()
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system},
            ],
            temperature=0.9,
            max_tokens=100
        )

        if completion and completion.choices:            
            answer = completion.choices[0].message.content
            logging.info(f"answer: {answer}")
        else:
            answer = "Произошла ошибка при получении ответа от ассистента."

    except Exception as e:
        logging.error(f"Ошибка {e}")
        answer = "Произошла ошибка при получении ответа от ассистента."

    return answer
    
async def save_task(state):
    # Получаем данные состояния
    data = await state.get_data()
    task_name = data.get("task_name", "")
    task_time = data.get("task_time", "")

    new_entry = {
        "Номер": 1,
        "Название задачи": task_name,
        "Время выполнения задачи": task_time,
    }

    output_dir = './table/'  # Используем относительный путь
    os.makedirs(output_dir, exist_ok=True)  # Создаем папку, если её нет
    file_name = os.path.join(output_dir, 'tasks.xlsx')
    # Проверяем, существует ли файл
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        total_rows = len(df)
        task_number = total_rows + 1
        new_entry['Номер'] = task_number
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)  # Добавляем новую строку
    else:
        df = pd.DataFrame([new_entry])  # Создаем новый DataFrame
    # Сохраняем файл
    df.to_excel(file_name, index=False)

async def get_tasks(): 
    output_dir = './table/'  # Используем относительный путь
    file_name = os.path.join(output_dir, 'tasks.xlsx')
    result = []
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        buttons = []
        for index, row in df.iterrows():
            result.append({'number': row['Номер'], 'task': f"{row['Название задачи']} {row['Время выполнения задачи']}"})            
    return result

async def delete_task(task_number):
    output_dir = './table/'
    file_name = os.path.join(output_dir, 'tasks.xlsx')

    df = pd.read_excel(file_name)

    # Проверяем, существует ли задача с таким номером
    if int(task_number) not in df['Номер'].values:
        return False

    # Способ 1: Удаляем по номеру (если номер - это значение в столбце)
    df_filtered = df[df['Номер'] != int(task_number)]
    
    logging.info(f"Filtered: {df_filtered}")

    # Логируем результат
    logging.info(f"После удаления: {len(df_filtered)} задач")
    
    # Сохраняем обратно в файл
    df_filtered.to_excel(file_name, index=False)
    
    return True

async def save_deal(state):
    # Получаем данные состояния
    data = await state.get_data()
    deal_name = data.get("deal_name", "")
    deal_cost = data.get("deal_cost", "")
    deal_status = data.get("deal_status", "")

    valid_statuses = ['Новая', 'В работе', 'Отложена', 'Отменена', 'Завершена']

    if deal_status not in valid_statuses:
        return False, f"❌ Недопустимый статус. Допустимые: {', '.join(valid_statuses)}"

    new_entry = {
        "Номер": 1,
        "Название сделки": deal_name,
        "Сумма сделки": deal_cost,
        "Статус сделки": deal_status,
    }

    output_dir = './table/'  # Используем относительный путь
    os.makedirs(output_dir, exist_ok=True)  # Создаем папку, если её нет
    file_name = os.path.join(output_dir, 'deals.xlsx')
    # Проверяем, существует ли файл
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        total_rows = len(df)
        deal_number = total_rows + 1
        new_entry['Номер'] = deal_number
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)  # Добавляем новую строку
    else:
        df = pd.DataFrame([new_entry])  # Создаем новый DataFrame
    # Сохраняем файл
    df.to_excel(file_name, index=False)
    
    return True, f"✅ Сделка сохранена #{deal_name} {deal_cost} {deal_status}"

async def change_deal_status(state):
    output_dir = './table/'
    file_name = os.path.join(output_dir, 'deals.xlsx')

    df = pd.read_excel(file_name)
    logging.info(f"Deals data {df}")
    data = await state.get_data()
    deal_number = int(data['deal_number'])
    new_status = data['deal_status']
        
    valid_statuses = ['Новая', 'В работе', 'Отложена', 'Отменена', 'Завершена']

    if new_status not in valid_statuses:
        return False, f"❌ Недопустимый статус. Допустимые: {', '.join(valid_statuses)}"

    # Проверяем, существует ли сделка с таким номером
    if deal_number not in df['Номер'].values:
        logging.warning(f"Сделка #{deal_number} не найдена")
        return False, f"❌ Сделка #{deal_number} не найдена"
    
    # Получаем старый статус для логирования
    old_status = df.loc[df['Номер'] == deal_number, 'Статус сделки'].values[0]
    
    # Обновляем статус
    df.loc[df['Номер'] == deal_number, 'Статус сделки'] = new_status
    
    # Сохраняем файл
    df.to_excel(file_name, index=False)
    
    logging.info(f"Сделка #{deal_number}: статус изменен '{old_status}' -> '{new_status}'")
    return True, f"✅ Статус сделки #{deal_number} изменен: '{old_status}' → '{new_status}'"

async def get_deals(): 
    output_dir = './table/'  # Используем относительный путь
    file_name = os.path.join(output_dir, 'deals.xlsx')
    result = []
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        buttons = []
        for index, row in df.iterrows():
            result.append({'number': row['Номер'], 'deal': f"{row['Название сделки']} {row['Сумма сделки']} {row['Статус сделки']}"})            
    return result