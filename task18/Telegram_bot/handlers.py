from aiogram.types import Message, BotCommand, CallbackQuery
from aiogram import Bot, Router, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup
from aiogram.types import InlineKeyboardButton as IKB
from aiogram.types import KeyboardButton as KB
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.enums import ChatAction 
from libs import chat_with_memory
from libs import chat_with_ai
from libs import save_task
from libs import save_deal
from libs import get_tasks
from libs import get_deals
from libs import delete_task
from libs import change_deal_status
import logging
import logger
import os
from dotenv import load_dotenv
from keyboards import create_correction_keyboard
from keyboards import delete_task_button
from keyboards import create_tasks_list_keyboard
from keyboards import create_deals_list_keyboard
from keyboards import confirm_keyboard
from keyboards import create_statuses_keyboard
load_dotenv()

router = Router()

class ManagerState(StatesGroup):
    task_name = State()         # Название задачи
    task_time = State()         # Время выполнения задачи
    delete_task = State()       # Удаление задачи
    confirm_delete = State()    # Подтверждение удаления
    deal_name = State()         # Название сделки
    deal_cost = State()         # Стоимость сделки
    deal_status = State()       # Статус сделки
    change_deal_status = State()  # Изменение статуса сделки 

async def get_motivation():
    system_prompt = """
    Ты — опытный коуч по продажам. 
    Сгенерируй одну короткую, мощную мотивационную фразу для помощника менеджера по продажам. 
    Фраза должна быть на русском языке, длиной 10–20 слов, вдохновлять на активные звонки клиентам, 
    преодоление отказов и закрытие сделок. 
    Используй яркие метафоры (например, огонь, полёт, битва), позитивный императивный тон и личное обращение ('ты'). 
    Пример стиля: 'Взрывай телефоны звонками — каждый отказ ведёт к твоей победе!' Не добавляй объяснения, только саму фразу.
    """
    motivation_phrase = await chat_with_ai(system=system_prompt)
    return motivation_phrase


# Обработчик команды /start
@router.message(Command('start'))  # Декоратор, который регистрирует хендлер для обработки команды /start
async def handle_cmd_start(message: Message):
    # message - объект класса Message. Содержит всю информацию о сообщении, полученном ботом: текст, отправителя, время отправки и прочее.
    # https://docs.aiogram.dev/en/latest/api/types/message.html

    # Отправляем приветственное сообщение пользователю
    await message.answer("Привет! Я ваш помощник по продажам и планированию сделок. Выберите одно из действий:", reply_markup=create_correction_keyboard())


# Обработчик состояния "Добавить задачу"
@router.callback_query(F.data == "add_task")
async def handle_add_task(callback_query: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback_query.message.answer("Введите название задачи")
    await state.set_state(ManagerState.task_name)    

@router.message(ManagerState.task_name)
async def handle_task_name(message: Message, state: FSMContext):
    await state.update_data(task_name = message.text)
    await message.answer("Введите время выполнения задачи")
    await state.set_state(ManagerState.task_time)

@router.message(ManagerState.task_time)
async def handle_task_time(message: Message, state: FSMContext):
    await state.update_data(task_time = message.text)
    await save_task(state)
    data = await state.get_data()
    logging.info(f"Task data {data}")
    motivation = await get_motivation()
    result_message = f"Задача сохранена: {data['task_name']} - {data['task_time']} \n{motivation}"
    await message.answer(result_message, reply_markup=create_correction_keyboard())
    await state.clear()

# Обработчик состояния "Посмотреть задачи"
@router.callback_query(F.data == "view_tasks")
async def handle_view_tasks(callback_query: CallbackQuery, state: FSMContext):
    await state.clear()
    tasks = await get_tasks()
    if (len(tasks) == 0):
        await callback_query.message.answer("Список задач пуст")
    else:
        await callback_query.message.answer(
            "📋 Список задач:",
            reply_markup=create_tasks_list_keyboard(tasks)
        )

# Обработчик "Удаления задачи"
@router.callback_query(F.data.startswith("delete_task_"))
async def handle_delete_task(callback_query: CallbackQuery, state: FSMContext):
    message_text = callback_query.data
    task_data = message_text.split("_")
    task_number = task_data[2]    
    await state.set_state(ManagerState.delete_task)
    await state.update_data(task_number=task_number)    
    await callback_query.message.answer("Вы точно хотите удалить задачу?", reply_markup = confirm_keyboard())

@router.callback_query(F.data == "confirm")
async def handle_confirm(callback_query: CallbackQuery, state: FSMContext):
    match await state.get_state():
        case ManagerState.delete_task:
            data = await state.get_data()
            task_number = data['task_number']
            if (await delete_task(task_number)):
                await callback_query.message.delete()
                await callback_query.message.answer("Задача удалена")
            else:
                await callback_query.message.answer("Ошибка удаления задачи")
        case _:
            await callback_query.message.answer("Выберите одно из действий:", reply_markup=create_correction_keyboard())

@router.callback_query(F.data == "decline")
async def handle_decline(callback_query: CallbackQuery, state: FSMContext):
    match await state.get_state():
        case ManagerState.delete_task:
            await callback_query.message.delete()
        case _:
            await callback_query.message.answer("Выберите одно из действий:", reply_markup=create_correction_keyboard())

# Обработчик кнопки назад в главное меню"
@router.callback_query(F.data == "back_to_menu")
async def handle_back_to_menu(callback_query: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback_query.message.delete()

# Обработчик состояния "Добавить сделку"
@router.callback_query(F.data == "add_deal")
async def handle_add_deal(callback_query: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback_query.message.answer("Введите название сделки")
    await state.set_state(ManagerState.deal_name)

@router.message(ManagerState.deal_name)
async def handle_deal_name(message: Message, state: FSMContext):
    await state.update_data(deal_name = message.text)
    await message.answer("Введите стоимость сделки")
    await state.set_state(ManagerState.deal_cost)

@router.message(ManagerState.deal_cost)
async def handle_deal_cost(message: Message, state: FSMContext):
    await state.update_data(deal_cost = message.text)
    await message.answer("📊 Выберите статус сделки:", reply_markup= create_statuses_keyboard())
    await state.set_state(ManagerState.deal_status)

@router.message(ManagerState.deal_status)
async def handle_deal_status(message: Message, state: FSMContext):
    await state.update_data(deal_status = message.text)
    status, result_message = await save_deal(state)    
    data = await state.get_data()
    logging.info(f"Deal data {data}")
    await message.answer(result_message, reply_markup=create_correction_keyboard())
    await state.clear()

# Обработчик состояния "Просмотреть сделки"
@router.callback_query(F.data == "view_deals")
async def handle_view_deals(callback_query: CallbackQuery, state: FSMContext):
    await state.clear()
    deals = await get_deals()
    logging.info(f"deals: {deals}")
    if (len(deals) == 0):
        await callback_query.message.answer("Список сделок пуст")
    else:
        await callback_query.message.answer(
            "📋 Список сделок:",
            reply_markup=create_deals_list_keyboard(deals)
        )     

# Обработчик состояния "Получить мотивацию"
@router.callback_query(F.data == "get_motivated")
async def handle_get_motivated(callback_query: CallbackQuery, state: FSMContext):
    await state.clear()
    motivation = await get_motivation()  # Добавлен await
    await callback_query.message.answer(motivation)

# Обработчик "Изменения статуса задачи"
@router.callback_query(F.data.startswith("edit_deal_"))
async def handle_edit_deal(callback_query: CallbackQuery, state: FSMContext):
    message_text = callback_query.data    
    deal_data = message_text.split("_")
    deal_number = deal_data[2]        
    await callback_query.message.answer("📊 Выберите новый статус сделки:", reply_markup= create_statuses_keyboard())
    # Устанавливаем состояние
    await state.set_state(ManagerState.change_deal_status)    
    # Сохраняем ТОЛЬКО номер сделки, статус пока неизвестен!
    await state.update_data(deal_number=deal_number)    
    logging.info(f"Начало изменения статуса для сделки #{deal_number}")
    await callback_query.answer()

@router.message(ManagerState.change_deal_status)
async def handle_change_deal_status(message: Message, state: FSMContext):
    logging.info(f"Получен новый статус: {message.text}")
    
    # Получаем данные из состояния
    data = await state.get_data()
    deal_number = data['deal_number']
    new_status = message.text  # ✅ Вот здесь правильный статус!
    
    # Обновляем статус в состоянии (хотя можно и не обновлять)
    await state.update_data(deal_status=new_status)
    
    # Показываем процесс
    await message.answer(
        f"⏳ Меняем статус сделки #{deal_number} на '{new_status}'...",
        reply_markup=None  # Убираем клавиатуру
    )
    
    # Меняем статус
    success, result_message = await change_deal_status(state)
    await message.answer(
        result_message,
        reply_markup=create_correction_keyboard()
    )
    
    await state.clear()
    