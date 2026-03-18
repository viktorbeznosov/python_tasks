from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
import logging
import logger

def create_correction_keyboard():
    """Создаёт inline-клавиатуру для выбора поля для исправления."""
    buttons = [
        [InlineKeyboardButton(text="Добавить задачу", callback_data="add_task")],
        [InlineKeyboardButton(text="Добавить сделку", callback_data="add_deal")],
        [InlineKeyboardButton(text="Посмотреть задачи", callback_data="view_tasks")],
        [InlineKeyboardButton(text="Посмотреть сделки", callback_data="view_deals")],
        [InlineKeyboardButton(text="Получить мотивацию", callback_data="get_motivated")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def confirm_keyboard():
    """Создаёт inline-клавиатуру для подтверждения."""
    buttons = [
        [InlineKeyboardButton(text="✅", callback_data="confirm")],
        [InlineKeyboardButton(text="🚫", callback_data="decline")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def delete_task_button(task: dict):
    """Создаёт inline-кнопку для удаления задачи."""
    button = InlineKeyboardButton(
        text=f"❌", 
        callback_data=f"delete_task_{task['number']}"
    )
    # Клавиатура с одной кнопкой в ряду
    return InlineKeyboardMarkup(inline_keyboard=[[button]])

def create_tasks_list_keyboard(tasks: list):
    """Создаёт компактный список всех задач с кнопками удаления"""
    buttons = []
    for task in tasks:
        # Для каждой задачи создаём строку с текстом и кнопкой
        buttons.append([
            InlineKeyboardButton(
                text=f"{task['task']} ❌", 
                callback_data=f"delete_task_{task['number']}"
            )
        ])
    
    # Добавляем кнопку "Назад"
    buttons.append([
        InlineKeyboardButton(
            text="🔙 Назад", 
            callback_data="back_to_menu"
        )
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_deals_list_keyboard(deals: list):
    """Создаёт компактный список всех сделок с кнопками удаления"""
    buttons = []
    for deal in deals:
        logging.info(f"Deal button {deal}")
        # Для каждой задачи создаём строку с текстом и кнопкой
        buttons.append([
            InlineKeyboardButton(
                text=f"{deal['deal']} ⚙️", 
                callback_data=f"edit_deal_{deal['number']}"
            )
        ])

        # Добавляем кнопку "Назад"
    buttons.append([
        InlineKeyboardButton(
            text="🔙 Назад", 
            callback_data="back_to_menu"
        )
    ])

    return InlineKeyboardMarkup(inline_keyboard=buttons)

# Функция создания Reply-клавиатуры
def create_statuses_keyboard():
    kb = [
        [
            KeyboardButton(text="Новая"),
            KeyboardButton(text="В работе"),
            KeyboardButton(text="Отложена"),
            KeyboardButton(text="Отменена"),
            KeyboardButton(text="Завершена"),
        ],
    ]
    return ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True
    )