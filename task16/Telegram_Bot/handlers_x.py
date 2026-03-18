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
from libs import answer_db_index, create_db_index, load_db_index
import logging
import logger
import os
from dotenv import load_dotenv
load_dotenv()

router = Router()
dict_memory = dict()  # Словарь для сохранения истории переписки

#@title Настройка команд

@router.startup()  # Устанавливаем действия, которые будут выполнены при старте бота
async def set_menu_button(bot: Bot):
    # Определение основных команд для главного меню - кнопка (Menu) слева внизу
    main_menu_commands = [
        BotCommand(command='/start', description='Start'),  # Добавляем команду /start с описанием "Start"
        BotCommand(command='/model', description='Model LLM'),  # Добавляем команду /model с описанием "Model"
        BotCommand(command='/options', description='Options'),  # Добавляем команду /options с описанием "Options"
        BotCommand(command='/help', description='Help Information'),  # Добавляем команду /help с описанием "Help Information"
        BotCommand(command='/about', description='About this bot')]  # Добавляем команду /about с описанием "About this bot"
    # Устанавливаем основные команды в главное меню бота
    await bot.set_my_commands(main_menu_commands)

# Обработчик команды /start
@router.message(Command('start'))  # Декоратор, который регистрирует хендлер для обработки команды /start
async def cmd_start(message: Message):
    # message - объект класса Message. Содержит всю информацию о сообщении, полученном ботом: текст, отправителя, время отправки и прочее.
    # https://docs.aiogram.dev/en/latest/api/types/message.html

    # Отправляем приветственное сообщение пользователю
    await message.answer("Привет! Я ваш помощник. Я здесь, чтобы помочь вам. Напишите /help, чтобы узнать больше о моих возможностях.")


# Обработчик команды /help
@router.message(Command('help'))  # Декоратор, который регистрирует хендлер для обработки команды /help
async def cmd_help(message: Message):
    # Отправляем сообщение с доступными командами и инструкциями для пользователя
    await message.answer("Вот список доступных команд:\n/start - начать работу со мной\n/help - показать это сообщение помощи\n/about - информация обо мне")


# Обработчик команды /about
@router.message(Command('about'))  # Декоратор, который регистрирует хендлер для обработки команды /about
async def cmd_about(message: Message):
    # Отправляем информацию о боте, его возможностях и назначении
    await message.answer("Я бот, созданный для демонстрации возможностей Aiogram 3. Я могу выполнять различные команды и помогать вам с вашими запросами.")

#@title Обработка текстовых сообщений
https://console.proxyapi.ru/
# Ловим все текстовые сообщения. Используем MagicFilters - F
@router.message(F.text.startswith('foo'))  # Декоратор для регистрации хендлера сообщений, который будет реагировать на текстовые сообщения.
async def handle_text_message(message: Message):  # Определение асинхронной функции-хендлера, принимающей объект сообщения.
    # message - объект класса Message. Содержит всю информацию о сообщении, полученном ботом: текст, отправителя, время отправки и прочее.
    # https://docs.aiogram.dev/en/latest/api/types/message.html

    ### Здесь можем вставить любой функционал по обработке полученного текстового сообщения (message.text)
    logging.info(f"User {message.from_user.username} writed message: {message.text}")
    await message.answer("bazZZ")  # Отправка ответа пользователю с текстом его сообщения.

# Ловим все текстовые сообщения. Используем MagicFilters - F
@router.message(F.text.in_({'GPT-4o-mini', 'GPT-4o', 'Gemini 1.5 Flash', 'Gemini 1.5 Pro', 'Llama 3'}))  # Декоратор для регистрации хендлера сообщений, который будет реагировать на текстовые сообщения.
async def handle_text_message(message: Message):  # Определение асинхронной функции-хендлера, принимающей объект сообщения.
    # message - объект класса Message. Содержит всю информацию о сообщении, полученном ботом: текст, отправителя, время отправки и прочее.
    # https://docs.aiogram.dev/en/latest/api/types/message.html

    ### Здесь можем вставить любой функционал по обработке полученного текстового сообщения (message.text)

    await message.answer(f"Выбрана модель {message.text}")  # Отправка ответа пользователю с текстом его сообщения.

#@title Inline-клавиатура

# Функция создания Inline-клавиатуры
async def inline_keyboard():
    # Список списков. Внутренний список - это кнопки в одну строку
    kb = [
        # Первая строка кнопок
        [IKB(text="LLM/GPT", callback_data='llm'), # Создаем кнопку с callback_data 'llm'
         IKB(text="DS", callback_data='ds')], # Создаем кнопку с callback_data 'ds'
        # Вторая строка кнопок.   # Создаем кнопку для перехода по URL
        [IKB(text="Перейти на сайт УИИ", url="https://neural-university.ru/")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb) # Возвращаем объект Inline-клавиатуру


# Функция создания такой же Inline-клавиатуры используя InlineKeyboardBuilder
async def inline_keyboard_builder():
    builder = InlineKeyboardBuilder() # Создаем экземпляр класса InlineKeyboardBuilder
    builder.button(text="LLM/GPT", callback_data='llm')  # Создаем кнопку с callback_data 'llm'
    builder.button(text="DS", callback_data='ds')  # Создаем кнопку с callback_data 'ds'
    builder.button(text="Перейти на сайт УИИ", url="https://neural-university.ru/")  # Создаем кнопку для перехода по URL
    builder.adjust(2)  # Устанавливаем количество кнопок в строке (2 кнопки в одной строке)
    return builder.as_markup() # Возвращаем объект Inline-клавиатуру

# Хендлер команды /options для отображения Inline-клавиатуры
@router.message(Command('options'))  # Декоратор, который регистрирует хендлер для команды /options
async def handle_options_command(message: Message):
    await message.answer("Выберите одну из опций:", # Отправляем сообщение с клавиатурой
                         reply_markup=await inline_keyboard_builder()) # Получаем созданную Inline-клавиатуру


# Хендлер callback-запросов
@router.callback_query(F.data)  # Декоратор для дюбых callback запросов
async def handle_callback(callback: CallbackQuery):
    if callback.data == 'llm':
        await callback.message.answer("Вы выбрали LLM/GPT")
    if callback.data == 'ds':
        await callback.message.answer("Вы выбрали DS")
    await callback.message.answer(f"Значение callback.data:  {callback.data}")
    # Убираем Inline-клавиатуру из сообщения
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer()  # Ответ на callback для предотвращения зависания интерфейса

# Определяем состояния
class ModelSelection(StatesGroup):
    choosing_model = State()

# Словарь для хранения выбора (если не используете FSM)
user_models = {}

# Функция создания Inline-клавиатуры
async def model_selection_keyboard():
    builder = InlineKeyboardBuilder()
    
    models = [
        ("🤖 GPT-4o-mini", "gpt4o_mini"),
        ("🚀 GPT-4o", "gpt4o"),
        ("✨ Gemini 1.5 Flash", "gemini_flash"),
        ("🌟 Gemini 1.5 Pro", "gemini_pro"),
        ("🦙 Llama 3", "llama3")
    ]
    
    for text, callback_data in models:
        builder.button(
            text=text,
            callback_data=f"model_{callback_data}"
        )
    
    builder.adjust(2)
    return builder.as_markup()

# Обработчик команды /model
@router.message(Command('model'))
async def cmd_model(message: Message, state: FSMContext):
    await state.set_state(ModelSelection.choosing_model)
    await message.answer(
        "🎯 Выберите модель для обработки запросов:",
        reply_markup=await model_selection_keyboard()
    )

# Обработчик выбора модели
@router.callback_query(ModelSelection.choosing_model, F.data.startswith("model_"))
async def model_chosen(callback: CallbackQuery, state: FSMContext):
    model = callback.data.replace("model_", "")
    
    model_names = {
        "gpt4o_mini": "GPT-4o-mini",
        "gpt4o": "GPT-4o",
        "gemini_flash": "Gemini 1.5 Flash",
        "gemini_pro": "Gemini 1.5 Pro",
        "llama3": "Llama 3"
    }
    
    model_name = model_names.get(model, model)
    
    # Сохраняем выбор пользователя
    await state.update_data(selected_model=model)
    user_models[callback.from_user.id] = model
    
    await callback.answer(f"✅ Выбрана модель: {model_name}")
    
    # Обновляем сообщение с клавиатурой
    await callback.message.edit_text(
        f"✅ Модель **{model_name}** выбрана!\n"
        f"Теперь вы можете задавать вопросы.",
        reply_markup=None
    )
    
    await state.clear()
    logging.info(f"User {callback.from_user.id} selected model: {model_name}")

# Обработчик обычных сообщений с использованием выбранной модели
@router.message()
async def handle_message(message: Message, state: FSMContext):
    user_id = message.from_user.id
    
    # Получаем выбранную модель пользователя
    data = await state.get_data()
    model = data.get('selected_model') or user_models.get(user_id, 'gpt4o_mini')
    
    if not model:
        await message.answer("Сначала выберите модель через /model")
        return
    
    # Здесь ваш код для обработки сообщения с выбранной моделью
    await message.answer(f"Обрабатываю запрос моделью {model}...")

