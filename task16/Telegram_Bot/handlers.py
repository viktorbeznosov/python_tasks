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
import logging
import logger
import os
from dotenv import load_dotenv
from theme_manager import ThemeManager
load_dotenv()

router = Router()

# Вместо динамического создания, определите класс явно
class ThemeSelection(StatesGroup):
    food = State()
    renovation = State()
    movie = State()

# Инициализация менеджера тем
theme_manager = ThemeManager()
system_prompt = """
Ты — AI-ассистент по кулинарии и правильному питанию.
Веди себя как дружелюбный и опытный кулинар, который любит делиться проверенными рецептами и полезными советами о здоровом питании.
Твоя цель — помогать пользователю готовить вкусные и питательные блюда, подбирать рационы под цели (похудение, набор массы, поддержание здоровья) и объяснять основы грамотного питания простыми словами.
Основные принципы общения:
Отвечай просто, понятно и без «псевдонаучных» терминов.
Давай конкретные рецепты, пропорции, время готовки и пошаговые инструкции.
При необходимости уточняй детали: диета пользователя, уровень кулинарных навыков, наличие оборудования или ингредиентов.
Можешь предлагать замену продуктов, если чего-то нет под рукой.
Указывай примерную калорийность, пользу ингредиентов и советы по хранению.
Не поучай — вдохновляй, поддерживай интерес к готовке и здоровому образу жизни.
Начни диалог с короткого приветствия.
"""
theme_manager.add_theme("food", "Кулинария и правильное питание", ThemeSelection.food, system_prompt)
system_prompt = """
Ты — AI-ассистент по ремонту и обустройству жилых помещений.
Веди себя как опытный мастер-строитель с 10+ годами практики, который объясняет вещи просто, без занудства и лишней теории.
Твоя задача — помогать пользователю решать практические вопросы, связанные с ремонтом, строительством и дизайном квартир, домов, дач, комнат, санузлов и кухонь.
Основные правила общения и поведения:
Говори дружелюбно, уверенно и на понятном русском, без сложных терминов.
Объясняй шаг за шагом, как сделать работу правильно (от выбора материалов до отделки).
Подсказывай по стоимости, порядку действий и типовым ошибкам, которых стоит избегать.
Уточняй детали, если вопрос неполный: тип помещения, площадь, бюджет, назначение комнаты.
Не используй точных цен, если нет данных, но можно давать примерные диапазоны.
Не навязывай бренды, но можешь назвать популярные варианты для ориентира.
Начни диалог с короткого приветствия.
"""
theme_manager.add_theme("renovation", "Ремонт квартир", ThemeSelection.renovation, system_prompt)
system_prompt = """
Ты — AI-ассистент по кино.
Веди себя как кинокритик и киноман, который разбирается во всех жанрах — от классики и советского кино до современных сериалов и блокбастеров. Говори живо и увлекательно, будто обсуждаешь фильм с другом после сеанса.
Твоя задача — подбирать фильмы под настроение пользователя, объяснять сюжет без спойлеров, рассказывать о режиссёрах, актёрах, интересных фактах и скрытых смыслах.
Основные принципы общения:
Говори легким, эмоциональным и дружелюбным языком.
Рекомендуй фильмы по запросу (жанр, страна, время, актёр, настроение).
Не раскрывай ключевые сюжетные повороты.
Можешь делиться любопытными фактами — история съёмок, награды, отличительные черты стиля режиссёра.
Если спрашивают про конкретный фильм, делай краткий обзор: жанр, сюжет, актёры, впечатления зрителей.
Уважай вкусы пользователя — не спорь, но можешь мягко предложить альтернативу.
Начни диалог с короткого приветствия.
"""
theme_manager.add_theme("movie", "Кино и немцы", ThemeSelection.movie, system_prompt)

#@title Настройка команд

@router.startup()  # Устанавливаем действия, которые будут выполнены при старте бота
async def set_menu_button(bot: Bot):
    # Определение основных команд для главного меню - кнопка (Menu) слева внизу
    main_menu_commands = [
        BotCommand(command='/start', description='Старт'),  # Добавляем команду /start с описанием "Start"
        BotCommand(command='/options', description='Выбрать тему'),  
        BotCommand(command='/help', description='Помощь'), 
        BotCommand(command='/about', description='Обо мне'),
    ]
    # Устанавливаем основные команды в главное меню бота
    await bot.set_my_commands(main_menu_commands)

# Обработчик команды /start
@router.message(Command('start'))  # Декоратор, который регистрирует хендлер для обработки команды /start
async def cmd_start(message: Message):
    # message - объект класса Message. Содержит всю информацию о сообщении, полученном ботом: текст, отправителя, время отправки и прочее.
    # https://docs.aiogram.dev/en/latest/api/types/message.html

    # Отправляем приветственное сообщение пользователю
    await message.answer("Привет! Я ваш помощник. Я здесь, чтобы пообсуждать с вами эту жизнь. Напишите /help, чтобы узнать больше о моих возможностях.")


# Обработчик команды /help
@router.message(Command('help'))  # Декоратор, который регистрирует хендлер для обработки команды /help
async def cmd_help(message: Message):
    # Отправляем сообщение с доступными командами и инструкциями для пользователя
    await message.answer("""Вот список доступных команд:
                         \n/start - начать работу со мной
                         \n/help - показать это сообщение помощи
                         \n/about - информация обо мне
                         \n/options - выбор темы""")


# Обработчик команды /about
@router.message(Command('about'))  # Декоратор, который регистрирует хендлер для обработки команды /about
async def cmd_about(message: Message):
    # Отправляем информацию о боте, его возможностях и назначении
    await message.answer("Я бот, созданный для демонстрации возможностей Aiogram 3. Я могу выполнять различные команды и помогать вам с вашими запросами.")

#@title Inline-клавиатура
async def inline_keyboard():
    return theme_manager.get_keyboard()

# Хендлер команды /options для отображения Inline-клавиатуры
@router.message(Command('options'))  # Декоратор, который регистрирует хендлер для команды /options
async def handle_options_command(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Выберите одну из тем:", # Отправляем сообщение с клавиатурой
                         reply_markup=await inline_keyboard()) # Получаем созданную Inline-клавиатуру

# Хендлер callback-запросов
@router.callback_query(F.data)
async def handle_callback(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    
    # Получаем состояние из маппинга
    target_state = theme_manager.get_state_mapping()[callback.data]
    
    # Устанавливаем состояние
    await state.set_state(target_state)
    
    # Показываем статус "печатает"
    await callback.message.bot.send_chat_action(
        chat_id=callback.message.chat.id, 
        action=ChatAction.TYPING
    )

    system_prompt = theme_manager.get_prompt_text(callback.data)
    answer = await chat_with_memory(history='', system=system_prompt, user="")
    
    # Сохраняем данные
    await state.update_data(history={user_id: answer})
    await state.update_data(current_theme = {user_id: callback.data})
    logging.info(f"CALLBACK: данные сохранены в state")
    
    # Отправляем ответ
    await callback.message.answer(answer)
    
    # Убираем клавиатуру
    await callback.message.edit_reply_markup(reply_markup=None)    
    await callback.message.delete()   
    await callback.answer()

# Обработка текстового сообщения от пользователя
@router.message(F.text)
async def handle_dialog(message: Message, state: FSMContext):
    user_id = message.from_user.id
    username = message.from_user.username or "без username"
    
    # Получаем текущее состояние
    current_state = await state.get_state()
    
    # Логируем для отладки
    logging.info(f"DIALOG: пользователь {user_id} ({username}), состояние: {current_state}")
    logging.info(f"DIALOG: текст сообщения: {message.text}")

    if current_state is None:
        await message.answer("Сначала выберите тему с помощью /options")
        return
    
    # Получаем данные из state
    data = await state.get_data()
    logging.info(f"DIALOG: данные из state: {data}")
    
    # Безопасно получаем историю
    history_dict = data.get('history', {})
    history = history_dict.get(user_id, '')
    current_theme_dict = data.get('current_theme', {})
    theme = current_theme_dict.get(user_id, '')
    
    # Показываем статус "печатает"
    await message.bot.send_chat_action(
        chat_id=message.chat.id, 
        action=ChatAction.TYPING
    )

    answer = await chat_with_memory(
        history=history, 
        system=theme_manager.get_prompt_text(theme),
        user=message.text
    )
    history += f"\n{message.text}"
    history += f"\n{answer}"

    await message.answer(answer)
    await state.update_data(history={user_id: history})
