from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup


# Список услуг
services = [
    "Подбор и установка новых окон",
    "Ремонт существующих окон/замена стеклопакетов",
    "Остекление/утепление балконов и лоджий",
    "Остекление веранд и беседок",
    "Демонтаж старых окон",
    "Установка москитных сеток",
    "Свой индивидуальный вариант"
]


def create_inline_keyboard(services):
    """Создаёт inline-клавиатуру для выбора услуг."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"{idx+1}. {service}", callback_data=f"service_{idx+1}")]
        for idx, service in enumerate(services)
    ])
    return keyboard


# Функция создания Reply-клавиатуры
async def create_reply_keyboard():
    kb = [
        [
            KeyboardButton(text="Онлайн-консультант"),
            KeyboardButton(text="Связь с менеджером компании"),
            KeyboardButton(text="Выбрать услугу")
        ],
    ]
    return ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True
    )


def create_correction_keyboard():
    """Создаёт inline-клавиатуру для выбора поля для исправления."""
    buttons = [
        [InlineKeyboardButton(text="1. Тип объекта", callback_data="edit_0")],
        [InlineKeyboardButton(text="2. Местоположение", callback_data="edit_1")],
        [InlineKeyboardButton(text="3. Вы (частное лицо/компания)", callback_data="edit_2")],
        [InlineKeyboardButton(text="4. Особые пожелания", callback_data="edit_3")],
        [InlineKeyboardButton(text="✅ Готово", callback_data="done_editing")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def create_correction_keyboard_final():
    keyboard = [
        [InlineKeyboardButton(text="✅ Всё верно", callback_data="final_confirm")],
        [InlineKeyboardButton(text="Внести исправления", callback_data="final_edit")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)