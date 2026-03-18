import logging
import asyncio
from aiogram import Bot, F
import os
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from aiogram import Dispatcher
import handlers

# Настройка логирования
logging.basicConfig(
    # Устанавливаем уровень INFO, чтобы записывать уровни логирования: INFO, WARNING, ERROR, CRITICAL
    level=logging.INFO,
    # Формат сообщения, включающий временную метку, имя логгера, уровень логирования и само сообщение
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bot.log'),  # Запись логов в файл "bot.log" для дальнейшего анализа
              logging.StreamHandler()])  # Вывод логов в консоль для отслеживания работы в реальном времени

load_dotenv() # Загружаем переменные окружения из файла .env

# Получение токена и пароля к базе данных из .env
telegram_token = os.getenv('API_TOKEN')

# Инициализация бота и диспетчера
bot = Bot(token=telegram_token)
dp = Dispatcher(storage=MemoryStorage())
# Включаем маршрутизаторы (роутеры) команд и обработчиков в объект dp для обработки входящих сообщений
dp.include_routers(handlers.router,
                   )

# -------------------------------------------------------------

async def main():
    # Запускаем бота
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())