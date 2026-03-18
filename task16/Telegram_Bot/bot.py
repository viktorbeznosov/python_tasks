import logging
import asyncio
from aiogram import Bot, Dispatcher
import os
import logger
from dotenv import load_dotenv
import handlers

load_dotenv()
bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
dp = Dispatcher()
dp.include_routers(handlers.router,)

logging.info("Запуск бота...")

async def main():
    try:
        logging.info("Запуск бота...")
        await dp.start_polling(bot)
    finally:
        logging.info("Остановка бота...")
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())