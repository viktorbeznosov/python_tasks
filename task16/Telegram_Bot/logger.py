# Настройка логирования для Docker (stdout) и файла
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Добавили %(name)s
    handlers=[
        logging.StreamHandler(sys.stdout),  # Для Docker compose logs
        logging.FileHandler('./logs/bot.log')  # Сохраняем в файл внутри контейнера
    ]
)
