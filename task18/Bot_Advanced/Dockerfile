FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR C:/TGbot/Bot_Advanced

# Копируем только requirements.txt, чтобы использовать кэш
COPY requirements.txt .

# Устанавливаем зависимости
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем оставшиеся файлы проекта
COPY . .

# Указываем команду запуска бота
CMD ["python", "bot.py"]