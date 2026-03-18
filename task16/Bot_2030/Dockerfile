# базовый образ
FROM python:3.12-slim

# рабочая директория
WORKDIR /proj/bot_secure_trains

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# копируем проект
COPY . .

# запуск бота
CMD ["python", "bot.py"]
