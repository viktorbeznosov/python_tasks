import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import logger

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# Асинхронная функция обращения к модели с реализацией памяти
async def chat_with_memory(history, system, user):
    # Запрос к GPT с историей
    try:
        client = AsyncOpenAI()
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"""Вот история вашего диалога: {history}. 
                 Ответь на вопрос пользователя {user}"""}
            ],
            temperature=0.15,
        )

        if completion and completion.choices:
            answer = completion.choices[0].message.content
        else:
            answer = "Произошла ошибка при получении ответа от ассистента."

    except Exception as e:
        logging.error(f"Ошибка {e}")
        answer = "Произошла ошибка при получении ответа от ассистента."

    return answer
