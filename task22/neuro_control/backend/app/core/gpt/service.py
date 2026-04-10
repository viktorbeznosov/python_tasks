# app/core/gpt/service.py

"""
Сервисный слой для обращения к LLM (GPT-модели) через Responses API.

Назначение:
1. Асинхронные запросы к внешнему API LLM
2. Отделение бизнес-логики от маршрутов (routes)
3. Возврат текста модели в стандартизированном виде
4. Логирование запроса и ответа модели
"""

import httpx
from fastapi import Request
from ...config import settings 
from openai import AsyncOpenAI

LLM_BASE_URL = getattr(settings, "LLM_BASE_URL", "https://api.proxyapi.ru/openai/v1")
LLM_MODEL = getattr(settings, "LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = getattr(settings, "LLM_API_KEY", "")
GPT_LOG_ENABLE = False

async def call_gpt(
    request: Request,
    prompt: str,
    temperature: float = 0.7
) -> str:
    """
    Асинхронный вызов GPT-модели через Responses API.

    Параметры:
    - request: Request — объект FastAPI (для логирования)
    - prompt: str — текст запроса
    - temperature: float — креативность ответа

    Возвращает:
    - str: текст ответа модели
    """

    log = request.app.state.log

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "input": prompt,
        "temperature": temperature
    }

    # --------------------------------------------------
    # ЛОГИРОВАНИЕ ЗАПРОСА
    # --------------------------------------------------
    if GPT_LOG_ENABLE:
        await log.log_info(
            target="gpt_request",
            message=(
                f"Модель: {LLM_MODEL}\n"
                f"Temperature: {temperature}\n"
                f"Длина промпта: {len(prompt)} символов\n"
                f"Промпт:\n{prompt[:5000]}"
            )
        )

    try:
        # async with httpx.AsyncClient(timeout=60) as client:
        #     resp = await client.post(
        #         LLM_BASE_URL,
        #         headers=headers,
        #         json=payload
        #     )

        # if resp.status_code != 200:
        #     await log.log_error(
        #         target="gpt_api_error",
        #         message=f"{resp.status_code}: {resp.text}"
        #     )
        #     raise RuntimeError(
        #         f"Ошибка при вызове LLM: {resp.status_code}, {resp.text}"
        #     )

        # data = resp.json()

        # try:
        #     answer = data["output"][0]["content"][0]["text"]
        # except (KeyError, IndexError, TypeError):
        #     await log.log_error(
        #         target="gpt_format_error",
        #         message=str(data)
        #     )
        #     raise RuntimeError(
        #         f"Unexpected LLM response format: {data}"
        #     )

        # # --------------------------------------------------
        # # ЛОГИРОВАНИЕ ОТВЕТА
        # # --------------------------------------------------
        # if GPT_LOG_ENABLE:
        #     await log.log_info(
        #         target="gpt_response",
        #         message=(
        #             f"Длина ответа: {len(answer)} символов\n"
        #             f"Ответ:\n{answer[:5000]}"
        #         )
        #     )

        answer = await chat_with_ai(system = prompt, temperature = temperature)

        return answer

    except Exception as e:
        await log.log_error(
            target="gpt_exception",
            message=str(e)
        )
        raise

# Асинхронная функция обращения к модели с реализацией памяти
async def chat_with_ai(
    system: str,
    temperature: float = 0.7,
    max_tokens: int = 300
):
    # Запрос к GPT с историей
    try:
        client = AsyncOpenAI()
        completion = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
            ],
            temperature = temperature,
            max_tokens = max_tokens
        )

        if completion and completion.choices:            
            answer = completion.choices[0].message.content
        else:
            answer = "Произошла ошибка при получении ответа от ассистента."

    except Exception as e:
        answer = "Произошла ошибка при получении ответа от ассистента."

    return answer