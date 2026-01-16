from openai import OpenAI, APIError, AuthenticationError, RateLimitError
import os
from dotenv import load_dotenv

def get_answer_from_gpt(content, model = "gpt-4o-mini"):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", default='https://api.proxyapi.ru/openai/v1')

    if not api_key:
        print("Ошибка: Переменная окружения OPENAI_API_KEY не найдена.")
        print("Создайте файл .env в корне проекта и добавьте:")
        print("OPENAI_API_KEY=ваш_ключ_здесь")
        return None

    try:
        client = OpenAI(
            api_key=api_key,  
            base_url=base_url 
        )
    except Exception as e:
        print(f"Ошибка инициализации клиента: {e}")
        return None

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ]
        )

        answer = completion.choices[0].message.content
        
        # Выводим ответ
        print("\n" + "="*50)
        print("ОТВЕТ:")
        print("="*50)
        print(answer + '\n')
        
        # Выводим информацию о токенах
        print("ИНФОРМАЦИЯ О ТОКЕНАХ:")
        print(f"   • Входящие токены: {completion.usage.prompt_tokens}")
        print(f"   • Исходящие токены: {completion.usage.completion_tokens}")
        print(f"   • Всего токенов: {completion.usage.total_tokens}")
        print("="*50 + "\n")
    except AuthenticationError:
        print("Ошибка аутентификации: Неверный API ключ")
        print("   Проверьте правильность OPENAI_API_KEY в файле .env")
    except RateLimitError:
        print("Превышен лимит запросов или закончились средства на счету")
    except APIError as e:
        print(f"Ошибка API OpenAI: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


content = input('Задайте свой вопрос: ')

get_answer_from_gpt(content=content)