import requests
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from pprint import pprint as pp
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple

class ConfigLoader:
    """Класс для загрузки конфигурации из .env файла."""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", default='https://api.proxyapi.ru/openai/v1')
        self.yandex_api_key = os.getenv("YANDEX_API_ACCESS_KEY", default='')
        self.yandex_api_url = os.getenv("YANDEX_API_URL", default="")
    
    def validate(self) -> bool:
        """Проверяет наличие необходимых ключей."""
        if not self.api_key:
            print("Ошибка: Переменная окружения OPENAI_API_KEY не найдена.")
            print("Создайте файл .env в корне проекта и добавьте:")
            print("OPENAI_API_KEY=ваш_ключ_здесь")
            return False
        return True
    
    def get_openai_config(self) -> Dict[str, str]:
        """Возвращает конфигурацию OpenAI."""
        return {
            'api_key': self.api_key,
            'base_url': self.base_url
        }
    
    def get_yandex_config(self) -> Dict[str, str]:
        """Возвращает конфигурацию Yandex Weather API."""
        return {
            'api_key': self.yandex_api_key,
            'url': self.yandex_api_url
        }

class OpenAIClient:
    """Класс для работы с OpenAI API."""
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    def generate_answer(
        self, 
        system: str = "", 
        user: str = "", 
        assistant: str = "", 
        user_assistant: str = "", 
        model: str = 'gpt-4o-mini', 
        temp: float = 0.1
    ) -> str:
        """
        Генерирует ответ от OpenAI API.
        
        Args:
            system: системный промпт
            user: текущий запрос пользователя
            assistant: предыдущий ответ ассистента
            user_assistant: предыдущий запрос пользователя
            model: модель OpenAI
            temp: температура (0.0 - 1.0)
        
        Returns:
            Ответ от модели
        """
        messages = self._build_messages(system, user, assistant, user_assistant)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Ошибка при вызове OpenAI API: {e}")
            return f"Извините, произошла ошибка: {e}"
    
    def _build_messages(
        self, 
        system: str, 
        user: str, 
        assistant: str, 
        user_assistant: str
    ) -> list:
        """Формирует список сообщений для API."""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        if user_assistant:
            messages.append({"role": "user", "content": user_assistant})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        if user:
            messages.append({"role": "user", "content": user})
        
        return messages
    
class WeatherService:
    """Класс для работы с Yandex Weather API."""
    
    def __init__(self, config: ConfigLoader):
        self.api_key = config.yandex_api_key
        self.api_url = config.yandex_api_url
    
    def get_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Получает прогноз погоды по координатам.
        
        Args:
            lat: широта
            lon: долгота
        
        Returns:
            Словарь с данными о погоде или None при ошибке
        """
        if not self.api_key or not self.api_url:
            print("Ошибка: Не настроены ключи Yandex Weather API")
            return None
        
        headers = {'X-Yandex-Weather-Key': self.api_key}
        
        try:
            response = requests.get(
                f'{self.api_url}/v2/forecast?lat={lat}&lon={lon}', 
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к Yandex Weather API: {e}")
            return None

class JSONValidator:
    """Класс для валидации JSON."""
    
    @staticmethod
    def is_valid(json_string: str) -> bool:
        """Проверяет, является ли строка валидным JSON."""
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False
    
    @staticmethod
    def parse(json_string: str) -> Optional[Dict[str, Any]]:
        """Парсит JSON строку в словарь."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return None
        
class PromptBuilder:
    """Класс для построения промптов."""
    
    def __init__(self):
        self.today = date.today()
        self.tomorrow = self.today + timedelta(days=1)
        self.day_after = self.today + timedelta(days=2)
    
    def build_system_prompt(self) -> str:
        """Строит системный промпт для определения погодного запроса."""
        return f"""
        Ты эксперт по геокодированию для прогноза погоды. 

        1. ПРОВЕРЬ, интересуется ли пользователь ПОГОДОЙ

        2. ЕСЛИ пользователь спрашивает про ПОГОДУ:
        - Найди название города (первый найденный)
        - Определи ДАТУ прогноза:
            * Сегодня - если не указана дата ("погода в Москве"), "сегодня", "сейчас"
            * Завтра - "на завтра", "на следующий день"? 
            * Послезавтра - "послезавтра", "на 3-й день", "через 2 дня"
        - Верни ТОЛЬКО JSON:
        {{
            "lat": 55.7558,
            "lon": 37.6176,
            "forecast_date": "{self.today}",
            "city": "Название города, который ввел пользователь"
        }}

        3. ЕСЛИ НЕ про погоду - верни ТОЛЬКО текст: "Извините, но я даю только прогноз погоды и не отвечаю на другие вопросы"

        Правила:
        - Только один город (первый найденный)
        - Точные известные координаты
        - НИКАКОГО дополнительного текста кроме указанных ответов
        - Текущая дата: {self.today}
        - Формат даты: YYYY-MM-DD
        """
    
    def build_example_assistant(self, city: str = "Москва") -> str:
        """Строит пример ответа ассистента."""
        return f"""
        {{
            "lat": 55.7522,
            "lon": 37.6156,
            "forecast_date": "{self.tomorrow}",
            "city": "{city}"
        }}
        """
    
    @staticmethod
    def build_weather_query(weather_data: Dict[str, Any], city: str, date: str) -> str:
        """Строит запрос для форматирования погодных данных."""
        return f"""
        Из json {weather_data} найди:
        температуру, осадки, влажность, давление
        И дай ответ в формате
        
        Город: {city}
        Прогноз на {date}
        Температура: 22.73 C° пасмурно
        Влажность: 66 %
        Давление: 1016 мм.рт.ст.
        Ветер: 4.1 м/с
        """

class WeatherBot:
    """Основной класс бота для прогноза погоды."""
    
    def __init__(self):
        self.config = ConfigLoader()
        if not self.config.validate():
            raise ValueError("Не удалось загрузить конфигурацию")
        
        self.openai = OpenAIClient(self.config)
        self.weather_service = WeatherService(self.config)
        self.validator = JSONValidator()
        self.prompt_builder = PromptBuilder()
        
        self.system_prompt = self.prompt_builder.build_system_prompt()
        self.example_assistant = self.prompt_builder.build_example_assistant()
        self.example_user = "Как погода будет завтра в Москве"
    
    def process_weather_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """
        Обрабатывает запрос на получение погоды.
        
        Args:
            request_data: словарь с данными запроса (lat, lon, forecast_date, city)
        
        Returns:
            Отформатированный ответ о погоде или None при ошибке
        """
        try:
            lat = float(request_data.get('lat', 0))
            lon = float(request_data.get('lon', 0))
            date = request_data.get('forecast_date', '')
            city = request_data.get('city', 'Неизвестный город')
            
            # Получаем данные о погоде
            weather_data = self.weather_service.get_weather(lat, lon)
            if not weather_data:
                return "Извините, не удалось получить данные о погоде."
            
            # Формируем запрос для форматирования ответа
            query = self.prompt_builder.build_weather_query(weather_data, city, date)
            
            # Получаем отформатированный ответ
            return self.openai.generate_answer(
                system="Ты помощник, который красиво форматирует данные о погоде",
                user=query
            )
            
        except Exception as e:
            print(f"Ошибка при обработке запроса погоды: {e}")
            return None
    
    def handle_message(self, user_message: str) -> str:
        """
        Обрабатывает сообщение пользователя.
        
        Args:
            user_message: сообщение пользователя
        
        Returns:
            Ответ бота
        """
        # Получаем ответ от OpenAI о том, что нужно сделать
        ai_response = self.openai.generate_answer(
            system=self.system_prompt,
            user=user_message,
            assistant=self.example_assistant,
            user_assistant=self.example_user
        )
        
        # Проверяем, является ли ответ JSON (значит нужна погода)
        if self.validator.is_valid(ai_response):
            request_data = self.validator.parse(ai_response)
            if request_data:
                weather_response = self.process_weather_request(request_data)
                if weather_response:
                    return weather_response
                else:
                    return "Извините, произошла ошибка при получении данных о погоде."
        
        # Если не JSON, возвращаем как есть (это текстовая ошибка или отказ)
        return ai_response
    
    def run(self):
        """Запускает диалог с пользователем."""
        print("Бот для прогноза погоды запущен. Введите 'стоп' для выхода.")
        
        while True:
            try:
                user_input = input("Вы: ").strip()
                
                if user_input.lower() == 'стоп':
                    print("До свидания!")
                    break
                
                if not user_input:
                    continue
                
                response = self.handle_message(user_input)
                print(f"Бот: {response}\n")
                
            except KeyboardInterrupt:
                print("\nПрограмма прервана пользователем.")
                break
            except Exception as e:
                print(f"Произошла непредвиденная ошибка: {e}")

bot = WeatherBot()
bot.run()