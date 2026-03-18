from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton as IKB
from aiogram.fsm.state import State, StatesGroup
import logging
import logger

class ThemeManager:
    def __init__(self):
        self.themes = []
        self.states = {}  # Будем хранить состояния здесь
    
    def add_theme(self, theme_id: str, display_text: str, state: State, system_prompt: str):
        """Добавляет новую тему и создает для нее состояние"""
        
        # Сохраняем тему
        theme = {
            "id": theme_id,
            "text": display_text,
            "state": state,
            "prompt": system_prompt
        }
        self.themes.append(theme)
        
        # Также сохраняем в states для обратной совместимости
        self.states[theme_id] = state
        
        return theme
    
    def get_state_mapping(self):
        """Возвращает mapping callback_data -> state"""
        return {theme["id"]: theme["state"] for theme in self.themes}
    
    def get_keyboard(self):
        """Создает клавиатуру на основе тем"""
        kb = [[IKB(text=theme["text"], callback_data=theme["id"])] 
              for theme in self.themes]
        return InlineKeyboardMarkup(inline_keyboard=kb)
    
    def get_theme_text(self, callback_data: str) -> str:
        """Возвращает текст темы по callback_data"""
        for theme in self.themes:
            if theme["id"] == callback_data:
                return theme["text"]
        return callback_data
    
    def get_prompt_text(self, callback_data: str) -> str:
        """Возвращает текст темы по callback_data"""
        for theme in self.themes:
            if theme["id"] == callback_data:
                return theme["prompt"]
        return callback_data    
    
    def get_all_states(self):
        """Возвращает список всех состояний"""
        return [theme["state"] for theme in self.themes]
    
    def get_states_class(self):
        """Возвращает класс состояний ThemeSelection"""
        return self.ThemeSelection