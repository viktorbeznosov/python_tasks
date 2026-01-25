# импортируем необходимые библиотеки
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import re
import requests
from langchain_core.documents import Document
import logging
from openai import OpenAI
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
#logging.getLogger("chromadb").setLevel(logging.ERROR)
import tiktoken
import io
from docx import Document
import math
import re
import os
from dotenv import load_dotenv

def load_openai_config():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", default='https://api.proxyapi.ru/openai/v1')

    if not api_key:
        print("Ошибка: Переменная окружения OPENAI_API_KEY не найдена.")
        print("Создайте файл .env в корне проекта и добавьте:")
        print("OPENAI_API_KEY=ваш_ключ_здесь")
        return False

    return {'api_key': api_key, 'base_url': base_url}

def load_document_text(url: str) -> str:
    try:
        # Извлекаем идентификатор документа из URL
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
        if match_ is None:
            raise ValueError('Invalid Google Docs URL')
        doc_id = match_.group(1)

        # Загружаем документ как простой текст
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()  # Проверяем успешность ответа
        document_text = response.text
        return document_text
    except requests.exceptions.HTTPError as http_err:
        # Ошибка доступа к документу, например, документ не расшарен
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        # Обработка других ошибок
        print(f"An error occurred: {err}")
        return None
    
"""
Функция split_text_into_fragments разбивает заданный текст на фрагменты, основываясь на двух критериях:
разделении по предложениям и ограничении длины каждого фрагмента.

"""

def split_text_into_fragments(text, num_questions):
    # Разделяем текст на предложения, сохраняя точки
    sentences = re.split(r'(?<=\.)\s', text)

    # Вычисляем ориентировочную длину каждого фрагмента
    desired_length_per_fragment = len(text) / num_questions

    # Ограничиваем максимальную длину фрагмента 5000 символами
    max_fragment_length = min(desired_length_per_fragment, 5000)

    fragments = []  # Список для хранения фрагментов текста
    current_fragment = ''  # Текущий фрагмент

    for sentence in sentences:
        # Проверяем, не превысит ли длина фрагмента ориентировочную длину при добавлении следующего предложения
        if len(current_fragment) + len(sentence) > max_fragment_length:
            # Если превышает, сохраняем текущий фрагмент и начинаем новый
            fragments.append(current_fragment)
            current_fragment = sentence
        else:
            # Иначе, добавляем предложение к текущему фрагменту
            current_fragment += (' ' if current_fragment else '') + sentence

    # Добавляем последний фрагмент, если он не пустой
    if current_fragment:
        fragments.append(current_fragment)

    # Если количество фрагментов больше, чем num_questions, присоединяем последний фрагмент к предпоследнему
    if len(fragments) > num_questions:
        fragments[-2] += ' ' + fragments[-1]
        fragments.pop()  # Удаляем последний фрагмент после присоединения

    return fragments

# функция, которая генерирует открытые вопросы, без вариантов ответов
def generate_questions_with_correct_answer(fragment):

    client = OpenAI()

    system = '''
    Ты - нейро-экзаменатор, который всегда четко выполняет инструкции. Твоя задача - сгенерировать открытый вопрос на основе предоставленного тебе
    текста, а так же сгенерировать правильный ответ на сгенерированный тобою вопрос. Внимательно проанализируй текст и выдели ключевые факты или информацию, которая позволит сформулировать вопрос.
    Вопрос должен быть коротким и понятным, чтобы читатель мог быстро оценить, на что он должен ответить.
    Убедись, что информация в тексте четко подтверждает правильный ответ и не допускает альтернативных интерпретаций.
    Вопрос должен быть основан исключительно на этой ключевой информации.

    В итоге, твой ответ должен быть в формате: строка, состоящая из:
    1. вопроса по содержанию текста с пояснением: "##_ Вопрос"
    2. правильного ответа на поставленный вопрос с пояснением "##_ Правильный ответ"
    '''

    user_assist = """Сгенерируй вопрос и правильный ответ на него на основе данного текста: 'Теоретический аспект: при исследовании
    некоторой задачи результаты теории алгоритмов позволяют ответить на вопрос – является ли эта задача в принципе алгоритмически разрешимой
    – для алгоритмически неразрешимых задач возможно их сведение к задаче останова машины Тьюринга. В случае алгоритмической разрешимости задачи
    – следующий важный теоретический вопрос – это вопрос о принадлежности этой задачи к классу NP–полных задач, при утвердительном ответе на который,
    можно говорить о существенных временных затратах для получения точного решения для больших размерностей исходных данных.'"""

    assist = """
    ##_ В каком году и кем были опубликованы первые фундаментальные работы по теории алгоритмов, в которых были предложены эквивалентные формализмы алгоритма (машина Тьюринга, машина Поста и лямбда-исчисление)?
    ##_ В 1936 году — независимо Аланом Тьюрингом, Алоизом (Алonzo) Чёрчем и Эмилем Постом.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature = 0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_assist},
            {'role':'assistant', 'content': assist},
            {"role": "user", "content": f'Сгенерируй вопрос и правильный ответ на него на основе данного текста: {fragment}.'}
        ]
    )

    # сгенерированный ответ
    generated_response = response.choices[0].message.content

    # код обрабатывает список строк answers. Для каждой строки item в answers, он разделяет её на подстроки, используя '##_ ' как разделитель.
    # После этого, срез [1:] возвращает все подстроки, кроме первой. Таким образом, удаляется часть строки, которая находится до '##_ ', включая саму эту метку.
    processed_data = generated_response.split("##_")[1:]
    return {"question": processed_data[0].strip(), "correct_answer": processed_data[1].strip()}

# функция, которая проверяет ответы студента на основании текста и вопроса
def verify_answers(fragment, question, user_answer, correct_answer):
    client = OpenAI()
    system = '''
        Ты - нейро-экзаменатор, который всегда четко выполняет инструкции. Твоя задача - на основе предоставленного тебе текста и правильного ответа на вопрос проверить правильность ответа студента на вопрос.
        Внимательно проанализируй текст, вопрос, заданный студенту, правильный ответ на него, и ответ студента.
        Сравни ответ студента с правильным ответом
        Оцени правильность ответа студента по шкале от -2 до +2 в соответствии, насколько ответ студента совпадает с правильным ответом, где:

        -2 - (верность ответа составляет от 0% до 20% от правильного ответа);
        -1 - (верность ответа составляет от 21% до 40% от правильного ответа);
        0 - (верность ответа составляет от 41% до 60% от правильного ответа);
        +1 - (верность ответа составляет от 61% до 80% от правильного ответа);
        +2 - (верность ответа составляет от 81% до 100% от правильного ответа).


        Учитывай при оценке глубину и точность ответа, а также его соответствие ключевым аспектам вопроса и предоставленного текста.
        Стремись к объективной и точной оценке, основанной на содержании ответа и его релевантности поставленному вопросу.
        Укажи точную оценку и краткий комментарий, почему ты поставил такую оценку в следующем формате:
        1. "##_ Оценка: Оценка ответа по шкале от -2 до +2"
        2. "##_ Пояснение: Краткое пояснение, почему ты поставил такую оценку"
        3. "##_ Правильный ответ: Правильный ответ: " 
        Строго следуй указанному формату ответа.

        '''

    user_assist = """Оцени правильность ответа студента на вопрос по тексту по шкале от -2 до +2. Текст: 'Теоретический аспект: при исследовании
        некоторой задачи результаты теории алгоритмов позволяют ответить на вопрос – является ли эта задача в принципе алгоритмически разрешимой
        – для алгоритмически неразрешимых задач возможно их сведение к задаче останова машины Тьюринга. В случае алгоритмической разрешимости задачи
        – следующий важный теоретический вопрос – это вопрос о принадлежности этой задачи к классу NP–полных задач, при утвердительном ответе на который,
        можно говорить о существенных временных затратах для получения точного решения для больших размерностей исходных данных.' 
        Вопрос: 'В каком году и кем были опубликованы первые фундаментальные работы по теории алгоритмов, в которых были предложены эквивалентные формализмы алгоритма (машина Тьюринга, машина Поста и лямбда-исчисление)?'. 
        Ответ студента: 'В 1936 году — независимо Аланом Тьюрингом.' 
        Правильный ответ: В 1936 году — независимо Аланом Тьюрингом, Алоизом (Алonzo) Чёрчем и Эмилем Постом.
        """

    assist = """
    3##_ +2 
    ##_ Пояснение: ответ студента полностью соответствует заданному вопросу, он точно отражает ключевые аспекты текста и хорошо структурирован
    ##_ Правльный ответ: В 1936 году — независимо Аланом Тьюрингом, Алоизом (Алonzo) Чёрчем и Эмилем Постом.
    """

    response = client.chat.completions.create(
            model="gpt-4o",
            temperature = 0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_assist},
                {'role':'assistant', 'content': assist},
                {"role": "user", "content": f"Оцени правильность ответа студента на вопрос по тексту по шкале от -2 до +2. Текст:  '{fragment}'. Вопрос: '{question}', Ответ студента: '{user_answer}', Правильный ответ {correct_answer}"}
            ]
        )
    ans = response.choices[0].message.content
    response_parts = ans.split("##_ ")[1:]  # Разделяем ответ на части и удаляем пустые элементы
    return response_parts

def start_exam_dialogue():
    print("Добро пожаловать в систему нейро-экзаменатора!")

    # Получение URL документа от пользователя
    document_url = input("Пожалуйста, введите ссылку на Google Документ с текстом для экзамена: ")

    # Получение количества вопросов
    try:
        num_questions = int(input("Сколько вопросов вы хотите сгенерировать? "))
    except ValueError:
        print("Пожалуйста, введите корректное число.")
        return

    # Загрузка и разделение текста документа на фрагменты
    document_text = load_document_text(document_url)
    document_fragments = split_text_into_fragments(document_text, num_questions)

    # Генерация вопросов и сбор ответов студента
    # накапливаем сгенерированные вопросы
    questions_and_answers = []
    for fragment in document_fragments:
        processing_data = generate_questions_with_correct_answer(fragment)
        
        print(f"\nВопрос: {processing_data['question']}")
        student_answer = input("Введите ваш ответ: ")
        questions_and_answers.append(
            {
                'question': processing_data['question'], 
                'student_answer': student_answer, 
                'correct_answer': processing_data['correct_answer']
            }
        )

    # Проверка ответов студента
    for (i, item) in enumerate(questions_and_answers, start=1):
        fragment = document_fragments[i - 1]  # Получение соответствующего фрагмента текста
        evaluation = verify_answers(fragment, item['question'], item['student_answer'], item['correct_answer'])
        print(f"\nРезультаты по вопросу {i}:")
        print("Оценка:", evaluation[0])  # Выводит оценку
        print(evaluation[1])  # Выводит пояснение
        print(evaluation[2])  # Выводит правильный ответ

_ = load_openai_config()

start_exam_dialogue()