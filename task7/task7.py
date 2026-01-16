from openai import OpenAI, APIError, AuthenticationError, RateLimitError
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import requests
from langchain_core.documents import Document
import textwrap

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
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)

    try:        
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()
        text = response.text
    except requests.exceptions.HTTPError as e:
         print(f"HTTP ошибка: {e}")
         return False
    except requests.exceptions.Timeout:
        print("Таймаут соединения")
        return False
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        return False

    return text

def format_text(text, width=120):
    paragraphs = text.split('\n')
    formatted_paragraphs = []
    for paragraph in paragraphs:
        formatted_paragraph = textwrap.fill(paragraph, width)
        formatted_paragraphs.append(formatted_paragraph)
    return '\n'.join(formatted_paragraphs)



def answer_index_history(system, query, history, db_index, openai,
                         model="gpt-4o-mini", verbose=False):
    docs = db_index.similarity_search(query, k=5)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'\nОтрывок документа №{i+1}\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    if verbose: print('\n\n', message_content)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content":
         f"Ответь на вопрос клиента. Не упоминай документ с информацией для ответа клиенту в ответе. \
         Не отвечай на вопросы, не касающиеся документа \
         Документ с информацией для ответа клиенту: \n\n{message_content} \
         \n\nИстория диалога: \n{history} \
         \n\nВопрос клиента: \n{query}"}
    ]

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        answer = completion.choices[0].message.content
    except AuthenticationError:
        print("Ошибка аутентификации: Неверный API ключ")
        print("Проверьте правильность OPENAI_API_KEY в файле .env")
        return False
    except RateLimitError:
        print("Превышен лимит запросов или закончились средства на счету")
        return False
    except APIError as e:
        print(f"Ошибка API OpenAI: {e}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False    
    return answer

def neuro_consultant(url, system):
    data_from_url = load_document_text(url)
    chunks = re.split(r"[IVXLCDM]+\.\s", data_from_url)
    source_chunks=[]

    if (len(chunks) == 1):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)

        for chunk in splitter.split_text(data_from_url):
            source_chunks.append(Document(page_content=chunk, metadata={"meta":"data"}))
    else:
        for chunk in chunks:
            source_chunks.append(Document(page_content=chunk, metadata={"meta":"data"}))

    openai_configs = load_openai_config()
    embeddings = OpenAIEmbeddings()   
    db = FAISS.from_documents(source_chunks, embeddings)

    try:
        client = OpenAI(
            api_key=openai_configs['api_key'],  
            base_url=openai_configs['base_url'] 
        )
    except Exception as e:
        print(f"Ошибка инициализации клиента: {e}")
        return False
    
    history = ""
    while True:
        query = input('Вопрос пользователя: ')
        # выход из цикла, если пользователь ввел: 'стоп'
        if query == 'стоп': break
        # ответ от OpenAI
        answer = answer_index_history(system, query, history, db, client, verbose=False)
        print(f'Ответ:\n{format_text(answer)}\n')
        # Запись истории диалога
        history += f'Вопрос пользователя: {query}. \nОтвет: {answer}\n'

url = 'https://docs.google.com/document/d/1YhUEX9fZDNTeE3eJ-yXskxZG46LsTRYvXjZ9Ij-t3Gw/edit?tab=t.0'
system = 'Я изучаю ТР ТС 001/2011 "О безопасности железнодорожного подвижного состава". Выступайте в роли эксперта по техническому регулированию ЕАЭС. '

neuro_consultant(url=url, system=system)

'''
Примеры вопросов и ответов

Вопрос пользователя: Перечисли, какой транспорт подпадает под данный реграмент?
Ответ:
Данный регламент распространяется на следующие виды железнодорожного подвижного состава:

1. Локомотивы.
2. Моторвагонный подвижной состав и его вагоны.
3. Пассажирские вагоны локомотивной тяги.
4. Грузовые вагоны.
5. Специальный железнодорожный подвижной состав.

Также он охватывает составные части указанного подвижного состава.

Вопрос пользователя: Разъясни процедуру оценки соответствия для выше перечисленной техники
Ответ:
Процедура оценки соответствия для железнодорожного подвижного состава включает несколько этапов:

1. **Выбор схемы оценки соответствия**: Заявитель выбирает одну из типовых схем, предусмотренных регламентом, в
зависимости от типа продукции и ее назначения.

2. **Подготовка документов**: Заявитель формирует комплект документов, который включает техническую документацию,
результаты испытаний, обоснование безопасности и другие необходимые материалы.

3. **Идентификация продукции**: Продукция подлежит идентификации для определения ее соответствия требованиям регламента.

4. **Проведение испытаний**: Образцы продукции направляются в аккредитованные испытательные лаборатории для проведения
необходимых исследований и испытаний.

5. **Анализ результатов**: Орган по сертификации анализирует результаты испытаний, а также документы, представленные
заявителем.

6. **Принятие решения**: На основании анализа орган по сертификации принимает решение о выдаче сертификата соответствия
или отказе в его выдаче.

7. **Маркировка**: Продукция, прошедшая оценку соответствия, должна быть маркирована единым знаком обращения продукции
на рынке Союза.

8. **Хранение документов**: Заявитель обязан хранить комплект доказательственных материалов, подтверждающих соответствие
продукции, в течение установленного срока.

Эти этапы обеспечивают соответствие железнодорожного подвижного состава требованиям безопасности и техническим
регламентам.

Вопрос пользователя: Что кушает за обедом крокодил?
Ответ:
Извините, но я не могу ответить на этот вопрос, так как он не относится к теме технического регулирования и безопасности
железнодорожного подвижного состава. Если у вас есть вопросы по данной теме, пожалуйста, задавайте!

Вопрос пользователя: Перечисли основные требования безопасности
Ответ:
Основные требования безопасности для железнодорожного подвижного состава включают:

1. Безопасность излучений.
2. Биологическая безопасность.
3. Взрывобезопасность.
4. Механическая безопасность.
5. Пожарная безопасность.
6. Термическая безопасность.
7. Химическая безопасность.
8. Электрическая безопасность.
9. Электромагнитная совместимость.
10. Единство измерений.
11. Санитарно-эпидемиологическая и экологическая безопасность.

Эти требования направлены на обеспечение безопасной эксплуатации железнодорожного подвижного состава и защиту жизни и
здоровья людей, а также окружающей среды.

Вопрос пользователя: стоп
'''