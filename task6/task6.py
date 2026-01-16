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

def document_analysis(url, system, topic, verbose=True):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", default='https://api.proxyapi.ru/openai/v1')

    if not api_key:
        print("Ошибка: Переменная окружения OPENAI_API_KEY не найдена.")
        print("Создайте файл .env в корне проекта и добавьте:")
        print("OPENAI_API_KEY=ваш_ключ_здесь")
        return False

    data_from_url= load_document_text(url)
    if (not data_from_url):
        print("Ошибка!")
        return False
   
    source_chunks=[]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)

    for chunk in splitter.split_text(data_from_url):
        source_chunks.append(Document(page_content=chunk, metadata={"meta":"data"}))

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(source_chunks, embeddings)

    docs = db.similarity_search(topic, k=4)
    if verbose: print('\n ===========================================: ')
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    if verbose: print('message_content :\n ======================================== \n', message_content)

    try:
        client = OpenAI(
            api_key=api_key,  
            base_url=base_url 
        )
    except Exception as e:
        print(f"Ошибка инициализации клиента: {e}")
        return False

    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": f"Ответь на вопрос клиента. Не упоминай документ с информацией для ответа клиенту \
         в ответе. Документ с информацией для ответа клиенту: {message_content} \
         \n\nВопрос клиента: \n{topic}"}
    ]
    if verbose: print('\n ===========================================: ')
    try:
        completion = client.chat.completions.create(
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
    
    return format_text(answer)  # возвращает ответ

url = "https://docs.google.com/document/d/1q4l912Re8zuIfBax4FDS3ZppYmVPzER3Si2wrmznddc/edit?tab=t.0"

system = "Ты специалист по промышленной безопасности. \
Не придумывай ничего от себя, отвечай максимально по документу. Не упоминай Документ с информацией для \
ответа клиенту. Клиент ничего не должен знать про Документ с информацией для ответа клиенту"

topic = "Перечисли основные три вида нарушений техники безопаснсти на предприятиях?"
print(document_analysis(url, system, topic, verbose=False))

topic = "Какие возможно наказания а нарушение техники безопасности на предприятиях?"
print(document_analysis(url, system, topic, verbose=False))

topic = "За какие нарушения техники безопасности возможна уголовная ответственность?"
print(document_analysis(url, system, topic, verbose=False))