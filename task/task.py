
def saveToFile(str: str):
    with open('./result.txt', 'w', encoding='utf-8') as file:
        file.write(str)

saveToFile("Задание. Пакет курсов GPT Engineer | Программисты | IDE. Среда разработки")