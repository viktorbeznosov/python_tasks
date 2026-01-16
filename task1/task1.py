'''
Задача 1.
Создайте функцию "list_expert()", которая будет принимать неограниченное количество параметров 
и если передали числа, то вернуть их сумму, а если есть другие типы, то вывети списком их типы. 
Но если ввести 1 параметр, то его же и вывести.

Вывод:
list_expert(1,2,3,4,5)  # 15
list_expert("a", True, [1, 2, 3])  # ['str', 'bool', 'list']
list_expert("obj")  # obj
'''

def list_expert(*args):
    if (len(args) == 1):
        result = args[0]
    elif (len(args) > 1):
        numbersList = list(filter(lambda x: type(x) in [int, float], args))
        if (len(numbersList) < len(args)):
            result = list(map(lambda x: type(x).__name__, args))
        else:
            result = sum(args)

    return result;
print(list_expert(1,2,3,4,5))  # 15
print(list_expert("a", True, [1, 2, 3]))  # ['str', 'bool', 'list']
print(list_expert("obj"))  # obj

'''
Задача 2
Допишите функцию из 1-го задания, чтобы, когда не передали вообще параметров, 
то она вернет строку "Пустая функция".
'''

def list_expert2(*args):
    result = "Пустая функция"

    if (len(args) == 1):
        result = args[0]
    elif (len(args) > 1):
        numbersList = list(filter(lambda x: type(x) in [int, float], args))
        if (len(numbersList) < len(args)):
            result = list(map(lambda x: type(x).__name__, args))
        else:
            result = sum(args)

    return result;
print(list_expert2())

'''
Задача 3: Использование модуля random
Напишите программу, которая генерирует список из 10 случайных чисел от 1 до 100, 
а затем находит их максимум, минимум и среднее.
'''

import random

random_numbers = list(map(lambda x: random.randint(1, 100), range(10)))
print(random_numbers)
print(f"Максимум {max(random_numbers)}")
print(f"Минимум {min(random_numbers)}")
print(f"Среднее {sum(random_numbers) / len(random_numbers)}")



