import math
import time

'''
Задача 1: Генератор степеней
Создайте функцию, которая принимает число (степень) и возвращает замыкание. 
Это замыкание должно принимать другое число (основание) и возводить его в запомненную степень.
'''
def degree(deg):
    def foundation(fnd): 
        nonlocal deg;
        return math.pow(fnd, deg)
    return foundation

fnd = degree(3)
print(fnd(2))

'''
Задача 2: Использование декоратора
Напишите декоратор execution_timer, который измеряет время выполнения функции. 
Примените его к функции, вычисляющей сумму квадратов чисел от 1 до 1 000 000.
'''
def execution_timer(func):
    def wrapper(*args):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Время выполнения функции '{func.__name__}': {execution_time:.4f} секунд")
        print(f"Результат: {result}")
    return wrapper


@execution_timer
def square_sum(count: int):
    sum = 0;
    for i in range(count):
        sum += i * i
    return sum

square_sum(1000000)

