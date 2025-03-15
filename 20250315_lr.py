import matplotlib.pyplot as plt
import numpy as np
import math

#Ввод начальных значений
step: int
e = []

#Весовые коэффициенты 1 слоя
w11 = 1
w12 = 1
w21 = 1
w22 = 1

#Весовые коэффициенты 2 слоя
w1 = 0.001
w2 = 0.001

x_in_value = np.array([[1,1],[1,2],[2,1],[2,2],[1,1]],float) #Входные параметры
d_out_value = np.array([1,2,1,1,1],float) #Выходные параметры
n_step_value = 0.001 #Изначальная скорость обучения
a = 1

#Функция обучения нейронной сети из 3 нейронов
def neural_search(n):
    global w11, w12, w21, w22, w1, w2, x_in_value, d_out_value, e
    e=[]
    i:int
    e_error: float
    for i in range(2000):
        j:int
        for j in range(np.size(d_out_value)):
            x1 = x_in_value[j,0]
            x2 = x_in_value[j,1]

            s1 = w11*x1 + w21*x2
            s2 = w12*x1 + w22*x2

            f1 = 1/(1+math.exp(-a*s1))
            f2 = 1/(1+math.exp(-a*s2))

            s21 = w1*f1 + w2*f2
            y = 1/(1+math.exp(-a*s21))

            e_error = ((d_out_value[j] - y) ** 2)/2 #Нахождение ошибки

            # Изменение весовых коэффициентов
            w1 = w1 + n * e_error*f1
            w2 = w2 + n * e_error*f2
            w11 = w11 + n * e_error*x1
            w12 = w12 + n * e_error*x1
            w21 = w21 + n * e_error*x2
            w22 = w22 + n * e_error*x2
            j = j + 1

        e.append(e_error) #Добавление ошибки в массив
        i = i + 1

    plt.plot(range(i), e) #Добавление массива ошибок в график


neural_search(0.00001) #Запуск обучения с шагов 0.00001
neural_search(0.0002) #Запуск обучения с шагов 0.0002
neural_search(0.003) #Запуск обучения с шагов 0.003
plt.show() #Вывод графика

print('w1 = ',w1)
print('w2 = ',w2)
print('w11 = ',w11)
print('w12 = ',w12)
print('w21 = ',w21)
print('w22 = ',w22)