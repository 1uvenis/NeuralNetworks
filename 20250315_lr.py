import matplotlib.pyplot as plt
import numpy as np
import math

step: int
e = []

w11 = 1
w12 = 1
w21 = 1
w22 = 1

w1 = 0.001
w2 = 0.001

x_in_value = np.array([[1,1],[1,2],[2,1],[2,2],[1,1]],float)
d_out_value = np.array([1,2,1,1,1],float)
n_step_value = 0.001
a = 1

def neural_search():
    global w11, w12, w21, w22, w1, w2, x_in_value, d_out_value, e
    i:int
    e_error: float
    for i in range(9000):
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

            e_error = ((d_out_value[j] - y) ** 2)/2

            w1 = w1 + n_step_value * e_error*f1
            w2 = w2 + n_step_value * e_error*f2
            w11 = w11 + n_step_value * e_error*x1
            w12 = w12 + n_step_value * e_error*x1
            w21 = w21 + n_step_value * e_error*x2
            w22 = w22 + n_step_value * e_error*x2
            j = j + 1

        e.append(e_error)
        i = i + 1

    plt.plot(range(i), e)
    plt.show()

neural_search()

print('w1 = ',w1)
print('w2 = ',w2)
print('w11 = ',w11)
print('w12 = ',w12)
print('w21 = ',w21)
print('w22 = ',w22)