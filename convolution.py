import numpy as np
"""
from style_f import *
from generationSignal import *
"""
#Свертка
def convolution(firstSignal, SecondSignal):
    svertka_func = []
    summ = 0
    SecondSignal = SecondSignal[::-1]
    for i in range (0,len(SecondSignal)):
        firstSignal = np.concatenate([[0],firstSignal,[0]])

    for j in range(0, len(firstSignal)-len(SecondSignal)):
        for i in range(0, len(SecondSignal)):
            summ += firstSignal[i + j] * SecondSignal[i]
        svertka_func.append(summ)
        summ = 0
    svertka_func = svertka_func/max(svertka_func)
    return svertka_func
#Корреляция
def correlation(firstSignal, SecondSignal):
    correlation_func = []
    summ = 0
    for i in range (0,len(SecondSignal)):
        firstSignal = np.concatenate([[0],firstSignal,[0]])
    for j in range(0, len(firstSignal)-len(SecondSignal)):
        for i in range(0, len(SecondSignal)):
            summ += firstSignal[i + j] * SecondSignal[i]
        correlation_func.append(summ)
        summ = 0
    correlation_func = correlation_func/max(correlation_func)
    return correlation_func
#Пример:
"""
t, pulse = rectangular_pulse(start=0,
                             duration=100,
                             amplitude=1,
                             t_start=0,
                             t_end=2,
                             fs=2000)

#t, pulse = triangular_pulse(peak_time=0.5,t_start =0,t_end=2, A=1,fs=2000)
t1, pulse1 =  gaussian_pulse(μ=1, σ=0.3, A=1, t_start=0, t_end=2, fs=2000)

#a=convolution(pulse,pulse1)
#b=correlation(pulse,pulse1)

pm = PlotManager()

data_dict = {
        'y1': pulse,
        #'y2': b
    }
df = pm.load_data(data=data_dict)
# Построение графиков с автоматическими точками по оси X
pm.plot(
        data=df, #None, для скопированных данных
        x_column=None,  # None, для автоматического расположения точек, 0 для скопированных данных
        y_column=None,
        labels={'y1': 'Сигнал', 'y2': 'График 2'},
        title='Прямоугольный сигнал',
        xlabel='Номер точки',
        ylabel='Значение',
        marker=None,
        color='Blue',
        linewidth=1.5,
        xstart=None,
        ystart=None,
        xticks=None,
        yticks=None,

        ask_input=False #True, если надо вводить скопированные столбцы
    )
"""