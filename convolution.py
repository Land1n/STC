import numpy as np

from style_f import *
from generationSignal import *

#Свертка
def convolution(firstSignal, SecondSignal, fs):
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
    timeline = np.linspace(-1 * len(SecondSignal) / fs, (len(firstSignal)) / fs, fs * 2)
    return timeline,svertka_func
#Корреляция
def correlation(firstSignal, SecondSignal,fs):
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
    timeline = np.linspace(-1*len(SecondSignal)/fs, (len(firstSignal))/fs, fs*2)
    return timeline, correlation_func
#Пример:
"""
t, pulse = rectangular_pulse(start=-0.5,
                             duration=1,
                             amplitude=1,
                             t_start=-1,
                             t_end=1,
                             fs=1000)

t1, pulse1 = rectangular_pulse(start=0,
                             duration=1,
                             amplitude=1,
                             t_start=-1,
                             t_end=1,
                             fs=1000)
#t, pulse1 = triangular_pulse(peak_time=0.5,t_start =0,t_end=2, A=1,fs=1000)
#t1, pulse1 =  gaussian_pulse(μ=1, σ=0.3, A=1, t_start=0, t_end=2, fs=2000)

#a=convolution(pulse,pulse1)
tX,b=correlation(pulse,pulse1,1000)
print(len(tX))
print(len(b))
pm = PlotManager()

data_dict = {
    'x':tX,
        'y1':  b,
        #'y2': pulse1
    }
df = pm.load_data(data=data_dict)
# Построение графиков с автоматическими точками по оси X
pm.plot(
        data=df, #None, для скопированных данных
        x_column='x',  # None, для автоматического расположения точек, 0 для скопированных данных
        y_column='y1',
        labels={'y1': 'Второй сигнал', 'y2': 'Второй сигнал'},
        title='Сигналы используемые в корреляции',
        xlabel='Время, с',
        ylabel='Значение',
        marker=None,
        color="saddlebrown",
        linewidth=1.5,
        xstart=-1,
        ystart=None,
        xticks=0.25,
        yticks=None,

        ask_input=False #True, если надо вводить скопированные столбцы
    )
"""